import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def letterbox(image, new_shape=320, color=(114, 114, 114)):
    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - new_w
    pad_h = new_shape - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, r, left, top


def preprocess_image(image_path, input_size=320):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image, _, _, _ = letterbox(image, input_size)

    # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image)
    return image


class ImageCalibBatcher:
    def __init__(self, calib_dir, batch_size=1, input_size=320):
        self.calib_dir = Path(calib_dir)
        self.batch_size = batch_size
        self.input_size = input_size

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(sorted(self.calib_dir.glob(ext)))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No calibration images found in: {calib_dir}")

        self.index = 0
        print(f"[CalibBatcher] Found {len(self.image_paths)} calibration images.")
        print(f"[CalibBatcher] batch_size={batch_size}, input_size={input_size}")

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.index:self.index + self.batch_size]
        if len(batch_paths) < self.batch_size:
            return None

        batch = np.zeros(
            (self.batch_size, 3, self.input_size, self.input_size),
            dtype=np.float32,
        )

        for i, img_path in enumerate(batch_paths):
            batch[i] = preprocess_image(img_path, self.input_size)

        self.index += self.batch_size
        return np.ascontiguousarray(batch)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batcher, cache_file):
        super().__init__()
        self.batcher = batcher
        self.cache_file = cache_file
        self.current_batch = None

        dummy = np.zeros(
            (batcher.batch_size, 3, batcher.input_size, batcher.input_size),
            dtype=np.float32,
        )
        self.device_input = cuda.mem_alloc(dummy.nbytes)
        self.batch_nbytes = dummy.nbytes

    def get_batch_size(self):
        return self.batcher.batch_size

    def get_batch(self, names):
        batch = self.batcher.next_batch()
        if batch is None:
            return None

        self.current_batch = batch
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Using calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[Calibrator] Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(
    onnx_path,
    engine_path,
    calib_dir,
    cache_file,
    input_size=320,
    batch_size=1,
    workspace_gb=2,
):
    onnx_path = str(onnx_path)
    engine_path = str(engine_path)
    cache_file = str(cache_file)

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    print("[Build] ONNX:", onnx_path)
    print("[Build] Engine:", engine_path)
    print("[Build] Calib dir:", calib_dir)
    print("[Build] Cache:", cache_file)

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    workspace_bytes = int(workspace_gb * (1 << 30))
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except Exception:
        config.max_workspace_size = workspace_bytes

    if not builder.platform_has_fast_int8:
        print("[Warning] This platform does not report fast INT8 support.")

    config.set_flag(trt.BuilderFlag.INT8)

    # Keep FP16 enabled as fallback when some layers cannot run INT8.
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    batcher = ImageCalibBatcher(
        calib_dir=calib_dir,
        batch_size=batch_size,
        input_size=input_size,
    )
    calibrator = EntropyCalibrator(batcher=batcher, cache_file=cache_file)
    config.int8_calibrator = calibrator

    with open(onnx_path, "rb") as f:
        model = f.read()

    print("[Build] Parsing ONNX...")
    if not parser.parse(model):
        print("[Error] Failed to parse ONNX.")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed.")

    print("[Network] Inputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {i}: name={inp.name}, shape={inp.shape}, dtype={inp.dtype}")

    print("[Network] Outputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {i}: name={out.name}, shape={out.shape}, dtype={out.dtype}")

    print("[Build] Building INT8 engine. This may take several minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build INT8 engine.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print("[Build] INT8 engine saved:", engine_path)
    print("[Build] Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        default="models/onnx/yolov8n_seg_320_opset13_sim.onnx",
    )
    parser.add_argument(
        "--engine",
        default="deploy/tensorrt/engines/yolov8n_seg_320_int8.engine",
    )
    parser.add_argument(
        "--calib_dir",
        default="data/calib/yolov8n_seg/images",
    )
    parser.add_argument(
        "--cache",
        default="deploy/tensorrt/engines/yolov8n_seg_320_int8_calib.cache",
    )
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workspace_gb", type=float, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_int8_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        calib_dir=args.calib_dir,
        cache_file=args.cache,
        input_size=args.input_size,
        batch_size=args.batch_size,
        workspace_gb=args.workspace_gb,
    )
