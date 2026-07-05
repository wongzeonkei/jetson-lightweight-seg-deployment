import os
import time
import importlib.util
import numpy as np
import onnxruntime as ort


spec = importlib.util.spec_from_file_location(
    "ort_yolo_seg",
    "scripts/infer_onnxruntime_yolov8n_seg.py"
)
ort_yolo_seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ort_yolo_seg)


def summarize_diff(name, a, b):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    print(f"\n{name} diff:")
    print(" shape:", diff.shape)
    print(" max_abs:", float(diff.max()))
    print(" mean_abs:", float(diff.mean()))
    print(" median_abs:", float(np.median(diff)))
    print(" p95_abs:", float(np.percentile(diff, 95)))
    print(" p99_abs:", float(np.percentile(diff, 99)))


def main():
    onnx_path = "models/onnx/yolov8n_seg_320_opset13_sim.onnx"
    image_path = "data/samples/bus.jpg"

    trt_output0_path = "outputs/tensorrt/trt_fp16_output0.npy"
    trt_output1_path = "outputs/tensorrt/trt_fp16_output1.npy"

    if not os.path.exists(trt_output0_path) or not os.path.exists(trt_output1_path):
        raise FileNotFoundError("Missing TensorRT npy outputs. Run postprocess_trtexec_yolov8n_seg.py first.")

    original, x, scale, pad_x, pad_y = ort_yolo_seg.preprocess(image_path, 320)

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    print("Available providers:", available)
    print("Requested providers:", providers)

    sess = ort.InferenceSession(onnx_path, providers=providers)
    print("Session providers:", sess.get_providers())

    for _ in range(10):
        ort_outputs = sess.run(None, {"images": x})

    times = []
    for _ in range(100):
        t0 = time.time()
        ort_outputs = sess.run(None, {"images": x})
        times.append((time.time() - t0) * 1000)

    ort_output0, ort_output1 = ort_outputs
    trt_output0 = np.load(trt_output0_path)
    trt_output1 = np.load(trt_output1_path)

    os.makedirs("outputs/compare", exist_ok=True)
    np.save("outputs/compare/ort_output0.npy", ort_output0)
    np.save("outputs/compare/ort_output1.npy", ort_output1)

    print("\nORT latency:")
    print(" mean ms:", float(np.mean(times)))
    print(" median ms:", float(np.median(times)))
    print(" min/max ms:", float(np.min(times)), float(np.max(times)))

    print("\nOutput shapes:")
    print(" ORT output0:", ort_output0.shape)
    print(" TRT output0:", trt_output0.shape)
    print(" ORT output1:", ort_output1.shape)
    print(" TRT output1:", trt_output1.shape)

    summarize_diff("output0", ort_output0, trt_output0)
    summarize_diff("output1", ort_output1, trt_output1)

    _, ort_det = ort_yolo_seg.postprocess(
        ort_output0, ort_output1, original, scale, pad_x, pad_y,
        conf_thres=0.25, iou_thres=0.45, input_size=320
    )

    _, trt_det = ort_yolo_seg.postprocess(
        trt_output0, trt_output1, original, scale, pad_x, pad_y,
        conf_thres=0.25, iou_thres=0.45, input_size=320
    )

    print("\nORT detections:")
    for d in ort_det:
        print(d)

    print("\nTRT detections:")
    for d in trt_det:
        print(d)

    print("\nDetection count:")
    print(" ORT:", len(ort_det))
    print(" TRT:", len(trt_det))

    print("\nSaved ORT outputs:")
    print(" outputs/compare/ort_output0.npy")
    print(" outputs/compare/ort_output1.npy")


if __name__ == "__main__":
    main()
