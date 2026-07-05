import os
import json
import importlib.util
import numpy as np

# Load preprocess function from ONNXRuntime script
spec = importlib.util.spec_from_file_location(
    "ort_yolo_seg",
    "scripts/infer_onnxruntime_yolov8n_seg.py"
)
ort_yolo_seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ort_yolo_seg)

image_path = "data/samples/bus.jpg"
out_dir = "outputs/tensorrt/input_bins"
bin_path = os.path.join(out_dir, "bus_images_fp32_1x3x320x320.bin")
meta_path = os.path.join(out_dir, "bus_images_meta.json")

os.makedirs(out_dir, exist_ok=True)

original, x, scale, pad_x, pad_y = ort_yolo_seg.preprocess(image_path, 320)

x = np.ascontiguousarray(x.astype(np.float32))
x.tofile(bin_path)

meta = {
    "image_path": image_path,
    "input_name": "images",
    "input_shape": list(x.shape),
    "dtype": "float32",
    "scale": float(scale),
    "pad_x": float(pad_x),
    "pad_y": float(pad_y),
    "original_shape": list(original.shape)
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Saved input bin:", bin_path)
print("Saved meta:", meta_path)
print("Input shape:", x.shape)
print("Input dtype:", x.dtype)
print("Input min/max:", float(x.min()), float(x.max()))
print("Scale:", scale, "pad_x:", pad_x, "pad_y:", pad_y)
