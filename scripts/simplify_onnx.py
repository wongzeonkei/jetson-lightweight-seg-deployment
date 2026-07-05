import os
import onnx
from onnxsim import simplify

src = "models/onnx/yolov8n_seg_320_opset13.onnx"
dst = "models/onnx/yolov8n_seg_320_opset13_sim.onnx"

if not os.path.exists(src):
    raise FileNotFoundError(src)

print("Loading:", src)
model = onnx.load(src)

print("Simplifying ONNX...")
model_simp, check = simplify(
    model,
    overwrite_input_shapes={"images": [1, 3, 320, 320]}
)

if not check:
    raise RuntimeError("ONNX simplifier check failed")

onnx.save(model_simp, dst)

print("Simplified ONNX saved to:", dst)
print("Original size MB:", os.path.getsize(src) / 1024 / 1024)
print("Simplified size MB:", os.path.getsize(dst) / 1024 / 1024)

onnx.checker.check_model(model_simp)
print("ONNX checker passed for simplified model.")
