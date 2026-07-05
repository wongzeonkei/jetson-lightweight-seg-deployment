import os
import shutil
from ultralytics import YOLO

root = os.getcwd()
pt_path = os.path.join(root, "models/pytorch/yolov8n-seg.pt")
onnx_dst = os.path.join(root, "models/onnx/yolov8n_seg_320_opset13.onnx")

os.makedirs(os.path.dirname(onnx_dst), exist_ok=True)

if not os.path.exists(pt_path):
    raise FileNotFoundError(f"Missing weight: {pt_path}")

model = YOLO(pt_path)

exported = model.export(
    format="onnx",
    imgsz=320,
    opset=13,
    simplify=False,
    dynamic=False,
    nms=False
)

print("Ultralytics exported:", exported)

possible_paths = [
    os.path.join(root, "models/pytorch/yolov8n-seg.onnx"),
    os.path.join(root, "yolov8n-seg.onnx"),
    str(exported)
]

src = None
for p in possible_paths:
    if os.path.exists(p):
        src = p
        break

if src is None:
    raise FileNotFoundError(f"Cannot find exported ONNX. Checked: {possible_paths}")

if os.path.abspath(src) != os.path.abspath(onnx_dst):
    shutil.move(src, onnx_dst)

print("Saved ONNX to:", onnx_dst)
