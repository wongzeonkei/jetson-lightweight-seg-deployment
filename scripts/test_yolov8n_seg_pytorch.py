import os
from ultralytics import YOLO

root = os.getcwd()
weight_path = os.path.join(root, "models/pytorch/yolov8n-seg.pt")
image_path = os.path.join(root, "data/samples/bus.jpg")
project_dir = os.path.join(root, "outputs/pytorch")

if not os.path.exists(weight_path):
    raise FileNotFoundError(f"Missing weight: {weight_path}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Missing image: {image_path}")

model = YOLO(weight_path)

results = model.predict(
    source=image_path,
    imgsz=320,
    conf=0.25,
    save=True,
    project=project_dir,
    name="yolov8n_seg_320",
    exist_ok=True
)

print("PyTorch YOLOv8n-seg inference done.")
print("Weights:", weight_path)
print("Image:", image_path)
print("Saved results to:", os.path.join(project_dir, "yolov8n_seg_320"))
