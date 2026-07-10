from pathlib import Path
import cv2
import numpy as np

src = Path("data/samples/bus.jpg")
out_dir = Path("data/calib/yolov8n_seg_toy/images")
out_dir.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(src))
if img is None:
    raise RuntimeError(f"Failed to read {src}")

h, w = img.shape[:2]

idx = 0

def save(im, name):
    global idx
    path = out_dir / f"{idx:03d}_{name}.jpg"
    cv2.imwrite(str(path), im)
    idx += 1

save(img, "orig")

# brightness / contrast variants
for alpha in [0.75, 0.9, 1.1, 1.25]:
    for beta in [-25, 0, 25]:
        aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        save(aug, f"alpha{alpha}_beta{beta}")

# horizontal flip
save(cv2.flip(img, 1), "flip")

# center crops with resizing back
for scale in [0.75, 0.85, 0.95]:
    nh, nw = int(h * scale), int(w * scale)
    y1 = (h - nh) // 2
    x1 = (w - nw) // 2
    crop = img[y1:y1+nh, x1:x1+nw]
    crop = cv2.resize(crop, (w, h))
    save(crop, f"crop{scale}")

# shifted crops
for dx, dy in [(-80, 0), (80, 0), (0, -60), (0, 60)]:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    save(shifted, f"shift_{dx}_{dy}")

print(f"Saved {idx} calibration images to {out_dir}")
