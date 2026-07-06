import os
import json
import cv2
import numpy as np


def letterbox(img, new_shape=320, color=(114, 114, 114)):
    h, w = img.shape[:2]

    scale = min(float(new_shape) / float(w), float(new_shape) / float(h))

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    dw = new_shape - new_w
    dh = new_shape - new_h

    left = int(round(dw / 2.0 - 0.1))
    right = int(round(dw / 2.0 + 0.1))
    top = int(round(dh / 2.0 - 0.1))
    bottom = int(round(dh / 2.0 + 0.1))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top


def preprocess(image_path, input_size=320):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image: " + image_path)

    padded, scale, pad_x, pad_y = letterbox(img, input_size)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(x, dtype=np.float32)

    return img, x, scale, pad_x, pad_y


def main():
    image_path = "data/samples/bus.jpg"
    out_dir = "outputs/tensorrt/input_bins"
    bin_path = os.path.join(out_dir, "bus_images_fp32_1x3x320x320.bin")
    meta_path = os.path.join(out_dir, "bus_images_meta.json")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    original, x, scale, pad_x, pad_y = preprocess(image_path, 320)
    x.tofile(bin_path)

    meta = {
        "image_path": image_path,
        "input_name": "images",
        "input_shape": list(x.shape),
        "dtype": "float32",
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "original_shape": list(original.shape)
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved input bin:", bin_path)
    print("Saved meta:", meta_path)
    print("Input shape:", x.shape)
    print("Input dtype:", x.dtype)
    print("Input min/max:", float(x.min()), float(x.max()))
    print("Scale:", scale, "pad_x:", pad_x, "pad_y:", pad_y)


if __name__ == "__main__":
    main()
