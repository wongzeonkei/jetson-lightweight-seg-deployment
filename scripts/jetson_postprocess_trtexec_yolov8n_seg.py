import os
import json
import cv2
import numpy as np


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def flatten_numeric(obj):
    vals = []
    if isinstance(obj, (int, float)):
        vals.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(flatten_numeric(x))
    elif isinstance(obj, dict):
        for key in ("value", "val", "data", "values", "Values"):
            if key in obj:
                vals.extend(flatten_numeric(obj[key]))
                return vals
        for v in obj.values():
            vals.extend(flatten_numeric(v))
    return vals


def extract_named_tensor(data, name, shape):
    candidates = []

    def visit(obj):
        if isinstance(obj, dict):
            obj_name = str(obj.get("name", obj.get("Name", obj.get("binding", ""))))
            if obj_name == name or obj_name.endswith(name):
                for key in ("values", "Values", "data", "Data"):
                    if key in obj:
                        candidates.append(obj[key])
            if name in obj:
                candidates.append(obj[name])
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(data)

    expected = int(np.prod(shape))

    for cand in candidates:
        arr = np.array(flatten_numeric(cand), dtype=np.float32)
        if arr.size == expected:
            return arr.reshape(shape)

    raise RuntimeError("Failed to extract tensor {} with shape {}".format(name, shape))


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


def nms_boxes(boxes, scores, conf_thres=0.25, iou_thres=0.45):
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

    if len(indices) == 0:
        return []

    if isinstance(indices, tuple):
        return []

    indices = np.array(indices).reshape(-1).tolist()
    return indices


def postprocess(output0, output1, original, scale, pad_x, pad_y,
                conf_thres=0.25, iou_thres=0.45, input_size=320):
    pred = output0[0].transpose(1, 0)

    boxes_xywh = pred[:, 0:4]
    class_scores = pred[:, 4:84]
    mask_coeffs = pred[:, 84:116]

    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    keep = scores >= conf_thres
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    boxes = []
    boxes_float = []

    for box in boxes_xywh:
        cx, cy, w, h = box
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1c = max(0.0, min(float(x1), float(input_size)))
        y1c = max(0.0, min(float(y1), float(input_size)))
        x2c = max(0.0, min(float(x2), float(input_size)))
        y2c = max(0.0, min(float(y2), float(input_size)))

        boxes_float.append([x1c, y1c, x2c, y2c])
        boxes.append([int(round(x1c)), int(round(y1c)), int(round(x2c - x1c)), int(round(y2c - y1c))])

    indices = nms_boxes(boxes, scores.tolist(), conf_thres, iou_thres)

    proto = output1[0]
    proto_flat = proto.reshape(32, -1)

    vis = original.copy()
    h0, w0 = original.shape[:2]
    detections = []

    for idx in indices:
        coeff = mask_coeffs[idx]
        mask = sigmoid(np.dot(coeff, proto_flat)).reshape(80, 80)

        x1, y1, x2, y2 = boxes_float[idx]

        mask320 = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        crop = np.zeros_like(mask320)
        ix1 = max(0, min(int(x1), input_size - 1))
        iy1 = max(0, min(int(y1), input_size - 1))
        ix2 = max(0, min(int(x2), input_size))
        iy2 = max(0, min(int(y2), input_size))

        if ix2 > ix1 and iy2 > iy1:
            crop[iy1:iy2, ix1:ix2] = mask320[iy1:iy2, ix1:ix2]

        mask_bin = (crop > 0.5).astype(np.uint8)

        x_unpad1 = int(pad_x)
        y_unpad1 = int(pad_y)
        x_unpad2 = int(input_size - pad_x)
        y_unpad2 = int(input_size - pad_y)

        mask_unpad = mask_bin[y_unpad1:y_unpad2, x_unpad1:x_unpad2]
        mask_orig = cv2.resize(mask_unpad, (w0, h0), interpolation=cv2.INTER_NEAREST)

        bx1 = int(max(0, min((x1 - pad_x) / scale, w0 - 1)))
        by1 = int(max(0, min((y1 - pad_y) / scale, h0 - 1)))
        bx2 = int(max(0, min((x2 - pad_x) / scale, w0 - 1)))
        by2 = int(max(0, min((y2 - pad_y) / scale, h0 - 1)))

        cls_id = int(class_ids[idx])
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        score = float(scores[idx])

        color = (
            int((37 * (cls_id + 1)) % 255),
            int((17 * (cls_id + 3)) % 255),
            int((29 * (cls_id + 5)) % 255)
        )

        colored = np.zeros_like(vis)
        colored[mask_orig > 0] = color
        vis = cv2.addWeighted(vis, 1.0, colored, 0.45, 0.0)

        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            vis,
            "{} {:.2f}".format(cls_name, score),
            (bx1, max(0, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        det = {
            "class_id": cls_id,
            "class_name": cls_name,
            "score": score,
            "box": [bx1, by1, bx2, by2],
            "mask_pixels": int(mask_orig.sum())
        }
        detections.append(det)

    return vis, detections


def main():
    json_path = "outputs/tensorrt/yolov8n_seg_320_fp16_jetson_bus_output.json"
    image_path = "data/samples/bus.jpg"
    output_path = "outputs/tensorrt/yolov8n_seg_320_bus_jetson_fp16.jpg"

    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(json_path, "r") as f:
        data = json.load(f)

    output0 = extract_named_tensor(data, "output0", (1, 116, 2100))
    output1 = extract_named_tensor(data, "output1", (1, 32, 80, 80))

    original = cv2.imread(image_path)
    if original is None:
        raise RuntimeError("Failed to read image: " + image_path)

    _, scale, pad_x, pad_y = letterbox(original, 320)

    vis, detections = postprocess(
        output0,
        output1,
        original,
        scale,
        pad_x,
        pad_y,
        conf_thres=0.25,
        iou_thres=0.45,
        input_size=320
    )

    cv2.imwrite(output_path, vis)

    print("Parsed Jetson TensorRT outputs:")
    print(" output0:", output0.shape, output0.dtype, "min/max:", float(output0.min()), float(output0.max()))
    print(" output1:", output1.shape, output1.dtype, "min/max:", float(output1.min()), float(output1.max()))

    print("\nDetections:")
    for d in detections:
        print(d)

    print("\nSaved visualization:", output_path)


if __name__ == "__main__":
    main()
