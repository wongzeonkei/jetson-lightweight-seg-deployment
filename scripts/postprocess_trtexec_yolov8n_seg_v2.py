import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_shape(shape):
    if isinstance(shape, str):
        return tuple(int(x) for x in shape.replace(" ", "").split("x") if x)
    if isinstance(shape, (list, tuple)):
        return tuple(int(x) for x in shape)
    return None


def find_tensors(obj, out):
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("Name")
        values = None
        for k in ["values", "data", "Data"]:
            if k in obj:
                values = obj[k]
                break

        shape = None
        for k in ["dimensions", "shape", "dims"]:
            if k in obj:
                shape = parse_shape(obj[k])
                break

        if name is not None and values is not None and shape is not None:
            arr = np.asarray(values, dtype=np.float32).reshape(shape)
            out[name] = arr

        for v in obj.values():
            find_tensors(v, out)

    elif isinstance(obj, list):
        for item in obj:
            find_tensors(item, out)


def load_trtexec_outputs(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    tensors = {}
    find_tensors(data, tensors)

    if "output0" not in tensors or "output1" not in tensors:
        print("Found tensor names:", list(tensors.keys()))
        raise RuntimeError("Failed to find output0/output1 in TensorRT JSON.")

    return tensors["output0"], tensors["output1"]


def load_meta(meta_path, image):
    h, w = image.shape[:2]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    scale = meta.get("scale", meta.get("ratio", None))
    pad_x = meta.get("pad_x", meta.get("dw", 0))
    pad_y = meta.get("pad_y", meta.get("dh", 0))

    orig_w = meta.get("orig_w", meta.get("original_width", meta.get("width", w)))
    orig_h = meta.get("orig_h", meta.get("original_height", meta.get("height", h)))

    if "orig_shape" in meta and isinstance(meta["orig_shape"], list):
        orig_h, orig_w = meta["orig_shape"][:2]
    if "original_shape" in meta and isinstance(meta["original_shape"], list):
        orig_h, orig_w = meta["original_shape"][:2]

    if scale is None:
        scale = min(320 / orig_h, 320 / orig_w)

    return {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
    }


def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms_class_aware(boxes, scores, classes, iou_thres):
    keep = []

    for cls in np.unique(classes):
        idxs = np.where(classes == cls)[0]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]

        while len(idxs) > 0:
            cur = idxs[0]
            keep.append(cur)

            if len(idxs) == 1:
                break

            ious = box_iou(boxes[cur], boxes[idxs[1:]])
            idxs = idxs[1:][ious <= iou_thres]

    keep = np.asarray(keep, dtype=np.int32)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return keep


def deletterbox_boxes(boxes, meta):
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale

    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h - 1)
    return out


def reconstruct_mask(coeff, proto, box_letterbox, meta, input_size=320, mask_thres=0.5):
    # proto: [32, 80, 80]
    c, mh, mw = proto.shape
    mask = sigmoid(np.matmul(coeff, proto.reshape(c, -1))).reshape(mh, mw)

    # resize prototype mask to model input size
    mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # crop by bbox in letterbox coordinates
    x1, y1, x2, y2 = box_letterbox.astype(np.int32)
    x1 = int(np.clip(x1, 0, input_size - 1))
    y1 = int(np.clip(y1, 0, input_size - 1))
    x2 = int(np.clip(x2, 0, input_size - 1))
    y2 = int(np.clip(y2, 0, input_size - 1))

    cropped = np.zeros_like(mask)
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    # remove letterbox padding
    pad_x = int(round(meta["pad_x"]))
    pad_y = int(round(meta["pad_y"]))

    unpad = cropped[
        pad_y:input_size - pad_y if pad_y > 0 else input_size,
        pad_x:input_size - pad_x if pad_x > 0 else input_size,
    ]

    mask_orig = cv2.resize(
        unpad,
        (meta["orig_w"], meta["orig_h"]),
        interpolation=cv2.INTER_LINEAR,
    )

    return (mask_orig > mask_thres).astype(np.uint8)


def visualize(image, detections, masks):
    vis = image.copy()
    union_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for det, mask in zip(detections, masks):
        cls = det["class_id"]
        score = det["score"]
        x1, y1, x2, y2 = det["box"]

        color = (
            int((37 * cls + 50) % 255),
            int((17 * cls + 120) % 255),
            int((29 * cls + 200) % 255),
        )

        union_mask[mask > 0] = 255

        overlay = vis.copy()
        overlay[mask > 0] = (
            0.5 * overlay[mask > 0] + 0.5 * np.array(color)
        ).astype(np.uint8)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_NAMES[cls]} {score:.3f}"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis, union_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="trtexec exported output json")
    parser.add_argument("--meta", required=True, help="preprocess meta json")
    parser.add_argument("--image", required=True, help="original image path")
    parser.add_argument("--out", required=True, help="output visualization path")
    parser.add_argument("--mask_out", default=None, help="output union mask path")
    parser.add_argument("--npy_prefix", default=None, help="prefix for saving output0/output1 npy")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--mask_thres", type=float, default=0.5)
    args = parser.parse_args()

    output0, output1 = load_trtexec_outputs(args.json)

    print("Parsed TensorRT outputs:")
    print(" output0:", output0.shape, output0.dtype, "min/max:", float(output0.min()), float(output0.max()))
    print(" output1:", output1.shape, output1.dtype, "min/max:", float(output1.min()), float(output1.max()))

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    meta = load_meta(args.meta, image)

    pred = output0[0].transpose(1, 0)  # [2100, 116]
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:84]
    coeffs = pred[:, 84:116]

    classes = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]

    valid = scores >= args.conf
    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    classes = classes[valid]
    coeffs = coeffs[valid]

    boxes_letterbox = xywh_to_xyxy(boxes_xywh)
    boxes_letterbox = np.clip(boxes_letterbox, 0, 320)

    keep = nms_class_aware(boxes_letterbox, scores, classes, args.iou)

    boxes_letterbox = boxes_letterbox[keep]
    boxes_orig = deletterbox_boxes(boxes_letterbox, meta)
    scores = scores[keep]
    classes = classes[keep]
    coeffs = coeffs[keep]

    proto = output1[0]

    detections = []
    masks = []

    for i in range(len(scores)):
        mask = reconstruct_mask(
            coeffs[i],
            proto,
            boxes_letterbox[i],
            meta,
            input_size=320,
            mask_thres=args.mask_thres,
        )

        x1, y1, x2, y2 = boxes_orig[i].round().astype(int).tolist()
        det = {
            "class_id": int(classes[i]),
            "class_name": COCO_NAMES[int(classes[i])],
            "score": float(scores[i]),
            "box": [x1, y1, x2, y2],
            "mask_pixels": int(mask.sum()),
        }

        detections.append(det)
        masks.append(mask)

    print("\nDetections:")
    for det in detections:
        print(det)

    vis, union_mask = visualize(image, detections, masks)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, vis)
    print("\nSaved visualization:", args.out)

    if args.mask_out is None:
        args.mask_out = str(Path(args.out).with_suffix("")) + "_mask.png"

    cv2.imwrite(args.mask_out, union_mask)
    print("Saved mask:", args.mask_out)

    if args.npy_prefix is not None:
        Path(args.npy_prefix).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.npy_prefix + "_output0.npy", output0)
        np.save(args.npy_prefix + "_output1.npy", output1)
        print("Saved npy:")
        print(" ", args.npy_prefix + "_output0.npy")
        print(" ", args.npy_prefix + "_output1.npy")


if __name__ == "__main__":
    main()
