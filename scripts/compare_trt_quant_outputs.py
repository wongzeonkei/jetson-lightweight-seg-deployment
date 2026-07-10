import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from postprocess_trtexec_yolov8n_seg_v2 import (
    COCO_NAMES,
    load_meta,
    xywh_to_xyxy,
    deletterbox_boxes,
    nms_class_aware,
    reconstruct_mask,
)


def box_iou_single(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def mask_iou_dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    total = a.sum() + b.sum()

    iou = inter / (union + 1e-6)
    dice = 2 * inter / (total + 1e-6)
    return float(iou), float(dice)


def decode_outputs(output0, output1, image, meta, conf=0.25, iou=0.45, mask_thres=0.5):
    pred = output0[0].transpose(1, 0)

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:84]
    coeffs = pred[:, 84:116]

    classes = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]

    valid = scores >= conf
    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    classes = classes[valid]
    coeffs = coeffs[valid]

    boxes_letterbox = xywh_to_xyxy(boxes_xywh)
    boxes_letterbox = np.clip(boxes_letterbox, 0, 320)

    keep = nms_class_aware(boxes_letterbox, scores, classes, iou)

    boxes_letterbox = boxes_letterbox[keep]
    boxes_orig = deletterbox_boxes(boxes_letterbox, meta)
    scores = scores[keep]
    classes = classes[keep]
    coeffs = coeffs[keep]

    proto = output1[0]

    dets = []
    masks = []

    for i in range(len(scores)):
        mask = reconstruct_mask(
            coeffs[i],
            proto,
            boxes_letterbox[i],
            meta,
            input_size=320,
            mask_thres=mask_thres,
        )

        x1, y1, x2, y2 = boxes_orig[i].round().astype(int).tolist()

        dets.append({
            "class_id": int(classes[i]),
            "class_name": COCO_NAMES[int(classes[i])],
            "score": float(scores[i]),
            "box": [x1, y1, x2, y2],
            "mask_pixels": int(mask.sum()),
        })
        masks.append(mask)

    return dets, masks


def load_pair(prefix):
    output0 = np.load(prefix + "_output0.npy")
    output1 = np.load(prefix + "_output1.npy")
    return output0, output1


def match_to_reference(ref_dets, ref_masks, cmp_dets, cmp_masks):
    rows = []
    used = set()

    for i, ref in enumerate(ref_dets):
        best_j = -1
        best_iou = -1.0

        for j, cmp_det in enumerate(cmp_dets):
            if j in used:
                continue

            # Prefer same class, but allow mismatch to reveal class confusion.
            iou = box_iou_single(ref["box"], cmp_det["box"])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0:
            used.add(best_j)
            cmp_det = cmp_dets[best_j]
            cmp_mask = cmp_masks[best_j]

            miou, mdice = mask_iou_dice(ref_masks[i], cmp_mask)

            rows.append({
                "ref_index": i,
                "ref_class": ref["class_name"],
                "cmp_class": cmp_det["class_name"],
                "class_match": int(ref["class_id"] == cmp_det["class_id"]),
                "ref_score": ref["score"],
                "cmp_score": cmp_det["score"],
                "score_diff": cmp_det["score"] - ref["score"],
                "box_iou": best_iou,
                "mask_iou": miou,
                "mask_dice": mdice,
                "ref_mask_pixels": ref["mask_pixels"],
                "cmp_mask_pixels": cmp_det["mask_pixels"],
                "mask_pixels_diff": cmp_det["mask_pixels"] - ref["mask_pixels"],
                "ref_box": ref["box"],
                "cmp_box": cmp_det["box"],
            })
        else:
            rows.append({
                "ref_index": i,
                "ref_class": ref["class_name"],
                "cmp_class": "missing",
                "class_match": 0,
                "ref_score": ref["score"],
                "cmp_score": 0.0,
                "score_diff": -ref["score"],
                "box_iou": 0.0,
                "mask_iou": 0.0,
                "mask_dice": 0.0,
                "ref_mask_pixels": ref["mask_pixels"],
                "cmp_mask_pixels": 0,
                "mask_pixels_diff": -ref["mask_pixels"],
                "ref_box": ref["box"],
                "cmp_box": [],
            })

    # Extra detections not matched to FP16 reference
    for j, cmp_det in enumerate(cmp_dets):
        if j not in used:
            rows.append({
                "ref_index": "extra",
                "ref_class": "none",
                "cmp_class": cmp_det["class_name"],
                "class_match": 0,
                "ref_score": 0.0,
                "cmp_score": cmp_det["score"],
                "score_diff": cmp_det["score"],
                "box_iou": 0.0,
                "mask_iou": 0.0,
                "mask_dice": 0.0,
                "ref_mask_pixels": 0,
                "cmp_mask_pixels": cmp_det["mask_pixels"],
                "mask_pixels_diff": cmp_det["mask_pixels"],
                "ref_box": [],
                "cmp_box": cmp_det["box"],
            })

    return rows


def summarize(rows):
    valid_rows = [r for r in rows if r["ref_index"] != "extra"]

    if len(valid_rows) == 0:
        return {}

    return {
        "matched_ref_count": len(valid_rows),
        "class_match_count": sum(r["class_match"] for r in valid_rows),
        "mean_score_diff": float(np.mean([r["score_diff"] for r in valid_rows])),
        "mean_box_iou": float(np.mean([r["box_iou"] for r in valid_rows])),
        "mean_mask_iou": float(np.mean([r["mask_iou"] for r in valid_rows])),
        "mean_mask_dice": float(np.mean([r["mask_dice"] for r in valid_rows])),
        "extra_detection_count": sum(1 for r in rows if r["ref_index"] == "extra"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/samples/bus.jpg")
    parser.add_argument("--meta", default="outputs/tensorrt/input_bins/bus_images_meta.json")
    parser.add_argument("--fp16_prefix", default="outputs/tensorrt/trt_fp16")
    parser.add_argument("--int8_prefix", default="outputs/tensorrt/trt_int8")
    parser.add_argument("--int8_toy_prefix", default="outputs/tensorrt/trt_int8_toycalib")
    parser.add_argument("--out_csv", default="docs/benchmarks/int8/fp16_int8_consistency.csv")
    parser.add_argument("--out_summary", default="docs/benchmarks/int8/fp16_int8_consistency_summary.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--mask_thres", type=float, default=0.5)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    meta = load_meta(args.meta, image)

    fp16_o0, fp16_o1 = load_pair(args.fp16_prefix)
    int8_o0, int8_o1 = load_pair(args.int8_prefix)
    toy_o0, toy_o1 = load_pair(args.int8_toy_prefix)

    fp16_dets, fp16_masks = decode_outputs(fp16_o0, fp16_o1, image, meta, args.conf, args.iou, args.mask_thres)
    int8_dets, int8_masks = decode_outputs(int8_o0, int8_o1, image, meta, args.conf, args.iou, args.mask_thres)
    toy_dets, toy_masks = decode_outputs(toy_o0, toy_o1, image, meta, args.conf, args.iou, args.mask_thres)

    print("FP16 detections:", fp16_dets)
    print("INT8 1-image detections:", int8_dets)
    print("INT8 toycalib detections:", toy_dets)

    all_rows = []

    for name, dets, masks in [
        ("INT8_1image", int8_dets, int8_masks),
        ("INT8_toycalib", toy_dets, toy_masks),
    ]:
        rows = match_to_reference(fp16_dets, fp16_masks, dets, masks)
        for r in rows:
            r["compare_name"] = name
            all_rows.append(r)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "compare_name",
        "ref_index",
        "ref_class",
        "cmp_class",
        "class_match",
        "ref_score",
        "cmp_score",
        "score_diff",
        "box_iou",
        "mask_iou",
        "mask_dice",
        "ref_mask_pixels",
        "cmp_mask_pixels",
        "mask_pixels_diff",
        "ref_box",
        "cmp_box",
    ]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    summary_rows = []
    for name in ["INT8_1image", "INT8_toycalib"]:
        rows = [r for r in all_rows if r["compare_name"] == name]
        s = summarize(rows)
        s["compare_name"] = name
        summary_rows.append(s)

    summary_fields = [
        "compare_name",
        "matched_ref_count",
        "class_match_count",
        "mean_score_diff",
        "mean_box_iou",
        "mean_mask_iou",
        "mean_mask_dice",
        "extra_detection_count",
    ]

    with open(args.out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\nSaved:", args.out_csv)
    print("Saved:", args.out_summary)

    print("\nSummary:")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
