import base64
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def encode_image_base64(img: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class YOLOv8SegONNX:
    def __init__(
        self,
        model_path: str,
        input_size: int = 320,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h

        left = int(round(pad_w / 2 - 0.1))
        right = int(round(pad_w / 2 + 0.1))
        top = int(round(pad_h / 2 - 0.1))
        bottom = int(round(pad_h / 2 + 0.1))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return padded, scale, left, top

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        padded, scale, pad_x, pad_y = self.letterbox(img_bgr)

        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = np.ascontiguousarray(x, dtype=np.float32)

        meta = {
            "scale": float(scale),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
            "original_shape": list(img_bgr.shape),
        }

        return x, meta

    def postprocess(
        self,
        output0: np.ndarray,
        output1: np.ndarray,
        original: np.ndarray,
        meta: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:

        pred = output0[0].transpose(1, 0)

        boxes_xywh = pred[:, 0:4]
        class_scores = pred[:, 4:84]
        mask_coeffs = pred[:, 84:116]

        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        keep = scores >= self.conf_thres
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        if boxes_xywh.shape[0] == 0:
            empty = np.zeros(original.shape[:2], dtype=np.uint8)
            return original.copy(), empty, []

        boxes_float = []
        boxes_for_nms = []

        for box in boxes_xywh:
            cx, cy, w, h = box
            x1 = max(0.0, min(float(cx - w / 2), float(self.input_size)))
            y1 = max(0.0, min(float(cy - h / 2), float(self.input_size)))
            x2 = max(0.0, min(float(cx + w / 2), float(self.input_size)))
            y2 = max(0.0, min(float(cy + h / 2), float(self.input_size)))

            boxes_float.append([x1, y1, x2, y2])
            boxes_for_nms.append([
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            ])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            scores.tolist(),
            self.conf_thres,
            self.iou_thres,
        )

        if len(indices) == 0:
            empty = np.zeros(original.shape[:2], dtype=np.uint8)
            return original.copy(), empty, []

        indices = np.array(indices).reshape(-1).tolist()

        proto = output1[0]
        proto_flat = proto.reshape(32, -1)

        h0, w0 = original.shape[:2]
        scale = float(meta["scale"])
        pad_x = int(meta["pad_x"])
        pad_y = int(meta["pad_y"])

        vis = original.copy()
        merged_mask = np.zeros((h0, w0), dtype=np.uint8)
        detections = []

        for idx in indices:
            coeff = mask_coeffs[idx]
            mask = sigmoid(np.dot(coeff, proto_flat)).reshape(80, 80)

            x1, y1, x2, y2 = boxes_float[idx]

            mask320 = cv2.resize(
                mask,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )

            crop = np.zeros_like(mask320)

            ix1 = max(0, min(int(x1), self.input_size - 1))
            iy1 = max(0, min(int(y1), self.input_size - 1))
            ix2 = max(0, min(int(x2), self.input_size))
            iy2 = max(0, min(int(y2), self.input_size))

            if ix2 > ix1 and iy2 > iy1:
                crop[iy1:iy2, ix1:ix2] = mask320[iy1:iy2, ix1:ix2]

            mask_bin = (crop > 0.5).astype(np.uint8)

            x_unpad1 = pad_x
            y_unpad1 = pad_y
            x_unpad2 = self.input_size - pad_x
            y_unpad2 = self.input_size - pad_y

            mask_unpad = mask_bin[y_unpad1:y_unpad2, x_unpad1:x_unpad2]
            mask_orig = cv2.resize(mask_unpad, (w0, h0), interpolation=cv2.INTER_NEAREST)

            merged_mask[mask_orig > 0] = 255

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
                int((29 * (cls_id + 5)) % 255),
            )

            overlay = np.zeros_like(vis)
            overlay[mask_orig > 0] = color
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.45, 0.0)

            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(
                vis,
                "{} {:.2f}".format(cls_name, score),
                (bx1, max(0, by1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "score": round(score, 6),
                "box": [bx1, by1, bx2, by2],
                "mask_pixels": int(mask_orig.sum()),
            })

        return vis, merged_mask, detections

    def predict(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        x, meta = self.preprocess(img_bgr)
        t1 = time.perf_counter()

        outputs = self.session.run(self.output_names, {self.input_name: x})
        t2 = time.perf_counter()

        vis, mask, detections = self.postprocess(outputs[0], outputs[1], img_bgr, meta)
        t3 = time.perf_counter()

        return {
            "num_detections": len(detections),
            "detections": detections,
            "latency_ms": {
                "preprocess": round((t1 - t0) * 1000, 3),
                "inference": round((t2 - t1) * 1000, 3),
                "postprocess": round((t3 - t2) * 1000, 3),
                "total": round((t3 - t0) * 1000, 3),
            },
            "model_info": {
                "model_path": self.model_path,
                "input_name": self.input_name,
                "output_names": self.output_names,
                "providers": self.providers,
                "input_size": self.input_size,
            },
            "result_image_base64": encode_image_base64(vis, ".jpg"),
            "mask_image_base64": encode_image_base64(mask, ".png"),
        }
