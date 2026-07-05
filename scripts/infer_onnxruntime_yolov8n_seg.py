import os
import time
import cv2
import numpy as np
import onnxruntime as ort


CLASSES = [
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


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    dw = new_w - resized_w
    dh = new_h - resized_h
    dw /= 2
    dh /= 2

    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top


def preprocess(img_path, input_size=320):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    original = img.copy()
    padded, scale, pad_x, pad_y = letterbox(img, (input_size, input_size))

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(x, dtype=np.float32)

    return original, x, scale, pad_x, pad_y


def xywh_to_xyxy(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def crop_mask(mask, box):
    x1, y1, x2, y2 = box.astype(np.int32)
    h, w = mask.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    cropped = np.zeros_like(mask)
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def postprocess(output0, output1, original_img, scale, pad_x, pad_y,
                conf_thres=0.25, iou_thres=0.45, input_size=320):
    pred = output0[0].transpose(1, 0)  # [2100, 116]
    proto = output1[0]                 # [32, 80, 80]

    boxes_xywh = pred[:, 0:4]
    class_scores = pred[:, 4:84]
    mask_coeffs = pred[:, 84:116]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep = scores > conf_thres
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    if boxes_xywh.shape[0] == 0:
        return original_img, []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # NMS needs xywh boxes
    nms_boxes = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b
        nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=scores.astype(float).tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres
    )

    if len(indices) == 0:
        return original_img, []

    indices = np.array(indices).reshape(-1)

    boxes_xyxy = boxes_xyxy[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    mask_coeffs = mask_coeffs[indices]

    proto_flat = proto.reshape(32, -1)  # [32, 6400]
    masks = sigmoid(mask_coeffs @ proto_flat)
    masks = masks.reshape(-1, 80, 80)

    vis = original_img.copy()
    h0, w0 = original_img.shape[:2]
    detections = []

    for i in range(len(indices)):
        box_in = boxes_xyxy[i].copy()
        box_in = np.clip(box_in, 0, input_size)

        mask80 = masks[i]
        mask320 = cv2.resize(mask80, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        mask320 = crop_mask(mask320, box_in)
        mask320 = mask320 > 0.5

        # remove padding and resize mask to original image
        x1_unpad = int(round(pad_x))
        y1_unpad = int(round(pad_y))
        x2_unpad = int(round(input_size - pad_x))
        y2_unpad = int(round(input_size - pad_y))

        mask_unpad = mask320[y1_unpad:y2_unpad, x1_unpad:x2_unpad]
        mask_orig = cv2.resize(
            mask_unpad.astype(np.uint8),
            (w0, h0),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # map box from input coordinates back to original image
        box = box_in.copy()
        box[[0, 2]] -= pad_x
        box[[1, 3]] -= pad_y
        box /= scale

        x1, y1, x2, y2 = box.astype(np.int32)
        x1 = max(0, min(x1, w0 - 1))
        y1 = max(0, min(y1, h0 - 1))
        x2 = max(0, min(x2, w0 - 1))
        y2 = max(0, min(y2, h0 - 1))

        cls_id = int(class_ids[i])
        score = float(scores[i])
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)

        color = (
            int(37 * (cls_id + 1) % 255),
            int(17 * (cls_id + 3) % 255),
            int(29 * (cls_id + 5) % 255)
        )

        colored = np.zeros_like(vis)
        colored[mask_orig] = color
        vis = cv2.addWeighted(vis, 1.0, colored, 0.45, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append({
            "class_id": cls_id,
            "class_name": name,
            "score": score,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "mask_pixels": int(mask_orig.sum())
        })

    return vis, detections


def main():
    onnx_path = "models/onnx/yolov8n_seg_320_opset13_sim.onnx"
    image_path = "data/samples/bus.jpg"
    output_path = "outputs/onnxruntime/yolov8n_seg_320_bus_ort.jpg"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    print("Available providers:", available)
    print("Requested providers:", providers)

    sess = ort.InferenceSession(onnx_path, providers=providers)
    print("Session providers:", sess.get_providers())

    original, x, scale, pad_x, pad_y = preprocess(image_path, 320)

    # warmup
    for _ in range(10):
        outputs = sess.run(None, {"images": x})

    times = []
    for _ in range(100):
        t0 = time.time()
        outputs = sess.run(None, {"images": x})
        times.append((time.time() - t0) * 1000)

    output0, output1 = outputs
    vis, detections = postprocess(
        output0, output1, original,
        scale, pad_x, pad_y,
        conf_thres=0.25,
        iou_thres=0.45,
        input_size=320
    )

    cv2.imwrite(output_path, vis)

    print("\nOutput shapes:")
    print(" output0:", output0.shape)
    print(" output1:", output1.shape)

    print("\nLatency:")
    print(" mean ms:", float(np.mean(times)))
    print(" median ms:", float(np.median(times)))
    print(" min/max ms:", float(np.min(times)), float(np.max(times)))

    print("\nDetections:")
    for d in detections:
        print(d)

    print("\nSaved visualization:", output_path)


if __name__ == "__main__":
    main()
