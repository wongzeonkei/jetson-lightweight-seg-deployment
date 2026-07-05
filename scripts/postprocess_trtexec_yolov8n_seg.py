import os
import json
import importlib.util
import numpy as np
import cv2


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
        # Prefer numeric payload fields if trtexec uses dict entries.
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

    # Fallback: sometimes exportOutput is a list ordered by output bindings.
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                item_name = str(item.get("name", item.get("Name", "")))
                if item_name == name:
                    arr = np.array(flatten_numeric(item), dtype=np.float32)
                    if arr.size >= expected:
                        return arr[-expected:].reshape(shape)

    raise RuntimeError(
        f"Failed to extract tensor {name} with shape {shape}. "
        f"Candidates found: {len(candidates)}"
    )


# Load preprocess/postprocess from the ORT script so both paths share the same logic.
spec = importlib.util.spec_from_file_location(
    "ort_yolo_seg",
    "scripts/infer_onnxruntime_yolov8n_seg.py"
)
ort_yolo_seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ort_yolo_seg)


def main():
    json_path = "outputs/tensorrt/trt_fp16_yolov8n_seg_320_bus_output.json"
    image_path = "data/samples/bus.jpg"
    output_path = "outputs/tensorrt/yolov8n_seg_320_bus_trt_fp16.jpg"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output0 = extract_named_tensor(data, "output0", (1, 116, 2100))
    output1 = extract_named_tensor(data, "output1", (1, 32, 80, 80))

    original, x, scale, pad_x, pad_y = ort_yolo_seg.preprocess(image_path, 320)

    vis, detections = ort_yolo_seg.postprocess(
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

    np.save("outputs/tensorrt/trt_fp16_output0.npy", output0)
    np.save("outputs/tensorrt/trt_fp16_output1.npy", output1)

    print("Parsed TensorRT outputs:")
    print(" output0:", output0.shape, output0.dtype, "min/max:", float(output0.min()), float(output0.max()))
    print(" output1:", output1.shape, output1.dtype, "min/max:", float(output1.min()), float(output1.max()))

    print("\nDetections:")
    for d in detections:
        print(d)

    print("\nSaved visualization:", output_path)
    print("Saved npy:")
    print(" outputs/tensorrt/trt_fp16_output0.npy")
    print(" outputs/tensorrt/trt_fp16_output1.npy")


if __name__ == "__main__":
    main()
