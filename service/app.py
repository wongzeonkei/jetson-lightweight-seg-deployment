import logging
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from service.yolov8_seg_onnx import YOLOv8SegONNX


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "models/onnx/yolov8n_seg_320_opset13_sim.onnx"
)
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "320"))
CONF_THRES = float(os.environ.get("CONF_THRES", "0.25"))
IOU_THRES = float(os.environ.get("IOU_THRES", "0.45"))

app = FastAPI(
    title="YOLOv8n-seg FastAPI Inference Service",
    description="FastAPI service for YOLOv8n-seg ONNX Runtime inference.",
    version="0.2.0"
)

model = None


@app.on_event("startup")
def startup_event():
    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model file not found: {}. Please check MODEL_PATH.".format(MODEL_PATH)
        )

    logging.info("Loading model from %s", MODEL_PATH)

    model = YOLOv8SegONNX(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
    )

    logging.info("Model loaded successfully.")
    logging.info("Providers: %s", model.providers)


@app.get("/")
def root():
    return {
        "service": "YOLOv8n-seg FastAPI Inference Service",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "input_size": INPUT_SIZE,
        "providers": model.providers if model is not None else None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    request_t0 = time.perf_counter()

    image_bytes = await file.read()
    np_buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    result = model.predict(img)

    request_t1 = time.perf_counter()
    result["latency_ms"]["request_total"] = round((request_t1 - request_t0) * 1000, 3)

    result["file"] = {
        "filename": file.filename,
        "content_type": file.content_type,
        "image_shape": list(img.shape),
    }

    logging.info(
        "file=%s detections=%d inference_ms=%.3f request_total_ms=%.3f",
        file.filename,
        result["num_detections"],
        result["latency_ms"]["inference"],
        result["latency_ms"]["request_total"],
    )

    return result
