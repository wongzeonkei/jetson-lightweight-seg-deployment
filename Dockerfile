FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt /app/service/requirements.txt

RUN pip install --no-cache-dir -r /app/service/requirements.txt

COPY service /app/service

ENV MODEL_PATH=/models/yolov8n_seg_320_opset13_sim.onnx
ENV INPUT_SIZE=320
ENV CONF_THRES=0.25
ENV IOU_THRES=0.45

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
