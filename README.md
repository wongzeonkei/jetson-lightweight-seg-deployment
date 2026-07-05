# Jetson Lightweight Segmentation Deployment

This project deploys a lightweight segmentation model to Jetson Nano B01.

Main target:

- Model: YOLOv8n-seg
- Input size: 320x320
- Deployment path: PyTorch -> ONNX -> ONNX Runtime -> TensorRT -> Jetson Nano B01
- Inference implementation: Python and C++ with OpenCV/TensorRT
- Metrics: latency, FPS, engine size, RAM/SWAP, GPU utilization
