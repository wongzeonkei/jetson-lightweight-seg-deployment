# Jetson Lightweight Segmentation Deployment

This project implements an end-to-end lightweight instance segmentation deployment pipeline for YOLOv8n-seg, covering PyTorch inference, ONNX export, ONNX Runtime CUDA validation, TensorRT FP32/FP16 benchmarking, C++ OpenCV + TensorRT inference, and Jetson Nano B01 deployment.

## Model

- Model: YOLOv8n-seg
- Input size: 320 x 320
- Task: instance segmentation
- Dataset/sample: COCO sample image `bus.jpg`
- Outputs:
  - `output0`: `[1, 116, 2100]`
  - `output1`: `[1, 32, 80, 80]`

`output0` contains bounding boxes, class scores, and mask coefficients. `output1` contains prototype masks used for YOLOv8 segmentation mask reconstruction.

## Deployment Pipeline

```text
PyTorch YOLOv8n-seg
        ↓
ONNX export
        ↓
ONNX simplification
        ↓
ONNX Runtime CUDA validation
        ↓
TensorRT FP32 / FP16 engine build
        ↓
trtexec benchmark
        ↓
real-input TensorRT inference
        ↓
C++ OpenCV + TensorRT inference
        ↓
Jetson Nano B01 TensorRT FP16 deployment
RTX 3090 Results
BackendPrecisionInputSourceMean LatencyMedian LatencyThroughputNotes
PyTorchFP32320x320Ultralytics Python65.3 ms--Framework baseline
ONNX RuntimeFP32320x320CUDA EP5.10 ms4.36 ms-Custom Python postprocess
TensorRTFP32320x320trtexec random2.30 ms1.89 ms443.99 qpsEngine benchmark
TensorRTFP16320x320trtexec random1.84 ms1.52 ms556.77 qpsEngine benchmark
TensorRTFP16320x320trtexec real input1.80 ms1.53 ms568.08 qpsbus.jpg input
TensorRTFP16320x320C++ OpenCV + TensorRT2.55 ms2.22 ms-Includes H2D, executeV2, D2H, sync
Jetson Nano B01 Results

Jetson Nano B01 uses its own locally built TensorRT engine. Engines generated on RTX 3090 are not portable to Jetson Nano because TensorRT engines are device-, CUDA-, TensorRT-, and GPU-architecture-specific.

DeviceBackendPrecisionInputSourceMean LatencyMedian LatencyThroughputGPU ComputeEngine SizeDetections
Jetson Nano B01TensorRTFP16320x320trtexec random18.13 ms18.14 ms55.11 qps17.84 ms9.5 MB-
Jetson Nano B01TensorRTFP16320x320trtexec real input18.25 ms18.25 ms54.76 qps17.95 ms9.5 MB3 persons + 1 bus
Jetson Runtime Observation

During Jetson TensorRT validation:

RAM increased from approximately 289 MB idle to about 1.65 GB during deployment.
SWAP remained 0 MB, indicating no swap pressure.
GR3D_FREQ reached up to 99%, indicating effective GPU utilization.
GPU temperature stayed around 26-28°C in the recorded run.
No out-of-memory, thermal throttling, or TensorRT runtime failure was observed.
Key Files
scripts/export_yolov8n_seg_onnx.py
scripts/simplify_onnx.py
scripts/infer_onnxruntime_yolov8n_seg.py
scripts/prepare_trt_input_yolov8n_seg.py
scripts/postprocess_trtexec_yolov8n_seg.py
scripts/compare_ort_trt_yolov8n_seg.py
scripts/jetson_prepare_trt_input_yolov8n_seg.py
scripts/jetson_postprocess_trtexec_yolov8n_seg.py
deploy/tensorrt/cpp/infer_yolov8n_seg_trt.cpp
deploy/tensorrt/cpp/CMakeLists.txt
docs/benchmarks/rtx3090_yolov8n_seg_320_backend_summary.csv
docs/benchmarks/jetson/jetson_yolov8n_seg_320_backend_summary.csv
Notes

Large runtime artifacts are intentionally excluded from Git:

*.pt
*.onnx
*.engine
outputs/
deploy/tensorrt/logs/
data/samples/

To reproduce the full pipeline, place the required model weights, ONNX model, TensorRT engine, and sample image in the corresponding directories.
