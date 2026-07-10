# YOLOv8n-seg TensorRT INT8 PTQ Experiment Report

## 1. Objective

This experiment evaluates TensorRT INT8 post-training quantization (PTQ) for YOLOv8n-seg. The goal is to compare FP32, FP16, and INT8 inference latency, analyze INT8 output consistency, and study the impact of calibration dataset quality.

## 2. Experimental Setup

- Model: YOLOv8n-seg
- Input size: 320 x 320
- ONNX model: yolov8n_seg_320_opset13_sim.onnx
- Device: NVIDIA RTX 3090
- TensorRT version: 8.6.1
- INT8 method: TensorRT PTQ with IInt8EntropyCalibrator2
- Calibration type: entropy calibration
- Input / output format: FP32 bindings with TensorRT internal mixed precision

## 3. Calibration Settings

Two calibration settings were tested:

1. 1-image calibration:
   - Only bus.jpg was used.
   - This setting is used only for INT8 pipeline validation.

2. Toy calibration:
   - 21 augmented images generated from bus.jpg.
   - Augmentations include brightness, contrast, flipping, cropping, and shifting.
   - This is still not a formal representative calibration dataset, but it is more diverse than the 1-image setting.

## 4. Latency Comparison

| Backend | Precision | Calibration | Engine Size | Throughput | Mean Latency | GPU Compute |
|---|---|---|---:|---:|---:|---:|
| TensorRT | FP32 | None | 17 MB | 443.99 QPS | 2.301 ms | 1.802 ms |
| TensorRT | FP16 | None | 8.8 MB | 568.08 QPS | 1.804 ms | 1.306 ms |
| TensorRT | INT8 | 1-image | 5.5 MB | 776.12 QPS | 1.402 ms | 0.906 ms |
| TensorRT | INT8 | Toy calib | 5.5 MB | 783.93 QPS | 1.395 ms | 0.899 ms |

INT8 reduces engine size and improves inference speed compared with FP16. The major speedup comes from reduced GPU compute time, while H2D and D2H latency remain nearly unchanged because the model bindings are still FP32.

## 5. Output Consistency Analysis

FP16 TensorRT output is used as the reference. IoU and Dice in this report measure consistency between FP16 and INT8 outputs, not ground-truth segmentation accuracy.

| INT8 Setting | Class Match | Mean Score Diff | Mean Box IoU | Mean Mask IoU | Mean Mask Dice |
|---|---:|---:|---:|---:|---:|
| 1-image calibration | 2 / 4 | -0.4441 | 0.7237 | 0.6942 | 0.7213 |
| Toy calibration | 4 / 4 | -0.1112 | 0.9592 | 0.9373 | 0.9675 |

## 6. Observations

With 1-image calibration, INT8 PTQ shows significant output degradation. The model suffers from confidence drop, bus/train class confusion, and unstable instance predictions.

With toy calibration, output consistency improves significantly. The detection result recovers to 3 persons + 1 bus, and the mean mask Dice with FP16 reference increases from 0.7213 to 0.9675.

This demonstrates that INT8 PTQ is highly sensitive to calibration data. A more representative calibration dataset can significantly improve quantized model stability.

## 7. Limitations

The current toy calibration dataset is generated from one image and is not a formal representative dataset. Therefore, the results should be considered as PTQ pipeline validation and calibration sensitivity analysis, not final production-level INT8 accuracy.

A formal INT8 evaluation should use a larger calibration dataset and a labeled validation set to report true mAP, Dice, or IoU.

## 8. Conclusion

TensorRT INT8 PTQ can significantly improve YOLOv8n-seg inference speed and reduce engine size. However, INT8 quantization may cause confidence degradation and class confusion when the calibration dataset is insufficient. Calibration dataset coverage is critical for stable INT8 deployment.
