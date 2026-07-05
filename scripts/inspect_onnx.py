import sys
import onnx
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "models/onnx/yolov8n_seg_320_opset13.onnx"
model = onnx.load(path)

print("ONNX path:", path)
print("IR version:", model.ir_version)

print("\nOpsets:")
for opset in model.opset_import:
    print(" ", opset.domain if opset.domain else "ai.onnx", opset.version)

print("\nInputs:")
for x in model.graph.input:
    dims = []
    for d in x.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else d.dim_param)
    print(" ", x.name, dims)

print("\nOutputs:")
for x in model.graph.output:
    dims = []
    for d in x.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else d.dim_param)
    print(" ", x.name, dims)

ops = Counter(node.op_type for node in model.graph.node)
print("\nTop op types:")
for k, v in ops.most_common():
    print(f"  {k}: {v}")

onnx.checker.check_model(model)
print("\nONNX checker passed.")
