import time
import numpy as np
import onnxruntime as ort

onnx_path = "models/onnx/yolov8n_seg_320_opset13_sim.onnx"

available = ort.get_available_providers()
providers = []
if "CUDAExecutionProvider" in available:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")

print("ONNX:", onnx_path)
print("Available providers:", available)
print("Requested providers:", providers)

sess = ort.InferenceSession(onnx_path, providers=providers)
print("Session providers:", sess.get_providers())

print("\nInputs:")
for i in sess.get_inputs():
    print(" ", i.name, i.shape, i.type)

print("\nOutputs:")
for o in sess.get_outputs():
    print(" ", o.name, o.shape, o.type)

x = np.random.rand(1, 3, 320, 320).astype(np.float32)

for _ in range(10):
    y = sess.run(None, {"images": x})

times = []
for _ in range(100):
    t0 = time.time()
    y = sess.run(None, {"images": x})
    times.append((time.time() - t0) * 1000)

print("\nRuntime output shapes:")
for idx, out in enumerate(y):
    print(f" output{idx}:", out.shape, out.dtype, "min/max:", float(out.min()), float(out.max()))

print("\nLatency mean ms:", float(np.mean(times)))
print("Latency median ms:", float(np.median(times)))
print("Latency min/max ms:", float(np.min(times)), float(np.max(times)))
