import torch
from torch.onnx import export
import torch.nn as nn

# Load YOLOv5 model using torch.hub.load
model = torch.hub.load('ultralytics/yolov5', 'custom', path='demo/best.pt', force_reload=True)

# Set the model in evaluation mode
model.eval()

# Example input
x = torch.zeros((1, 3, 640, 640), dtype=torch.float32)

# Export model to TorchScript
script_model = torch.jit.trace(model, x)

# Save TorchScript model
script_model.save("yolov5.pt")

# Convert TorchScript model to TensorFlow Lite
import onnx
from onnx_tf.backend import prepare

# Convert to ONNX format
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(script_model, dummy_input, "yolov5.onnx", opset_version=11)

# Load ONNX model
onnx_model = onnx.load("yolov5.onnx")

# Prepare ONNX model for TensorFlow Lite conversion
tf_rep = prepare(onnx_model)
tf_rep.export_graph("yolov5.pb")

# Convert to TensorFlow Lite
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph("yolov5.pb", input_arrays=["input_1"], output_arrays=["Identity"])
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "yolov5.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
