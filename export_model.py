"""
Converts a YOLO model trained in PyTorch (.pt) to ONNX (.onnx) format
for use on various inference platforms.

Steps:
- Load the YOLO model from a .pt file
- Export the model to ONNX format

Output:
- 'best.onnx' file in the current working directory
"""

from ultralytics import YOLO    

# Load a YOLO11n PyTorch model
model = YOLO("runs/datalabelstudio_75_15_10_clean_flip_train/y11n_batch16_epochs100/weights/best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'best.onnx'