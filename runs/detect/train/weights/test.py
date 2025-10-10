from ultralytics import YOLO
import gradio as gr
model = YOLO(
    "/home/aleisley/Documents/mengai/ai231/runs/detect/train/weights/best.onnx", task="detect")

# predict using webcam
model.predict(source=0,
              conf=0.5,
              show=True,
              iou=0.45,           # IoU threshold for NMS
              max_det=300,        # Maximum detections
              agnostic_nms=False,  # Class-agnostic NMS
              verbose=False,
              device="cpu"  # Use CPU for inference
              )
