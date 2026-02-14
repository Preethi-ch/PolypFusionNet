from ultralytics import YOLO

# Load YOLOv8n segmentation model (lightweight)
model = YOLO("yolov8n-seg.pt")

# Train the model
model.train(
    data="datasets/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    resume=True
)
