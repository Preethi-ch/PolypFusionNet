from ultralytics import YOLO

# Load YOLOv8 Small Segmentation model (higher accuracy, more compute)
model = YOLO("yolov8s-seg.pt")

model.train(
    data="datasets/data.yaml",
    epochs=15,
    imgsz=640,
    batch=4,          # smaller batch to avoid RAM crash
    device="cpu",     # laptop-friendly
    name="yolov8s_exp"
)
