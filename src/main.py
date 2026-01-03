from ultralytics import YOLO

# Load best trained model
model = YOLO("runs/segment/train/weights/best.pt")

# Run inference on validation images
model.predict(
    source="datasets/images/val",
    save=True,
    conf=0.25
)
