from ultralytics import YOLO


model = YOLO("yolov8s-seg.pt")

model.train(
    data="datasets/data.yaml",
    epochs=15,
    imgsz=640,
    batch=4,         
    device="cpu",     
    name="yolov8s_exp"
)
