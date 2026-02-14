# from ultralytics import YOLO

# # Load best trained model
# model = YOLO("runs/segment/train/weights/best.pt")

# # Run inference on validation images
# model.predict(
#     source="datasets/images/val",
#     save=True,
#     conf=0.25
# )


from ultralytics import YOLO

# Load trained segmentation model
model = YOLO("runs/segment/train/weights/best.pt")

# Run inference AND save masks
model.predict(
    source="datasets/images/val",
    save=True,          # save images
    save_txt=True,      # save segmentation labels
    save_conf=True,
    conf=0.25,
    project="runs/segment",
    name="predict_masks"
)
