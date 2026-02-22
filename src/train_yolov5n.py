import os
import sys

EPOCHS = 50
BATCH_SIZE = 1
IMG_SIZE = 416
WORKERS = 0
DEVICE = "cpu"

os.system(
    f"{sys.executable} ../yolov5/segment/train.py "
    f"--img {IMG_SIZE} "
    f"--batch {BATCH_SIZE} "
    f"--epochs {EPOCHS} "
    f"--data ../datasets/polyp.yaml "
    f"--weights yolov5n-seg.pt "
    f"--workers {WORKERS} "
    f"--device {DEVICE} "
    f"--cache False "
    f"--project runs/train-seg "
    f"--name train_yolov5n_seg "
    f"--exist-ok"
)
