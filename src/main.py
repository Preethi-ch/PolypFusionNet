from ultralytics import YOLO
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

IMAGE_SOURCE = args.image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "runs", "segment", "yolov8n_train", "weights", "best.pt"
)

RUNS_SEGMENT_DIR = os.path.join(PROJECT_ROOT, "runs", "segment")
PREDICT_NAME = "predict_with_binary"

BINARY_MASK_DIR = os.path.join(
    RUNS_SEGMENT_DIR,
    PREDICT_NAME,
    "binary_masks"
)

os.makedirs(BINARY_MASK_DIR, exist_ok=True)


print("[INFO] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully")


print("[INFO] Running inference on uploaded image...")

results = model.predict(
    source=IMAGE_SOURCE,
    conf=0.25,
    save=True,
    project=RUNS_SEGMENT_DIR,
    name=PREDICT_NAME,
    verbose=False
)


print("[INFO] Extracting binary mask...")

if results[0].masks is None:
    print("[RESULT] ❌ No polyp detected")
    exit()

mask = results[0].masks.data[0].cpu().numpy()
binary_mask = (mask > 0.5).astype(np.uint8) * 255

image_name = os.path.splitext(os.path.basename(IMAGE_SOURCE))[0]
mask_filename = f"{image_name}_mask.png"

mask_path = os.path.join(BINARY_MASK_DIR, mask_filename)
cv2.imwrite(mask_path, binary_mask)

print(f"[RESULT] ✅ Polyp detected")
print(f"[SAVED] Binary mask → {mask_path}")