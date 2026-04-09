from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os

from mqi import compute_mqi_from_mask
from repair import repair_mask_in_memory
from severity import compute_severity_from_mask

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

IMAGE_PATH = args.image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "segment", "yolov8n_train", "weights", "best.pt"
)

print("\n[INFO] Loading model...")
model = YOLO(MODEL_PATH)

print("[STEP 1] Detection & Segmentation")
results = model.predict(source=IMAGE_PATH, conf=0.25, save=False, verbose=False)

if results[0].masks is None:
    print("\n No polyp detected")
    exit()

mask = results[0].masks.data[0].cpu().numpy()
binary_mask = (mask > 0.5).astype(np.uint8) * 255
print(" Polyp detected")

print("\n[STEP 2] MQI")
mqi_score, mqi_label = compute_mqi_from_mask(binary_mask)
print(f"MQI: {mqi_score} ({mqi_label})")

if mqi_label == "Reject":
    print("\n Mask rejected")
    exit()

print("\n[STEP 3] Mask Repair")
if mqi_label == "Needs_Repair":
    binary_mask = repair_mask_in_memory(binary_mask)
    print("🛠 Mask repaired")
else:
    print("✔ No repair needed")

print("\n[STEP 4] Severity Scoring")
severity_score, severity_label = compute_severity_from_mask(binary_mask)

print("\n==============================")
print(" FINAL OUTPUT ")
print("==============================")
print("Polyp Detected : YES")
print(f"Severity       : {severity_label}")
print("==============================\n")