import os
import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

VAL_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "images", "val")
VAL_LABEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "labels", "val")

# ✅ USE EXISTING YOLOv8s TRAIN RUN
RUN_DIR = os.path.join(PROJECT_ROOT, "runs", "segment", "yolov8s_train")
WEIGHTS_PATH = os.path.join(RUN_DIR, "weights", "best.pt")

# ================= LOAD MODEL =================
model = YOLO(WEIGHTS_PATH)

# ================= MODEL PARAMETER INFO =================
def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

total_params = count_total_params(model.model)
trainable_params = count_trainable_params(model.model)
model_size = model_size_mb(model.model)

print("\n📊 YOLOv8s Model Information")
print("Total Parameters:", total_params)
print("Trainable Parameters:", trainable_params)
print("Model Size (MB):", round(model_size, 2))

# ================= HELPER FUNCTIONS =================
def is_image_file(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

def polygon_to_mask(label_path, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask.astype(bool)

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_shape[1]
            coords[:, 1] *= img_shape[0]
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

    return mask.astype(bool)

def dice(pred, gt):
    return (2 * (pred & gt).sum()) / ((pred.sum() + gt.sum()) + 1e-6)

def iou(pred, gt):
    return (pred & gt).sum() / ((pred | gt).sum() + 1e-6)

# ================= EVALUATION =================
dice_l, iou_l, prec_l, rec_l, f1_l, acc_l, time_l = [], [], [], [], [], [], []

for fname in sorted(os.listdir(VAL_IMG_DIR)):

    # Ignore non-image files
    if not is_image_file(fname):
        continue

    img_path = os.path.join(VAL_IMG_DIR, fname)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠ Skipping unreadable image: {fname}")
        continue

    label_path = os.path.join(
        VAL_LABEL_DIR, fname.rsplit(".", 1)[0] + ".txt"
    )

    gt_mask = polygon_to_mask(label_path, img.shape)

    start = time.time()
    result = model(img_path, imgsz=640, verbose=False)[0]
    infer_time = time.time() - start

    # Handle no detections safely
    if result.masks is None or result.masks.data.shape[0] == 0:
        pred_mask = np.zeros_like(gt_mask)
    else:
        pred_masks = result.masks.data.cpu().numpy()
        pred_mask = np.any(pred_masks > 0.5, axis=0)

    tp = (pred_mask & gt_mask).sum()
    fp = (pred_mask & ~gt_mask).sum()
    fn = (~pred_mask & gt_mask).sum()

    d = dice(pred_mask, gt_mask)
    i = iou(pred_mask, gt_mask)
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    acc = (pred_mask == gt_mask).mean()

    dice_l.append(d)
    iou_l.append(i)
    prec_l.append(p)
    rec_l.append(r)
    f1_l.append(f1)
    acc_l.append(acc)
    time_l.append(infer_time)

# ================= SAVE RESULTS =================
results = {
    "Total_Parameters": total_params,
    "Trainable_Parameters": trainable_params,
    "Model_Size_MB": round(model_size, 2),
    "Dice": np.mean(dice_l),
    "IoU": np.mean(iou_l),
    "Precision": np.mean(prec_l),
    "Recall": np.mean(rec_l),
    "F1": np.mean(f1_l),
    "Pixel_Accuracy": np.mean(acc_l),
    "Avg_Inference_ms": np.mean(time_l) * 1000,
    "FPS": 1 / np.mean(time_l)
}

out_csv = os.path.join(RUN_DIR, "eval_metrics.csv")
pd.DataFrame([results]).to_csv(out_csv, index=False)

print("\n✅ YOLOv8s Evaluation Completed")
for k, v in results.items():
    print(f"{k:25s}: {v:.4f}")

print("\n📁 Metrics saved to:", out_csv)