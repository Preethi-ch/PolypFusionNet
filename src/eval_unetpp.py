import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

VAL_IMG = os.path.join(PROJECT_ROOT, "datasets", "unet_data", "val", "images")
VAL_MASK = os.path.join(PROJECT_ROOT, "datasets", "unet_data", "val", "masks")

RUN_DIR = os.path.join(PROJECT_ROOT, "runs", "segment", "unetpp")
WEIGHT_PATH = os.path.join(RUN_DIR, "weights", "best_unetpp.pth")
OUT_CSV = os.path.join(RUN_DIR, "eval_metrics.csv")

DEVICE = torch.device("cpu")

class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.images = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)

model = smp.UnetPlusPlus(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation="sigmoid"
)

model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

total_params = count_total_params(model)
trainable_params = count_trainable_params(model)
model_size = model_size_mb(model)

print("\n📊 UNet++ Model Information")
print("Total Parameters:", total_params)
print("Trainable Parameters:", trainable_params)
print("Model Size (MB):", round(model_size, 2))

def dice(pred, gt):
    smooth = 1e-6
    return (2 * (pred * gt).sum() + smooth) / (pred.sum() + gt.sum() + smooth)

def iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-6)


loader = DataLoader(PolypDataset(VAL_IMG, VAL_MASK), batch_size=1)

dice_l, iou_l, prec_l, rec_l, f1_l, acc_l, time_l = [], [], [], [], [], [], []

with torch.no_grad():
    for img, gt in loader:
        img, gt = img.to(DEVICE), gt.to(DEVICE)

        start = time.time()
        pred = model(img)
        infer_time = time.time() - start

        pred = (pred > 0.5).int()
        gt = gt.int()

        tp = (pred & gt).sum().item()
        fp = (pred & (~gt)).sum().item()
        fn = ((~pred) & gt).sum().item()

        d = dice(pred, gt).item()
        i = iou(pred.bool(), gt.bool()).item()
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        acc = (pred == gt).float().mean().item()

        dice_l.append(d)
        iou_l.append(i)
        prec_l.append(p)
        rec_l.append(r)
        f1_l.append(f1)
        acc_l.append(acc)
        time_l.append(infer_time)


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

pd.DataFrame([results]).to_csv(OUT_CSV, index=False)

print("\n✅ UNet++ Evaluation Completed")
for k, v in results.items():
    print(f"{k:25s}: {v:.4f}")

print("\n📁 Metrics saved to:", OUT_CSV)