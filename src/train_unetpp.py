import os
import cv2
import time
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torch.nn as nn

# ================== BASE PATH SETUP ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # PolyFusionNet/

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "unet_data")

IMG_DIR_TRAIN = os.path.join(DATASET_ROOT, "train", "images")
MASK_DIR_TRAIN = os.path.join(DATASET_ROOT, "train", "masks")
IMG_DIR_VAL = os.path.join(DATASET_ROOT, "val", "images")
MASK_DIR_VAL = os.path.join(DATASET_ROOT, "val", "masks")

# ================== EXP CONFIG ==================
EXP_NAME = "unetpp"
RUN_DIR = os.path.join(PROJECT_ROOT, "runs", "segment", EXP_NAME)
WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")
CKPT_PATH = os.path.join(WEIGHTS_DIR, "last_checkpoint.pth")

BATCH_SIZE = 2
LR = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories
os.makedirs(WEIGHTS_DIR, exist_ok=True)

print("Train Images:", IMG_DIR_TRAIN)
print("Val Images:", IMG_DIR_VAL)

# ================== SAVE ARGS ==================
args = {
    "model": "UNet++",
    "encoder": "resnet18",
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "device": DEVICE,
    "resume_supported": True
}

with open(os.path.join(RUN_DIR, "args.yaml"), "w") as f:
    yaml.dump(args, f)

# ================== DATASET ==================
class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.images = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            if np.random.rand() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)

# ================== MODEL ==================
model = smp.UnetPlusPlus(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid"
).to(DEVICE)

# ================== LOSS ==================
bce = nn.BCELoss()

def dice_loss(pred, target):
    smooth = 1.0
    inter = (pred * target).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

# ================== DATA LOADERS ==================
train_ds = PolypDataset(IMG_DIR_TRAIN, MASK_DIR_TRAIN, augment=True)
val_ds = PolypDataset(IMG_DIR_VAL, MASK_DIR_VAL, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================== RESUME SUPPORT ==================
start_epoch = 1
best_val = float("inf")
results = []

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt["best_val"]
    print(f"🔁 Resuming from epoch {start_epoch}")
else:
    print("🆕 Starting from scratch")

# ================== TRAIN LOOP ==================
counter = 0
start_time = time.time()

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    model.train()
    train_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        pred = model(img)
        loss = loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            val_loss += loss_fn(model(img), mask).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    results.append([epoch, train_loss, val_loss])

    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    # Save best
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_unetpp.pth"))
    else:
        counter += 1

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val
    }, CKPT_PATH)

    if counter >= PATIENCE:
        print("⏹ Early stopping triggered")
        break

# ================== SAVE RESULTS ==================
df = pd.DataFrame(results, columns=["epoch", "train_loss", "val_loss"])
df.to_csv(os.path.join(RUN_DIR, "results.csv"), index=False)

plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train")
plt.plot(df["epoch"], df["val_loss"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("UNet++ Training Curve")
plt.savefig(os.path.join(RUN_DIR, "loss_curve.png"))
plt.close()

with open(os.path.join(RUN_DIR, "logs.txt"), "w") as f:
    f.write(f"Training time (hours): {(time.time()-start_time)/3600:.2f}\n")
    f.write(f"Best validation loss: {best_val:.4f}\n")

print("✅ Training Complete")
print("📁 Results saved to:", RUN_DIR)