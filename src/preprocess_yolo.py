import os
import cv2
import random
import numpy as np
from glob import glob

IMG_SIZE = 640
CLASS_ID = 0
SPLIT_RATIO = 0.8

def ensure_dirs():
    for p in [
        "datasets/images/train", "datasets/images/val",
        "datasets/labels/train", "datasets/labels/val"
    ]:
        os.makedirs(p, exist_ok=True)

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) != 2:
            continue
        poly = cnt / IMG_SIZE
        polygons.append(poly)
    return polygons

def process_images(image_paths, mask_dir):
    random.shuffle(image_paths)
    split = int(len(image_paths) * SPLIT_RATIO)

    for i, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        polygons = mask_to_polygon(mask)

        split_type = "train" if i < split else "val"
        img_out = f"datasets/images/{split_type}/{name}"
        lbl_out = f"datasets/labels/{split_type}/{name.rsplit('.',1)[0]}.txt"

        cv2.imwrite(img_out, img)

        with open(lbl_out, "w") as f:
            for poly in polygons:
                coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in poly])
                f.write(f"{CLASS_ID} {coords}\n")

def process_kvasir():
    imgs = glob("datasets/Kvasir-SEG/images/*")
    process_images(imgs, "datasets/Kvasir-SEG/masks")

def process_cvc():
    imgs = glob("datasets/CVC_clinic/Original/*")
    process_images(imgs, "datasets/CVC_clinic/Ground Truth")

def process_polypgen():
    base = "datasets/polypgen"
    for seq in os.listdir(base):
        imgs = glob(f"{base}/{seq}/images/*")
        process_images(imgs, f"{base}/{seq}/masks")

if __name__ == "__main__":
    ensure_dirs()
    process_kvasir()
    process_cvc()
    process_polypgen()
    print("Preprocessing completed successfully.")
