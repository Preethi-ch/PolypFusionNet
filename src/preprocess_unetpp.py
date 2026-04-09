import os
import cv2
import random
from glob import glob


IMG_SIZE = 256
SPLIT_RATIO = 0.8
SEED = 42


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_OUT = os.path.join(PROJECT_ROOT, "datasets", "unet_data")


def ensure_dirs():
    paths = [
        os.path.join(BASE_OUT, "train", "images"),
        os.path.join(BASE_OUT, "train", "masks"),
        os.path.join(BASE_OUT, "val", "images"),
        os.path.join(BASE_OUT, "val", "masks"),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def process_dataset(image_paths, mask_dir, prefix=""):
    random.seed(SEED)
    random.shuffle(image_paths)

    split_index = int(len(image_paths) * SPLIT_RATIO)

    for i, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)

        
        new_name = f"{prefix}_{name}" if prefix else name

        mask_path = os.path.join(mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            continue

        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        split_type = "train" if i < split_index else "val"

        cv2.imwrite(os.path.join(BASE_OUT, split_type, "images", new_name), img)
        cv2.imwrite(os.path.join(BASE_OUT, split_type, "masks", new_name), mask)


def process_kvasir():
    print("Processing Kvasir-SEG...")
    img_dir = os.path.join(PROJECT_ROOT, "datasets", "Kvasir-SEG", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "datasets", "Kvasir-SEG", "masks")

    imgs = glob(os.path.join(img_dir, "*"))
    process_dataset(imgs, mask_dir, prefix="kvasir")

def process_cvc():
    print("Processing CVC-ClinicDB...")
    img_dir = os.path.join(PROJECT_ROOT, "datasets", "CVC-clinic", "Original")
    mask_dir = os.path.join(PROJECT_ROOT, "datasets", "CVC-clinic", "Ground Truth")

    imgs = glob(os.path.join(img_dir, "*"))
    process_dataset(imgs, mask_dir, prefix="cvc")

if __name__ == "__main__":
    print("Starting UNet preprocessing...")
    ensure_dirs()
    process_kvasir()
    process_cvc()
    print("✅ UNet preprocessing completed successfully!")