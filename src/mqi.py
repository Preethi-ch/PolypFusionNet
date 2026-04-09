import cv2
import numpy as np
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MASK_DIR = os.path.join(
    PROJECT_ROOT, "runs", "segment", "predict_with_binary", "binary_masks"
)

RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULT_DIR, "mqi_results.csv")



def preprocess_mask(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return binary
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(binary)
    cleaned[labels == largest_label] = 255
    return cleaned

def compute_area_score(binary):
    area = np.sum(binary == 255)
    img_area = binary.shape[0] * binary.shape[1]
    return min((area / (img_area + 1e-6)) * 5, 1.0)

def compute_smoothness_score(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter==0:
        return 0.0
    return min((4 * np.pi * area) / (perimeter**2 + 1e-6), 1.0)

def compute_hole_score(binary):
    flood = binary.copy()
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood) & binary
    hole_ratio = np.sum(holes == 255) / (np.sum(binary == 255) + 1e-6)
    return max(1 - hole_ratio, 0)

def compute_mqi(mask):
    binary = preprocess_mask(mask)
    mqi = (
        0.3 * compute_area_score(binary) +
        0.5 * compute_smoothness_score(binary) +
        0.2 * compute_hole_score(binary)
    )
    if mqi >= 0.75:
        quality = "Good"
    elif mqi >= 0.5:
        quality = "Needs_Repair"
    else:
        quality = "Reject"
    return round(mqi, 3), quality



def compute_mqi_from_mask(mask):
    return compute_mqi(mask)
