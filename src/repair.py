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
REPAIRED_DIR = os.path.join(RESULT_DIR, "repaired_masks")

os.makedirs(REPAIRED_DIR, exist_ok=True)

MQI_CSV = os.path.join(RESULT_DIR, "mqi_results.csv")
REJECT_CSV = os.path.join(RESULT_DIR, "rejected_masks.csv")




def keep_largest_component(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(binary)
    cleaned[labels == largest_label] = 255
    return cleaned


def remove_small_components(binary, min_ratio=0.001):
    image_area = binary.shape[0] * binary.shape[1]
    min_area = image_area * min_ratio

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def fill_holes(binary):
    flood = binary.copy()
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    return binary | cv2.bitwise_not(flood)


def smooth_edges(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def repair_mask(mask):
    """
    Core repair logic.
    Used in BOTH research mode and demo mode.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    binary = remove_small_components(binary)
    binary = keep_largest_component(binary)
    binary = fill_holes(binary)
    binary = smooth_edges(binary)
    return binary




def repair_mask_in_memory(mask):
    """
    Used in demo_run.py
    No file reading
    No file saving
    """
    return repair_mask(mask)




if __name__ == "__main__":

    needs_repair = []
    reject_cases = []

    if not os.path.exists(MQI_CSV):
        print("[ERROR] MQI results file not found.")
        exit()

    with open(MQI_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Quality"] == "Needs_Repair":
                needs_repair.append(row["Mask_Name"])
            elif row["Quality"] == "Reject":
                reject_cases.append((row["Mask_Name"], row["MQI_Score"]))

    print(f"[INFO] Masks to repair: {len(needs_repair)}")
    print(f"[INFO] Reject cases: {len(reject_cases)}")

   
    for mask_file in needs_repair:

        mask_path = os.path.join(MASK_DIR, mask_file)

        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            continue

        repaired = repair_mask(mask)

        save_path = os.path.join(REPAIRED_DIR, mask_file)
        cv2.imwrite(save_path, repaired)

        print(f"[REPAIRED] {mask_file}")

    
    with open(REJECT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mask_Name", "MQI_Score", "Reason"])

        for name, score in reject_cases:
            writer.writerow([name, score, "Low segmentation quality"])

    print("\n[INFO] Mask repair process completed successfully.")
    print("[INFO] Repaired masks saved to:", REPAIRED_DIR)
    print("[INFO] Reject cases logged to:", REJECT_CSV)