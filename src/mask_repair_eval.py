import cv2
import numpy as np
import os
import csv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

BEFORE_DIR = os.path.join(
    PROJECT_ROOT,
    "runs", "segment", "predict_with_binary", "binary_masks"
)

AFTER_DIR = os.path.join(
    PROJECT_ROOT,
    "results", "repaired_masks"
)

RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULT_DIR, "mask_repair_full_evaluation.csv")
MQI_CSV = os.path.join(RESULT_DIR, "mqi_results.csv")


needs_repair_masks = []

with open(MQI_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Quality"] == "Needs_Repair":
            needs_repair_masks.append(row["Mask_Name"])

print(f"[INFO] Evaluating {len(needs_repair_masks)} repaired masks")



def safe_binary(mask):
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8) * 255


def compute_hole_ratio(binary):
    h, w = binary.shape
    flood = binary.copy()

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    holes = flood_inv & binary

    hole_area = np.sum(holes == 255)
    mask_area = np.sum(binary == 255)

    if mask_area == 0:
        return 1.0

    return hole_area / mask_area


def compute_metrics(mask):
    binary = safe_binary(mask)
    if binary is None:
        return 0, 0, 0, 1, 0, 0

    total_pixels = binary.shape[0] * binary.shape[1]
    area_pixels = np.sum(binary == 255)

    if area_pixels == 0:
        return 0, 0, 0, 1, 0, 0

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0, 0, 0, 1, 0, 0

    cnt = max(contours, key=cv2.contourArea)

    contour_area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        compactness = 0
    else:
        compactness = (4 * np.pi * contour_area) / (perimeter ** 2)

    compactness = min(compactness, 1.0)

    hole_ratio = compute_hole_ratio(binary)
    hole_score = 1 - hole_ratio

    area_ratio = area_pixels / total_pixels
    area_score = min(area_ratio * 5, 1.0)

    mqi = (
        0.3 * area_score +
        0.5 * compactness +
        0.2 * hole_score
    )

    return (
        area_pixels,
        perimeter,
        compactness,
        hole_ratio,
        area_score,
        round(mqi, 4)
    )



print("[INFO] Starting mask repair evaluation...")

all_improvements = []

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Mask_Name",
        "Area_Before", "Area_After",
        "Perimeter_Before", "Perimeter_After",
        "Compactness_Before", "Compactness_After",
        "HoleRatio_Before", "HoleRatio_After",
        "AreaScore_Before", "AreaScore_After",
        "MQI_Before", "MQI_After",
        "MQI_Improvement",
        "Improvement_Percentage"
    ])

    for mask_name in needs_repair_masks:

        before_path = os.path.join(BEFORE_DIR, mask_name)
        after_path = os.path.join(AFTER_DIR, mask_name)

        if not os.path.exists(after_path):
            continue

        before = cv2.imread(before_path, 0)
        after = cv2.imread(after_path, 0)

        (area_b, peri_b, comp_b, hole_b, area_s_b, mqi_b) = compute_metrics(before)
        (area_a, peri_a, comp_a, hole_a, area_s_a, mqi_a) = compute_metrics(after)

        improvement = round(mqi_a - mqi_b, 4)

        improvement_percent = (
            round((improvement / mqi_b) * 100, 2) if mqi_b > 0 else 0
        )

        all_improvements.append(improvement)

        writer.writerow([
            mask_name,
            area_b, area_a,
            round(peri_b, 2), round(peri_a, 2),
            round(comp_b, 4), round(comp_a, 4),
            round(hole_b, 4), round(hole_a, 4),
            round(area_s_b, 4), round(area_s_a, 4),
            mqi_b, mqi_a,
            improvement,
            improvement_percent
        ])

        print(f"{mask_name} | MQI: {mqi_b} → {mqi_a} | Δ = {improvement}")



if all_improvements:
    mean_improvement = round(np.mean(all_improvements), 4)
    std_improvement = round(np.std(all_improvements), 4)

    print("\n[SUMMARY]")
    print("Mean MQI Improvement:", mean_improvement)
    print("Std MQI Improvement:", std_improvement)

print("\n[INFO] Evaluation completed successfully.")
print("[INFO] Results saved to:", CSV_PATH)