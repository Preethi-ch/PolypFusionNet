import os
import csv
import cv2
import numpy as np



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PREDICTED_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "runs", "segment", "predict_with_binary", "binary_masks"
)

REPAIRED_MASK_DIR = os.path.join(
    PROJECT_ROOT, "results", "repaired_masks"
)

MQI_CSV = os.path.join(
    PROJECT_ROOT, "results", "mqi_results.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT, "results", "severity_results.csv"
)

MQI_GOOD_TH = 0.75
MQI_REPAIR_TH = 0.50




def severity_scoring(mask):
    """
    Core severity logic (USED IN BOTH MODES)
    Input: binary mask (numpy array)
    Output: severity_score, severity_label
    """

    binary = (mask > 0).astype(np.uint8)
    area = np.sum(binary)

    if area == 0:
        return 0.0, "No_Lesion"

    img_area = binary.shape[0] * binary.shape[1]
    rel_area = area / img_area

    
    if rel_area < 0.02:
        size_score = 1
    elif rel_area < 0.05:
        size_score = 2
    else:
        size_score = 3

    
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, "Invalid"

    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    if compactness < 1.3:
        shape_score = 1
    elif compactness < 1.6:
        shape_score = 2
    else:
        shape_score = 3

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smooth = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours_smooth, _ = cv2.findContours(
        smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours_smooth:
        cnt_smooth = max(contours_smooth, key=cv2.contourArea)
        perimeter_smooth = cv2.arcLength(cnt_smooth, True)
        roughness_ratio = perimeter / (perimeter_smooth + 1e-6)
    else:
        roughness_ratio = 1.0

    if roughness_ratio < 1.05:
        edge_score = 1
    elif roughness_ratio < 1.15:
        edge_score = 2
    else:
        edge_score = 3

   
    severity = (
        0.5 * size_score +
        0.3 * shape_score +
        0.2 * edge_score
    )

    if severity < 1.5:
        label = "Mild"
    elif severity < 2.3:
        label = "Moderate"
    else:
        label = "Severe"

    return round(severity, 3), label




def compute_severity_from_mask(mask):
    """
    Used in demo_run.py
    No file I/O, no CSV
    """
    return severity_scoring(mask)



if __name__ == "__main__":

    print("[INFO] Starting severity scoring pipeline...")

    with open(MQI_CSV, "r") as f:
        reader = csv.DictReader(f)
        mqi_rows = list(reader)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mask_Name",
            "MQI_Score",
            "MQI_Class",
            "Mask_Source",
            "Severity_Score",
            "Severity_Label"
        ])

        for row in mqi_rows:

            mask_name = row["Mask_Name"]
            mqi_score = float(row["MQI_Score"])

            if mqi_score >= MQI_GOOD_TH:
                mqi_class = "Good"
            elif mqi_score >= MQI_REPAIR_TH:
                mqi_class = "Needs_Repair"
            else:
                mqi_class = "Reject"

            if mqi_class == "Reject":
                continue

            if mqi_class == "Good":
                mask_path = os.path.join(PREDICTED_MASK_DIR, mask_name)
                mask_source = "Predicted"
            else:
                mask_path = os.path.join(REPAIRED_MASK_DIR, mask_name)
                mask_source = "Repaired"

            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, 0)
            if mask is None:
                continue

            severity, label = severity_scoring(mask)

            writer.writerow([
                mask_name,
                round(mqi_score, 3),
                mqi_class,
                mask_source,
                severity,
                label
            ])

            print(f"{mask_name} | {mask_source} → {label} ({severity:.2f})")

    print("\n✅ Severity scoring completed successfully.")
    print("Results saved to:", OUTPUT_CSV)