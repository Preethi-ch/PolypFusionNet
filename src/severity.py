import os
import csv

LABEL_DIR = "runs/segment/predict_masks/labels"
OUTPUT_CSV = "results/severity_scores.csv"

IMG_SIZE = 640

os.makedirs("results", exist_ok=True)


def polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def severity_level(score):
    """
    Severity based on relative polyp area
    Clinical inspiration:
    <5mm  -> low risk
    6–9mm -> moderate risk
    ≥10mm -> high risk
    """
    if score < 0.03:
        return "Mild"
    elif score < 0.10:
        return "Moderate"
    else:
        return "Severe"


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Severity Score", "Severity Level"])

    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(LABEL_DIR, label_file)

        with open(label_path, "r") as lf:
            lines = lf.readlines()

        total_polyp_area = 0.0

        for line in lines:
            data = line.strip().split()

            # Must have: class_id + at least 6 numbers (3 points)
            if len(data) < 7:
                continue

            coords = list(map(float, data[1:]))

            # Ensure even number of coordinates
            if len(coords) % 2 != 0:
                coords = coords[:-1]

            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * IMG_SIZE
                y = coords[i + 1] * IMG_SIZE
                points.append((x, y))

            if len(points) >= 3:
                total_polyp_area += polygon_area(points)

        image_area = IMG_SIZE * IMG_SIZE
        severity_score = total_polyp_area / image_area

        writer.writerow([
            label_file.replace(".txt", ".jpg"),
            round(severity_score, 4),
            severity_level(severity_score)
        ])

print("Severity scoring completed successfully.")
print(f"Results saved at: {OUTPUT_CSV}")
