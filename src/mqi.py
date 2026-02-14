import os
import csv
import cv2
import numpy as np

LABEL_DIR = "runs/segment/predict_masks/labels"
OUTPUT_CSV = "results/mqi_scores.csv"
IMG_SIZE = 640

os.makedirs("results", exist_ok=True)


def polygon_to_mask(points):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Area", "Perimeter", "Compactness", "MQI"])

    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue

        with open(os.path.join(LABEL_DIR, label_file)) as lf:
            lines = lf.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) < 7:
                continue

            coords = list(map(float, data[1:]))

            if len(coords) % 2 != 0:
                coords = coords[:-1]

            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * IMG_SIZE)
                y = int(coords[i + 1] * IMG_SIZE)
                points.append((x, y))

            if len(points) < 3:
                continue

            mask = polygon_to_mask(points)

            area = cv2.countNonZero(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True)

            if perimeter == 0:
                continue

            compactness = (4 * np.pi * area) / (perimeter ** 2)
            mqi = min(compactness, 1.0)

            writer.writerow([
                label_file.replace(".txt", ".jpg"),
                area,
                round(perimeter, 2),
                round(compactness, 4),
                round(mqi, 4)
            ])

print(" MQI computation completed.")
print(f" Results saved at: {OUTPUT_CSV}")
