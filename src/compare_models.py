import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = ROOT / "runs" / "segment"

RESULTS_DIR = ROOT / "results"

def load_last_row(csv_path):
    return pd.read_csv(csv_path).iloc[-1]

models = {
    "U-Net": BASE / "unet" / "eval_metrics.csv",
    "U-Net++": BASE / "unetpp" / "eval_metrics.csv",
    "YOLOv8n": BASE / "yolov8n_train" / "eval_metrics.csv",
    "YOLOv8s": BASE / "yolov8s_train" / "eval_metrics.csv",
}

rows = []

for name, path in models.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    data = load_last_row(path)
    rows.append({
        "Model": name,
        "Total_Parameters": data["Total_Parameters"],
        "Model_Size_MB": data["Model_Size_MB"],
        "Avg_Inference_ms": data["Avg_Inference_ms"],
        "FPS": data["FPS"],
        "Dice": data["Dice"],
        "IoU": data["IoU"],
        "Precision": data["Precision"],
        "Recall": data["Recall"],
        "F1": data["F1"],
        "Pixel_Accuracy": data["Pixel_Accuracy"],
    })

comparison = pd.DataFrame(rows)

out_path = RESULTS_DIR / "lightweight_comparison.csv"
comparison.to_csv(out_path, index=False)

print("\n✅ Lightweight comparison saved to:")
print(out_path)
print("\n", comparison)