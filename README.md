# PolypFusionNet: A Lightweight Multi-Task Framework for Real-Time Polyp Detection, Segmentation and Severity Estimation

## 📌 Overview

PolypFusionNet is a lightweight, end-to-end deep learning framework designed for real-time colorectal polyp analysis from colonoscopy images and videos. The system performs multiple clinical tasks within a single pipeline, including:

- Polyp Detection
- Polyp Segmentation
- Mask Quality Evaluation (MQI)
- Mask Repair
- Polyp Tracking
- Severity Estimation

The framework is built on **YOLOv8n-Seg**, enabling fast inference while maintaining competitive accuracy, making it suitable for real-time clinical applications.

---

## 🚀 Features

- ✅ Real-time polyp detection
- ✅ Accurate pixel-level segmentation
- ✅ Mask Quality Index (MQI) for segmentation reliability
- ✅ Automatic mask repair using morphological operations
- ✅ Polyp tracking across video frames
- ✅ Severity estimation (Mild, Moderate, Severe)
- ✅ Lightweight architecture for efficient deployment

---

## 🏗️ System Architecture

Input Colonoscopy Images / Videos
↓
YOLOv8n-Seg
↓
Detection + Segmentation
↓
Mask Quality Index (MQI)
↓
Mask Repair
↓
Feature Extraction
↓
Severity Estimation
↓
Polyp Tracking(for videos)
↓
Final Output

---

## 📂 Datasets

The model is trained and evaluated using publicly available datasets:

- **Kvasir-SEG**
- **CVC-ClinicDB**
- **PolypGen Video Dataset**

These datasets provide both image-based and video-based samples for detection, segmentation, and temporal analysis.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Streamlit
- Morphological Image Processing

---

## 📊 Evaluation Metrics

The framework is evaluated using:

- Precision
- Recall
- F1-Score
- Dice Coefficient
- Intersection over Union (IoU)
- Pixel Accuracy
- FPS (Frames Per Second)

---

## 📈 Results

| Metric | Value |
|---------|--------|
| Detection Precision | 91.61% |
| Dice Score | 68.10% |
| IoU | 63.20% |
| Pixel Accuracy | 95.93% |

### Key Achievements

- High detection accuracy
- Reliable segmentation masks
- Improved mask quality using MQI
- Stable video tracking
- Real-time performance using YOLOv8n-Seg

---



## 🔬 Future Improvements

- Evaluate on larger real-world clinical datasets
- Improve robustness against motion blur and lighting variations
- Enhance temporal consistency for long video sequences
- Deploy on embedded medical devices

---


## ⭐ Acknowledgements

I thank the creators of the following datasets and frameworks:

- Kvasir-SEG
- CVC-ClinicDB
- PolypGen
- Ultralytics YOLOv8
- PyTorch
