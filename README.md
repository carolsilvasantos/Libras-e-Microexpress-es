# Detector de Objetos AI 🚀

A high-performance computer vision system for real-time object detection using MediaPipe and OpenCV, now with premium features and a sleek HUD.

## 🚀 Overview

This project implements a robust pipeline for detecting and tracking 90+ classes of everyday objects (people, vehicles, electronics, etc.) using the EfficientDet-Lite0 model.

### Key Features:
- **Premium HUD**: High-tech translucent interface with real-time FPS and object counter.
- **Sleek Graphics**: Stylized bounding boxes with dynamic labels and cyberpunk aesthetics.
- **Snapshot Support**: Press `S` to save instant high-quality captures in `captures/`.
- **Video Recording**: Press `R` to toggle session recording into `recordings/`.
- **Ultra-Fast Inference**: Optimized for real-time performance on average CPUs.
- **Modular Architecture**: Clean, production-ready Python codebase following SOLID principles.

## 📁 Structure

- `src/capture`: Threaded video stream logic.
- `src/inference`: Object detection engine and MediaPipe integration.
- `src/utils`: Advanced visualization tools.
- `main.py`: Entry point and UI loop.
- `captures/`: Saved snapshots.
- `recordings/`: Video recordings.
- `models/`: Pre-trained TFLite models.

## 💻 Keyboard Controls

- `Q`: Quit application
- `S`: Take a snapshot (Saved to `captures/`)
- `R`: Toggle recording (Saved to `recordings/`)

## 🛠️ Requirements

- Python 3.9+
- MediaPipe >= 0.10.x
- OpenCV >= 4.x
- NumPy

## 💻 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the main script**:
   ```bash
   python main.py
   ```
