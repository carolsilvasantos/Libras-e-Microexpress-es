# Real-Time Object Detection AI 🚀

A high-performance computer vision system for real-time object detection using MediaPipe and OpenCV.

## 🚀 Overview

This project implements a robust pipeline for detecting and tracking 90+ classes of everyday objects (people, vehicles, electronics, etc.) using the EfficientDet-Lite0 model.

### Key Features:
- **Ultra-Fast Inference**: Optimized for real-time performance on average CPUs.
- **Concurrent Capture**: Threaded video stream logic to prevent UI lag.
- **Dynamic HUD**: Real-time FPS monitoring and object counter.
- **Modular Architecture**: Clean, production-ready Python codebase following SOLID principles.

## 📁 Structure

- `src/capture`: Threaded video stream logic.
- `src/inference`: Object detection engine and MediaPipe integration.
- `src/utils`: Visualization tools for bounding boxes and labels.
- `main.py`: Entry point and integration loop.
- `models/`: Pre-trained TFLite models.

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

## 📜 Principles

- **Performance**: Asynchronous video capture ensures smooth 30+ FPS.
- **Extensibility**: Easily swappable models for different detection tasks.
- **Clean Code**: High readability and maintainability.

---
*Original project focused on Libras, now evolved into a general Computer Vision toolkit.*
