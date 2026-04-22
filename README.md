# Libras & Microexpressions Recognition AI

Architectural skeleton for a real-time computer vision system integrating Brazilian Sign Language (Libras) detection with facial microexpression analysis.

## 🚀 Overview

This project implements a robust pipeline for:
- Concurrent webcam capture (threaded).
- Landmark extraction using MediaPipe Holistic.
- Distance-invariant feature normalization.
- Temporal sequence buffering for LSTM/Transformer-based classification.

## 📁 Structure

- `src/capture`: Threaded video stream logic.
- `src/inference`: MediaPipe integration and temporal buffer management.
- `src/utils`: Smoothing filters and normalization logic.
- `main.py`: Entry point and integration loop.

## 🛠️ Requirements

- Python 3.9+
- MediaPipe
- OpenCV
- NumPy
- TensorFlow/PyTorch

## 💻 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the main script:
   ```bash
   python main.py
   ```

## 📜 Principles

- **SOLID**: Each component has a single responsibility.
- **Performance**: Asynchronous landmarks extraction and video capture.
- **Robustness**: OneEuro Filter for jitter reduction and Z-score/Ref-point normalization.
