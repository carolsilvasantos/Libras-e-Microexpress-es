import cv2
import time
import requests
import json
import threading
from src.inference.engine import ObjectDetector
from src.utils.visualizer import draw_detections

# Configuration
API_URL = "http://localhost:8000/api/v1/internal/broadcast"
CAMERA_IDX = 0 # 0 for default webcam, or path to video file
MIN_CONFIDENCE = 0.5

class EdgeDevice:
    def __init__(self):
        self.detector = ObjectDetector(min_confidence=MIN_CONFIDENCE)
        self.cap = cv2.VideoCapture(CAMERA_IDX)
        self.running = True

    def start(self):
        print(f"[*] Starting Edge Device Simulation (Camera {CAMERA_IDX})")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[!] Failed to grab frame")
                break

            # 1. Run YOLOv8 Inference
            result, latency = self.detector.detect(frame)

            if result and hasattr(result, 'boxes'):
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    conf = float(box.conf[0])

                    # 2. If object detected, broadcast to Backend
                    payload = {
                        "type": "DETECTION",
                        "object": label,
                        "confidence": conf,
                        "timestamp": time.time(),
                        "latency_ms": latency
                    }
                    
                    try:
                        # Asynchronous-like POST request (could use a queue for better performance)
                        requests.post(API_URL, json=payload, timeout=0.1)
                    except Exception as e:
                        # Silently fail if backend is down
                        pass

            # 3. Visual Feedback (Optional for Edge)
            annotated_frame = draw_detections(frame.copy(), result)
            cv2.imshow("Edge Vision System (YOLOv8)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("[*] Edge Device stopped.")

if __name__ == "__main__":
    device = EdgeDevice()
    device.start()
