"""
Real-time Object Detection Entry Point.
Uses MediaPipe Object Detector for fast and accurate identification.
"""

import cv2
import numpy as np
import logging
import time
import os

from src.capture.video_stream import VideoStream
from src.inference.engine import ObjectDetectionEngine
from src.utils.visualizer import draw_detections

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")

def draw_hud(frame, fps, count):
    """Simple HUD for Object Detection."""
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
    cv2.putText(frame, f"OBJETOS: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
    cv2.putText(frame, "Pressione 'Q' para sair", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    logger.info("Iniciando Detector de Objetos...")
    
    vs = VideoStream(src=0).start()
    time.sleep(1.0) # Warm-up
    
    engine = ObjectDetectionEngine(model_path="models/efficientdet_lite0.tflite")
    
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue
            
            # Prepare frame
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            result = engine.process_frame(rgb)
            
            # Draw
            draw_detections(frame, result)
            
            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            
            obj_count = len(result.detections) if result and result.detections else 0
            draw_hud(frame, fps, obj_count)
            
            cv2.imshow("Detector de Objetos AI", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        logger.info("Encerrando...")
        vs.stop()
        engine.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
