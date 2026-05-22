import numpy as np
import time
import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Wrapper for YOLOv8 Object Detector.
    Uses standard YOLOv8n (nano) model for real-time performance.
    """

    def __init__(self, model_path: str = "yolov8n.pt", min_confidence: float = 0.50):
        """
        Initialize the YOLOv8 detector.
        If model_path is just a name (e.g. 'yolov8n.pt'), it will be downloaded automatically by ultralytics.
        """
        try:
            self.model = YOLO(model_path)
            self.min_confidence = min_confidence
            logger.info(f"YOLOv8 initialized with model: {model_path} (Confidence: {min_confidence * 100}%)")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray):
        """
        Processes a frame and returns detections.
        
        Returns:
            tuple: (results, latency_ms)
        """
        if frame is None:
            return None, 0.0

        start_inference = time.time()
        
        try:
            # results is a list of Results objects
            results = self.model.predict(
                source=frame,
                conf=self.min_confidence,
                verbose=False,
                device='cpu' # Use '0' for GPU if available
            )
            
            latency_ms = (time.time() - start_inference) * 1000
            
            # The UI normally expects a result object with a list of detections.
            # We'll return the first result in the list.
            return results[0], latency_ms
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None, 0.0

    def close(self):
        """No explicit close needed for Ultralytics YOLO loader, but kept for interface consistency."""
        logger.info("YOLOv8 released.")
