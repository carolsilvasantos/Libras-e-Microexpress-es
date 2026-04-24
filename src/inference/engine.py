"""
ObjectDetectionEngine using MediaPipe Tasks API.
Replaces the legacy Holistic Landmarker for general object detection.
"""

import numpy as np
import time
import logging
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe import Image, ImageFormat

logger = logging.getLogger(__name__)

class ObjectDetectionEngine:
    """Manages ObjectDetector lifecycle and inference."""

    def __init__(self, model_path: str = "models/efficientdet_lite0.tflite", score_threshold: float = 0.5):
        base_options = BaseOptions(model_asset_path=model_path)
        options = ObjectDetectorOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            score_threshold=score_threshold,
            max_results=5
        )
        self.detector = ObjectDetector.create_from_options(options)
        self._start_time = time.time()
        logger.info(f"ObjectDetectionEngine ready with model: {model_path}")

    def process_frame(self, rgb_frame: np.ndarray):
        """Detects objects in an RGB frame."""
        if rgb_frame is None:
            return None

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        try:
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            return result
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return None

    def close(self):
        self.detector.close()
        logger.info("ObjectDetectionEngine closed.")
