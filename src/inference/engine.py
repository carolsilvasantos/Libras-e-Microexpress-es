"""
Enhanced InferenceEngine using the MediaPipe Tasks API (0.10.30+).

Uses HolisticLandmarker in VIDEO running mode with detect_for_video().
"""

import numpy as np
import time
from collections import deque
import logging

from mediapipe.tasks.python.vision.holistic_landmarker import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe import Image, ImageFormat

from src.inference.classifier import LibrasClassifier, EmotionClassifier
from src.utils.feature_extractor import FeatureExtractor
from src.utils.filters import MovingAverageFilter

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages MediaPipe HolisticLandmarker lifecycle, dual sequence buffers, and classifiers."""

    def __init__(self, model_path: str = "models/holistic_landmarker.task", window_size: int = 30):
        # MediaPipe Tasks — VIDEO mode
        base_options = BaseOptions(model_asset_path=model_path)
        options = HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            output_face_blendshapes=False,
        )
        self.landmarker = HolisticLandmarker.create_from_options(options)

        self.window_size = window_size
        self._start_time = time.time()

        # Dual temporal buffers
        self._libras_buffer: deque = deque(maxlen=window_size)
        self._emotion_buffer: deque = deque(maxlen=window_size)

        # Smoothing filters
        self._libras_filter = MovingAverageFilter(window_size=3)
        self._emotion_filter = MovingAverageFilter(window_size=3)

        # Placeholder classifiers
        self._libras_num_features = (33 + 21 + 21) * 3  # 225
        self._emotion_num_features = None  # determined dynamically on first frame
        self.libras_clf = LibrasClassifier(window_size=window_size)
        self.emotion_clf = None  # lazy-init after we know face feature count

        # Cached predictions
        self.last_libras_pred: dict | None = None
        self.last_emotion_pred: dict | None = None

        logger.info("InferenceEngine ready  —  model=%s  window=%d", model_path, window_size)

    # ----- MediaPipe ---------------------------------------------------------
    def process_frame(self, rgb_frame: np.ndarray):
        """Runs HolisticLandmarker.detect_for_video on a single RGB frame."""
        if rgb_frame is None:
            return None

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        try:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            logger.warning("detect_for_video failed: %s", e)
            return None

        return result

    # ----- Buffer management -------------------------------------------------
    def feed(self, result) -> None:
        """Extracts features from HolisticLandmarkerResult and feeds both buffers."""
        if result is None:
            return

        libras_vec = FeatureExtractor.extract_libras_features(result)
        if libras_vec is not None:
            smoothed = self._libras_filter.apply(libras_vec)
            self._libras_buffer.append(smoothed)

        face_vec = FeatureExtractor.extract_face_features(result)
        if face_vec is not None:
            smoothed = self._emotion_filter.apply(face_vec)
            self._emotion_buffer.append(smoothed)

            # Lazy-init emotion classifier on first face vector
            if self.emotion_clf is None:
                self._emotion_num_features = len(face_vec)
                self.emotion_clf = EmotionClassifier(
                    window_size=self.window_size,
                    input_features=self._emotion_num_features,
                )
                logger.info("EmotionClassifier initialised with %d features", self._emotion_num_features)

    # ----- Inference ---------------------------------------------------------
    def infer(self) -> dict:
        """
        Runs both classifiers when their buffers are full.

        Returns dict ``{"libras": {...} | None, "emotion": {...} | None}``.
        """
        result = {"libras": self.last_libras_pred, "emotion": self.last_emotion_pred}

        if len(self._libras_buffer) == self.window_size:
            seq = np.expand_dims(np.array(self._libras_buffer), axis=0)
            self.last_libras_pred = self.libras_clf.predict(seq)
            result["libras"] = self.last_libras_pred

        if self.emotion_clf and len(self._emotion_buffer) == self.window_size:
            seq = np.expand_dims(np.array(self._emotion_buffer), axis=0)
            self.last_emotion_pred = self.emotion_clf.predict(seq)
            result["emotion"] = self.last_emotion_pred

        return result

    # ----- Cleanup -----------------------------------------------------------
    def close(self) -> None:
        self.landmarker.close()
        logger.info("InferenceEngine closed.")
