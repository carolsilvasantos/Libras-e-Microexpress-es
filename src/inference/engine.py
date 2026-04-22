"""
Enhanced InferenceEngine with dual temporal buffers (Libras + Emotion)
and integrated placeholder classifiers.
"""

import mediapipe as mp
import numpy as np
from collections import deque
import logging

from src.inference.classifier import LibrasClassifier, EmotionClassifier
from src.utils.feature_extractor import FeatureExtractor
from src.utils.filters import MovingAverageFilter

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages MediaPipe lifecycle, dual sequence buffers, and classifiers."""

    def __init__(self, window_size: int = 30):
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.window_size = window_size

        # Dual temporal buffers
        self._libras_buffer: deque = deque(maxlen=window_size)
        self._emotion_buffer: deque = deque(maxlen=window_size)

        # Smoothing filters (one per stream)
        self._libras_filter = MovingAverageFilter(window_size=3)
        self._emotion_filter = MovingAverageFilter(window_size=3)

        # Placeholder classifiers
        self.libras_clf = LibrasClassifier(window_size=window_size)
        self.emotion_clf = EmotionClassifier(window_size=window_size)

        # Last predictions (cached for rendering between ready-checks)
        self.last_libras_pred: dict | None = None
        self.last_emotion_pred: dict | None = None

        logger.info("InferenceEngine ready  –  window=%d frames", window_size)

    # ----- MediaPipe ---------------------------------------------------------
    def process_frame(self, rgb_frame: np.ndarray):
        """Runs MediaPipe Holistic on a single RGB frame."""
        if rgb_frame is None:
            return None
        return self.holistic.process(rgb_frame)

    # ----- Buffer management -------------------------------------------------
    def feed(self, results) -> None:
        """Extracts features from MediaPipe results and feeds both buffers."""
        libras_vec = FeatureExtractor.extract_libras_features(results)
        if libras_vec is not None:
            smoothed = self._libras_filter.apply(libras_vec)
            self._libras_buffer.append(smoothed)

        face_vec = FeatureExtractor.extract_face_features(results)
        if face_vec is not None:
            smoothed = self._emotion_filter.apply(face_vec)
            self._emotion_buffer.append(smoothed)

    # ----- Inference ---------------------------------------------------------
    def infer(self) -> dict:
        """
        Runs both classifiers when their buffers are full.

        Returns
        -------
        dict  – ``{"libras": {...} | None, "emotion": {...} | None}``
        """
        result = {"libras": self.last_libras_pred, "emotion": self.last_emotion_pred}

        if len(self._libras_buffer) == self.window_size:
            seq = np.expand_dims(np.array(self._libras_buffer), axis=0)
            self.last_libras_pred = self.libras_clf.predict(seq)
            result["libras"] = self.last_libras_pred

        if len(self._emotion_buffer) == self.window_size:
            seq = np.expand_dims(np.array(self._emotion_buffer), axis=0)
            self.last_emotion_pred = self.emotion_clf.predict(seq)
            result["emotion"] = self.last_emotion_pred

        return result

    # ----- Cleanup -----------------------------------------------------------
    def close(self) -> None:
        self.holistic.close()
        logger.info("InferenceEngine closed.")
