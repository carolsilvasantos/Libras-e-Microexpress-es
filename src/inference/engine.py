import mediapipe as mp
import numpy as np
from collections import deque
import logging

class InferenceEngine:
    """Manages MediaPipe lifecycle and sequence buffering."""
    def __init__(self, window_size=30):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size

    def process_frame(self, frame):
        """Extracts landmarks from a single frame."""
        if frame is None:
            return None
            
        results = self.holistic.process(frame)
        return results

    def update_buffer(self, flattened_landmarks):
        """Adds current landmarks to the temporal buffer."""
        self.buffer.append(flattened_landmarks)

    def is_buffer_ready(self):
        return len(self.buffer) == self.window_size

    def get_sequence(self):
        """Returns the current temporal sequence for classifier input."""
        return np.expand_dims(np.array(self.buffer), axis=0)

    def close(self):
        self.holistic.close()
