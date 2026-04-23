"""
Placeholder classifier models for Libras gestures and facial microexpressions.

These models use TensorFlow/Keras with random weights. They exist to validate
the full pipeline (capture → extraction → buffer → inference → output) end-to-end.
Replace with trained models when real datasets are available.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label Maps
# ---------------------------------------------------------------------------

LIBRAS_LABELS = [
    "Olá", "Obrigado", "Por favor", "Sim", "Não",
    "Bom dia", "Boa noite", "Desculpa", "Ajuda", "Amor",
]

EMOTION_LABELS = [
    "Neutro", "Alegria", "Tristeza", "Raiva",
    "Surpresa", "Medo", "Nojo",
]


class _BasePlaceholderModel:
    """
    Base class that mimics a Keras‑like predict() interface using pure NumPy.
    No TensorFlow import is required, making CI and unit tests lightweight.
    """

    def __init__(self, labels: list[str], input_features: int, window_size: int):
        self.labels = labels
        self.num_classes = len(labels)
        self.input_features = input_features
        self.window_size = window_size

        # Fake dense weight matrix (input_features → num_classes)
        rng = np.random.default_rng(seed=42)
        self._W = rng.standard_normal((input_features, self.num_classes)).astype(np.float32) * 0.01
        self._b = np.zeros(self.num_classes, dtype=np.float32)

        logger.info(
            "%s initialised  –  classes=%d  features=%d  window=%d",
            self.__class__.__name__, self.num_classes, input_features, window_size,
        )

    # ---- softmax helper ---------------------------------------------------
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    # ---- public API -------------------------------------------------------
    def predict(self, sequence: np.ndarray) -> dict:
        """
        Parameters
        ----------
        sequence : np.ndarray
            Shape ``(1, window_size, input_features)``.

        Returns
        -------
        dict with keys ``label``, ``confidence``, ``probabilities``.
        """
        if sequence.shape[1:] != (self.window_size, self.input_features):
            logger.warning(
                "Shape mismatch: expected (1, %d, %d), got %s",
                self.window_size, self.input_features, sequence.shape,
            )
            return {"label": "N/A", "confidence": 0.0, "probabilities": {}}

        # Collapse temporal axis → mean pooling
        pooled = sequence[0].mean(axis=0)                   # (input_features,)
        logits = pooled @ self._W + self._b                   # (num_classes,)
        probs = self._softmax(logits)

        idx = int(np.argmax(probs))
        return {
            "label": self.labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {l: float(p) for l, p in zip(self.labels, probs)},
        }


# ---------------------------------------------------------------------------
# Concrete classifiers
# ---------------------------------------------------------------------------

class LibrasClassifier(_BasePlaceholderModel):
    """Placeholder LSTM‑like classifier for Libras signs."""

    # MediaPipe Holistic: 33 pose + 21 left‑hand + 21 right‑hand landmarks × 3 coords
    DEFAULT_FEATURES = (33 + 21 + 21) * 3   # = 225

    def __init__(self, window_size: int = 30, input_features: int | None = None):
        super().__init__(
            labels=LIBRAS_LABELS,
            input_features=input_features or self.DEFAULT_FEATURES,
            window_size=window_size,
        )


class EmotionClassifier(_BasePlaceholderModel):
    """Placeholder classifier for facial microexpressions (AU / emotion)."""

    # MediaPipe Face Mesh: 468 landmarks × 3 coords
    DEFAULT_FEATURES = 468 * 3   # = 1404

    def __init__(self, window_size: int = 30, input_features: int | None = None):
        super().__init__(
            labels=EMOTION_LABELS,
            input_features=input_features or self.DEFAULT_FEATURES,
            window_size=window_size,
        )


class LibrasAlphabetClassifier:
    """Actual classifier for Libras Alphabet using a trained Scikit-Learn model."""
    
    DEFAULT_FEATURES = 225

    def __init__(self, model_path: str = "models/alphabet_classifier.pkl", window_size: int = 30):
        self.model_path = model_path
        self.window_size = window_size
        self.model = None
        self.labels = []
        self.is_loaded = False
        
        self.load_model()

    def load_model(self):
        import os
        import pickle
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.labels = data["labels"]
                    self.is_loaded = True
                logger.info(f"Loaded LibrasAlphabetClassifier from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load alphabet classifier: {e}")
        else:
            logger.warning(f"Alphabet classifier model not found at {self.model_path}. Collection needed.")

    def predict(self, sequence: np.ndarray) -> dict:
        if not self.is_loaded:
            return {"label": "N/A", "confidence": 0.0, "probabilities": {}}
            
        # For static alphabet signs, we can use the mean of the temporal window
        # or just the latest frame. We'll use the latest frame to be more responsive.
        latest_frame = sequence[0, -1].reshape(1, -1)
        
        try:
            probs = self.model.predict_proba(latest_frame)[0]
            label = self.model.predict(latest_frame)[0]
            confidence = float(np.max(probs))
            
            return {
                "label": str(label),
                "confidence": confidence,
                "probabilities": {str(l): float(p) for l, p in zip(self.model.classes_, probs)}
            }
        except Exception as e:
            logger.error(f"Prediction error in AlphabetClassifier: {e}")
            return {"label": "Error", "confidence": 0.0, "probabilities": {}}
