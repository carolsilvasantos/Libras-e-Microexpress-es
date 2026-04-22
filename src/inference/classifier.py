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
