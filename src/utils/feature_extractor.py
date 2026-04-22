"""
Enhanced feature extractor supporting separate extraction streams for
hands+pose (Libras) and face mesh (emotion/microexpressions).

Adapted for MediaPipe Tasks API (0.10.30+) where landmarks are returned
as lists of NormalizedLandmark dataclass objects with .x, .y, .z attributes.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Normalizes landmarks to be invariant to camera distance and position."""

    # ----- generic normaliser ------------------------------------------------
    @staticmethod
    def _normalize(coords: np.ndarray, ref_idx: int, scale_a: int, scale_b: int) -> np.ndarray:
        """
        1. Translate so that ``coords[ref_idx]`` is the origin.
        2. Scale by the distance between ``coords[scale_a]`` and ``coords[scale_b]``.
        """
        ref_point = coords[ref_idx]
        centred = coords - ref_point

        if max(scale_a, scale_b) < len(coords):
            scale = np.linalg.norm(coords[scale_a] - coords[scale_b])
        else:
            scale = 1.0

        if scale > 1e-6:
            centred /= scale

        return centred

    @staticmethod
    def _landmarks_to_array(landmarks) -> np.ndarray:
        """Convert a list of NormalizedLandmark objects to a numpy array."""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # ----- Libras stream (pose + both hands) ---------------------------------
    @staticmethod
    def extract_libras_features(result) -> np.ndarray | None:
        """
        Concatenates pose (33 × 3), left-hand (21 × 3), and right-hand (21 × 3)
        landmarks into a single flat vector of length 225.

        Returns ``None`` when pose landmarks are missing (hands are zero-padded).
        """
        if not result.pose_landmarks:
            return None

        pose = FeatureExtractor._landmarks_to_array(result.pose_landmarks)
        # Normalise by shoulder width (landmarks 11 and 12)
        pose = FeatureExtractor._normalize(pose, ref_idx=0, scale_a=11, scale_b=12)

        # Hands — zero-pad if absent
        if result.left_hand_landmarks:
            lh = FeatureExtractor._landmarks_to_array(result.left_hand_landmarks)
            lh = FeatureExtractor._normalize(lh, ref_idx=0, scale_a=0, scale_b=9)
        else:
            lh = np.zeros((21, 3))

        if result.right_hand_landmarks:
            rh = FeatureExtractor._landmarks_to_array(result.right_hand_landmarks)
            rh = FeatureExtractor._normalize(rh, ref_idx=0, scale_a=0, scale_b=9)
        else:
            rh = np.zeros((21, 3))

        return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

    # ----- Emotion stream (face mesh) ----------------------------------------
    @staticmethod
    def extract_face_features(result) -> np.ndarray | None:
        """
        Extracts all 478 face-mesh landmarks → flat vector of length 1434.

        Returns ``None`` when face landmarks are missing.
        """
        if not result.face_landmarks:
            return None

        face = FeatureExtractor._landmarks_to_array(result.face_landmarks)
        # Normalise by distance between inner eye corners (landmarks 133 and 362)
        safe_a = min(133, len(face) - 1)
        safe_b = min(362, len(face) - 1)
        face = FeatureExtractor._normalize(face, ref_idx=1, scale_a=safe_a, scale_b=safe_b)

        return face.flatten()
