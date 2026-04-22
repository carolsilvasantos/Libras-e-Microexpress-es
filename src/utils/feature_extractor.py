"""
Enhanced feature extractor supporting separate extraction streams for
hands+pose (Libras) and face mesh (emotion/microexpressions).
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

    # ----- Libras stream (pose + both hands) ---------------------------------
    @staticmethod
    def extract_libras_features(results) -> np.ndarray | None:
        """
        Concatenates pose (33 × 3), left-hand (21 × 3), and right-hand (21 × 3)
        landmarks into a single flat vector of length 225.

        Returns ``None`` when pose landmarks are missing (hands are zero-padded).
        """
        if results.pose_landmarks is None:
            return None

        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        # Normalise by shoulder width (landmarks 11 and 12)
        pose = FeatureExtractor._normalize(pose, ref_idx=0, scale_a=11, scale_b=12)

        # Hands — zero-pad if absent
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            lh = FeatureExtractor._normalize(lh, ref_idx=0, scale_a=0, scale_b=9)
        else:
            lh = np.zeros((21, 3))

        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            rh = FeatureExtractor._normalize(rh, ref_idx=0, scale_a=0, scale_b=9)
        else:
            rh = np.zeros((21, 3))

        return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

    # ----- Emotion stream (face mesh) ----------------------------------------
    @staticmethod
    def extract_face_features(results) -> np.ndarray | None:
        """
        Extracts all 468 face-mesh landmarks → flat vector of length 1404.

        Returns ``None`` when face landmarks are missing.
        """
        if results.face_landmarks is None:
            return None

        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        # Normalise by distance between inner eye corners (landmarks 133 and 362)
        face = FeatureExtractor._normalize(face, ref_idx=1, scale_a=133, scale_b=362)

        return face.flatten()

    # ----- legacy convenience wrapper ----------------------------------------
    @staticmethod
    def normalize_landmarks(landmarks, reference_point_idx=0):
        """
        Kept for backward compatibility with earlier pipeline code.
        """
        if landmarks is None or len(landmarks) == 0:
            return None

        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        ref_point = coords[reference_point_idx]
        normalized = coords - ref_point
        scale_factor = np.linalg.norm(coords[33] - coords[133]) if len(coords) > 133 else 1.0
        if scale_factor > 0:
            normalized /= scale_factor
        return normalized.flatten()
