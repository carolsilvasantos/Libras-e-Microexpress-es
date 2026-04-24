"""
Enhanced feature extractor supporting separate extraction streams for
hands+pose (Libras) and face mesh (emotion/microexpressions).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Normalizes landmarks to be invariant to camera distance and position."""

    @staticmethod
    def _normalize(coords: np.ndarray, ref_idx: int, scale_a: int, scale_b: int) -> np.ndarray:
        """1. Translate to ref_point. 2. Scale by distance(a, b)."""
        if coords.shape[0] == 0:
            return coords
            
        ref_point = coords[ref_idx] if ref_idx < len(coords) else coords[0]
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
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]) if landmarks else np.zeros((0, 3))

    @staticmethod
    def extract_libras_features(result) -> np.ndarray | None:
        """
        Concatenates pose (33), left-hand (21), and right-hand (21) landmarks.
        Returns data if ANY component exists.
        """
        # 1. Pose
        if result.pose_landmarks:
            pose = FeatureExtractor._landmarks_to_array(result.pose_landmarks)
            pose = FeatureExtractor._normalize(pose, ref_idx=0, scale_a=11, scale_b=12)
        else:
            pose = np.zeros((33, 3))

        # 2. Left Hand
        if result.left_hand_landmarks:
            lh = FeatureExtractor._landmarks_to_array(result.left_hand_landmarks)
            lh = FeatureExtractor._normalize(lh, ref_idx=0, scale_a=0, scale_b=9)
        else:
            lh = np.zeros((21, 3))

        # 3. Right Hand
        if result.right_hand_landmarks:
            rh = FeatureExtractor._landmarks_to_array(result.right_hand_landmarks)
            rh = FeatureExtractor._normalize(rh, ref_idx=0, scale_a=0, scale_b=9)
        else:
            rh = np.zeros((21, 3))

        # Check if we actually found anything (not just zeros)
        # We consider it "found" if at least one landmark is non-zero
        all_vecs = np.concatenate([pose, lh, rh])
        if np.all(all_vecs == 0):
            return None

        return all_vecs.flatten()

    @staticmethod
    def extract_face_features(result) -> np.ndarray | None:
        if not result.face_landmarks:
            return None
        face = FeatureExtractor._landmarks_to_array(result.face_landmarks)
        face = FeatureExtractor._normalize(face, ref_idx=1, scale_a=133, scale_b=362)
        return face.flatten()
