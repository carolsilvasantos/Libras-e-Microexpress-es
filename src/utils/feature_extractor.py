import numpy as np

class FeatureExtractor:
    """Normalizes landmarks to be invariant to camera distance and position."""
    
    @staticmethod
    def normalize_landmarks(landmarks, reference_point_idx=0):
        """
        Translates landmarks to center them at a reference point (e.g. nose).
        Scales landmarks by the distance between two reference points.
        """
        if landmarks is None or len(landmarks) == 0:
            return None
            
        # Convert landmarks to numpy array (x, y, z)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # 1. Translation: Center landmarks at reference point
        ref_point = coords[reference_point_idx]
        normalized = coords - ref_point
        
        # 2. Scaling: Normalize by distance between points (e.g. eye distance or shoulder width)
        # Using a fixed divisor for the skeleton; in production, this is dynamic.
        scale_factor = np.linalg.norm(coords[33] - coords[133]) if len(coords) > 133 else 1.0 # distance between eye corners
        if scale_factor > 0:
            normalized = normalized / scale_factor
            
        return normalized.flatten()
