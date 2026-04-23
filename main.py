"""
Libras & Microexpressions Recognition System — Main Entry Point.

Uses the new MediaPipe Tasks API (HolisticLandmarker) for landmark detection.
Integrates threaded webcam capture, dual temporal buffers, placeholder
classifiers, and a real-time HUD overlay.

Press 'q' to quit.
"""

import cv2
import numpy as np
import logging
import time
import os

from src.capture.video_stream import VideoStream
from src.inference.engine import InferenceEngine
from src.utils.data_collector import DataCollector
from src.utils.feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Drawing helpers (pure OpenCV — no dependency on mp.solutions)
# ---------------------------------------------------------------------------

# MediaPipe Holistic connection definitions for skeleton rendering
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


def _draw_landmarks_cv2(frame, landmarks, connections, color, thickness=2, radius=3):
    """Draw a set of landmarks and their connections on the frame using OpenCV."""
    if not landmarks:
        return
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], color, thickness)
    for pt in points:
        cv2.circle(frame, pt, radius, color, -1)


def draw_landmarks(frame, result):
    """Draw all detected landmarks on the BGR frame."""
    if result is None:
        return

    # Face mesh — thin teal dots
    if result.face_landmarks:
        h, w, _ = frame.shape
        for lm in result.face_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (80, 110, 10), -1)

    # Pose skeleton
    if result.pose_landmarks:
        _draw_landmarks_cv2(frame, result.pose_landmarks, POSE_CONNECTIONS,
                            color=(245, 117, 66), thickness=2, radius=3)

    # Left hand
    if result.left_hand_landmarks:
        _draw_landmarks_cv2(frame, result.left_hand_landmarks, HAND_CONNECTIONS,
                            color=(121, 22, 76), thickness=2, radius=2)

    # Right hand
    if result.right_hand_landmarks:
        _draw_landmarks_cv2(frame, result.right_hand_landmarks, HAND_CONNECTIONS,
                            color=(121, 44, 250), thickness=2, radius=2)


def draw_hud(frame, predictions: dict, fps: float):
    """Render a heads-up display with predictions and FPS."""
    h, w, _ = frame.shape

    # Semi-transparent bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

    # Libras prediction
    libras = predictions.get("libras")
    if libras and libras.get("label") != "N/A":
        label = libras["label"]
        conf = libras["confidence"]
        cv2.putText(frame, f"LIBRAS: {label}  ({conf:.0%})", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    else:
        cv2.putText(frame, "LIBRAS: Aguardando...", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

    # Emotion prediction
    emotion = predictions.get("emotion")
    if emotion and emotion.get("label") != "N/A":
        label = emotion["label"]
        conf = emotion["confidence"]
        cv2.putText(frame, f"EMOCAO: {label}  ({conf:.0%})", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    # Alphabet prediction
    alphabet = predictions.get("alphabet")
    if alphabet and alphabet.get("label") != "N/A":
        label = alphabet["label"]
        conf = alphabet["confidence"]
        cv2.putText(frame, f"ALFABETO: {label}  ({conf:.0%})", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)

def draw_collection_hud(frame, label: str, count: int, saved_msg: bool):
    """Render HUD for data collection mode."""
    cv2.putText(frame, f"COLETANDO: Letra {label} | Amostras: {count}", (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if saved_msg:
        cv2.putText(frame, "SALVO!", (15, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)



# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    """Application entry point."""
    logger.info("=" * 60)
    logger.info("  Libras & Microexpressions System  —  Starting")
    logger.info("=" * 60)

    vs = None
    engine = None

    try:
        # 1. Camera
        vs = VideoStream(src=0).start()
        time.sleep(1.0)  # warm-up
        logger.info("Camera stream started.")

        # 2. Inference engine (MediaPipe Tasks + classifiers)
        engine = InferenceEngine(
            model_path="models/holistic_landmarker.task",
            window_size=30,
        )
        
        # Data Collection Setup
        collector = DataCollector()
        current_label = 'A'
        samples_collected = 0
        show_saved_msg_until = 0

        prev_time = time.time()
        fps = 0.0

        while True:
            frame = vs.read()
            if frame is None:
                continue

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. MediaPipe processing
            result = engine.process_frame(rgb)

            # 4. Draw skeletons
            draw_landmarks(frame, result)

            # 5. Feed buffers + run classifiers
            engine.feed(result)
            predictions = engine.infer()

            # 6. FPS calculation (exponential moving average)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # 7. HUD overlay
            draw_hud(frame, predictions, fps)
            
            # Collection HUD
            draw_collection_hud(frame, current_label, samples_collected, time.time() < show_saved_msg_until)

            # 8. Display
            cv2.imshow("Libras & Microexpressions AI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User pressed 'q'. Exiting.")
                break
            elif ord('a') <= key <= ord('z'):
                current_label = chr(key).upper()
                logger.info(f"Target label changed to {current_label}")
            elif key == ord(' '):  # Spacebar to save
                features = FeatureExtractor.extract_libras_features(result)
                if features is not None:
                    if collector.save_sample(current_label, features):
                        samples_collected += 1
                        show_saved_msg_until = time.time() + 0.5
                        logger.info(f"Saved sample {samples_collected} for letter {current_label}")
                else:
                    logger.warning("Could not extract features. Make sure you are in frame.")

    except IOError as e:
        logger.error("Hardware failure (camera): %s", e, exc_info=True)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
    finally:
        logger.info("Shutting down...")
        if vs is not None:
            vs.stop()
        if engine is not None:
            engine.close()
        cv2.destroyAllWindows()
        logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
