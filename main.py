"""
Libras & Microexpressions Recognition System — Main Entry Point.

Integrates threaded webcam capture, MediaPipe Holistic landmark extraction,
dual temporal buffers, placeholder classifiers, and a real-time HUD overlay.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
import time
import os

from src.capture.video_stream import VideoStream
from src.inference.engine import InferenceEngine

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
# Drawing helpers
# ---------------------------------------------------------------------------
MP_DRAWING = mp.solutions.drawing_utils
MP_HOLISTIC = mp.solutions.holistic
MP_DRAW_SPEC = MP_DRAWING.DrawingSpec

# Custom styles for skeleton overlay
FACE_SPEC = MP_DRAW_SPEC(color=(80, 110, 10), thickness=1, circle_radius=1)
HAND_SPEC = MP_DRAW_SPEC(color=(80, 22, 10), thickness=2, circle_radius=2)
POSE_SPEC = MP_DRAW_SPEC(color=(245, 117, 66), thickness=2, circle_radius=2)


def draw_landmarks(frame, results):
    """Draw MediaPipe skeletons on the BGR frame."""
    # Face mesh
    if results.face_landmarks:
        MP_DRAWING.draw_landmarks(
            frame, results.face_landmarks, MP_HOLISTIC.FACEMESH_CONTOURS,
            landmark_drawing_spec=FACE_SPEC, connection_drawing_spec=FACE_SPEC,
        )
    # Pose
    if results.pose_landmarks:
        MP_DRAWING.draw_landmarks(
            frame, results.pose_landmarks, MP_HOLISTIC.POSE_CONNECTIONS,
            landmark_drawing_spec=POSE_SPEC, connection_drawing_spec=POSE_SPEC,
        )
    # Left hand
    if results.left_hand_landmarks:
        MP_DRAWING.draw_landmarks(
            frame, results.left_hand_landmarks, MP_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_SPEC, connection_drawing_spec=HAND_SPEC,
        )
    # Right hand
    if results.right_hand_landmarks:
        MP_DRAWING.draw_landmarks(
            frame, results.right_hand_landmarks, MP_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_SPEC, connection_drawing_spec=HAND_SPEC,
        )


def draw_hud(frame, predictions: dict, fps: float):
    """Render a heads-up display with predictions and FPS."""
    h, w, _ = frame.shape

    # --- Semi-transparent bar at the top ---
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
    else:
        cv2.putText(frame, "EMOCAO: Aguardando...", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    """Application entry point."""
    logger.info("=" * 60)
    logger.info("  Libras & Microexpressions System  –  Starting")
    logger.info("=" * 60)

    vs = None
    engine = None

    try:
        # 1. Camera
        vs = VideoStream(src=0).start()
        time.sleep(1.0)   # warm-up
        logger.info("Camera stream started.")

        # 2. Inference engine (MediaPipe + classifiers)
        engine = InferenceEngine(window_size=30)

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
            results = engine.process_frame(rgb)

            # 4. Draw skeletons
            draw_landmarks(frame, results)

            # 5. Feed buffers + run classifiers
            engine.feed(results)
            predictions = engine.infer()

            # 6. FPS calculation
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # 7. HUD overlay
            draw_hud(frame, predictions, fps)

            # 8. Show
            cv2.imshow("Libras & Microexpressions AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User pressed 'q'. Exiting.")
                break

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
