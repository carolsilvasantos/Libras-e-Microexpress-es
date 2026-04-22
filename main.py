import cv2
import logging
import time
from src.capture.video_stream import VideoStream
from src.inference.engine import InferenceEngine
from src.utils.feature_extractor import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main application loop."""
    logging.info("Starting Libras & Microexpressions System...")
    
    try:
        # Initialize components
        vs = VideoStream(src=0).start()
        engine = InferenceEngine(window_size=30)
        
        while True:
            frame = vs.read()
            if frame is None:
                continue
                
            # Mirror frame for natural feel
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Landmark Extraction
            results = engine.process_frame(rgb_frame)
            
            # 2. Visualization (Skeleton Overlay)
            if results.face_landmarks:
                # Add visualization logic here
                pass
            
            # 3. Buffer Update for Temporal Inference
            if results.pose_landmarks:
                norm_features = FeatureExtractor.normalize_landmarks(results.pose_landmarks.landmark)
                engine.update_buffer(norm_features)
            
            # 4. Inference (Placeholder logic)
            if engine.is_buffer_ready():
                # seq = engine.get_sequence()
                # prediction = classifier.predict(seq)
                cv2.putText(frame, "Analysing Sequence...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display Feedback
            cv2.imshow("Libras-Microexpressions AI", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logging.error("Critical System Failure: %s", str(e), exc_info=True)
    finally:
        logging.info("Shutting down system...")
        if 'vs' in locals(): vs.stop()
        if 'engine' in locals(): engine.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
