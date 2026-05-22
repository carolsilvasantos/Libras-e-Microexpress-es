"""
Utility for drawing bounding boxes and labels for YOLOv8 object detection.
"""
import cv2

def draw_detections(frame, results):
    """Draws premium bounding boxes and labels on the frame from YOLOv8 results."""
    if not results or not hasattr(results, 'boxes'):
        return frame

    for box in results.boxes:
        # Get coordinates
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1
        
        # Color palette (Cyberpunk Aqua)
        color = (200, 255, 0) # BGR
        thickness = 2
        
        # Draw corners (instead of full box for cleaner look)
        length = int(min(w, h) * 0.2)
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

        # Background box for aesthetics
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Label and score
        cls_id = int(box.cls[0])
        label = results.names[cls_id].upper()
        score = float(box.conf[0])
        result_text = f"{label} {score:.0%}"
        
        # Label background
        (tw, th), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, result_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_DUPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame
