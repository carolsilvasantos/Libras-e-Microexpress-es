"""
Utility for drawing bounding boxes and labels for object detection.
"""
import cv2

def draw_detections(frame, detection_result):
    """Draws bounding boxes and labels on the frame."""
    if not detection_result or not detection_result.detections:
        return frame

    for detection in detection_result.detections:
        # Get bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        
        # Draw box
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw label and score
        category = detection.categories[0]
        label = category.category_name
        score = category.score
        result_text = f"{label} ({score:.0%})"
        
        text_location = (bbox.origin_x + 5, bbox.origin_y + 20)
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)

    return frame
