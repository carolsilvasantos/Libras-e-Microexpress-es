"""
AI Object Detection & Inventory System v4.0 - WEB EDITION
Now accessible via HTTP with real-time video streaming and dashboard.
"""

import cv2
import time
import logging
import os
import threading
from flask import Flask, render_template, Response, jsonify

from src.capture.video_stream import VideoStream
from src.inference.engine import ObjectDetector
from src.utils.visualizer import draw_detections
from src.utils.tracker import CentroidTracker
from src.utils.stock_manager import StockManager

# Configure Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("InventoryWeb")

app = Flask(__name__)

class DetectorApp:
    def __init__(self, camera_idx=0):
        self.vs = VideoStream(src=camera_idx).start()
        self.detector = ObjectDetector(min_confidence=0.60)
        self.tracker = CentroidTracker(max_disappeared=30)
        self.stock = StockManager()

        # State
        self.output_frame = None
        self.lock = threading.Lock()
        self.obj_labels = {}
        self.crossed_ids = set()

        # Metrics
        self.fps = 0.0
        self.prev_time = time.time()

    def _process_tracking(self, frame, result):
        """Extract bounding boxes using YOLOv8 API and run centroid tracking."""
        h, w, _ = frame.shape
        line_x = w // 2
        rects = []
        temp_labels = []

        # YOLOv8 API: result.boxes (not result.detections)
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                # Only track allowed categories
                if label in self.stock.allowed_categories:
                    rects.append((x1, y1, x2, y2))
                    temp_labels.append(label)

        objects = self.tracker.update(rects)
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            for (obj_id, centroid) in objects.items():
                if centroid[0] == cX and centroid[1] == cY:
                    self.obj_labels[obj_id] = temp_labels[i]

        for (obj_id, centroid) in objects.items():
            label = self.obj_labels.get(obj_id)
            if not label or label not in self.stock.allowed_categories:
                continue
            history = self.tracker.positions.get(obj_id, [])
            if len(history) < 2:
                continue
            prev_x, curr_x = history[-2][0], centroid[0]
            if obj_id not in self.crossed_ids:
                if prev_x < line_x and curr_x >= line_x:
                    if self.stock.add_item(label):
                        self.crossed_ids.add(obj_id)
                elif prev_x > line_x and curr_x <= line_x:
                    if self.stock.remove_item(label):
                        self.crossed_ids.add(obj_id)
        return line_x

    def _draw_hud(self, frame, line_x):
        cv2.line(frame, (line_x, 40), (line_x, frame.shape[0] - 40), (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 1)

    def run_detection(self):
        logger.info("Background detection thread started.")
        while True:
            frame = self.vs.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            result, _ = self.detector.detect(frame)

            line_x = self._process_tracking(frame, result)
            draw_detections(frame, result)
            self._draw_hud(frame, line_x)

            # Metrics
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.prev_time, 0.001))
            self.prev_time = now

            with self.lock:
                self.output_frame = frame.copy()


detector_app = DetectorApp()


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    while True:
        with detector_app.lock:
            if detector_app.output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', detector_app.output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/inventory')
def api_inventory():
    return jsonify(detector_app.stock.get_stock())


if __name__ == "__main__":
    # Start detection in background
    t = threading.Thread(target=detector_app.run_detection, daemon=True)
    t.start()

    # Start Flask server
    logger.info("Iniciando servidor web em http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
