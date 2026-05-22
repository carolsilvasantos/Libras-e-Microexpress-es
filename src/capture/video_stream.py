import cv2
import threading
import logging
import time

class VideoStream:
    """Threaded camera capture to ensure the main loop doesn't block."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            logging.error("Failed to open video source %s", src)
            raise IOError(f"Cannot open webcam {src}")
            
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        """Starts the capture thread."""
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        """Internal loop to grab frames."""
        while not self.stopped:
            grabbed, frame = self.stream.read()
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            
            if not grabbed:
                logging.warning("Stream ended or camera disconnected.")
                self.stop()
                break
                
            # Slight sleep to reduce CPU usage if camera has low FPS
            time.sleep(0.001)

    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stops the thread and releases resources."""
        self.stopped = True
        self.stream.release()
