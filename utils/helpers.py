import cv2
import logging
import os
from datetime import datetime

# Assuming config.py is in the base directory
try:
    from config import LOG_FILE, LOG_FORMAT
except ImportError:
    print("Warning: config.py not found or could not be imported. Using default values for helpers.")
    LOG_FILE = 'events.log'
    LOG_FORMAT = "% असतांडर्ड_output:Writing utils/helpers.py
standard_error:
traceback:

def setup_logging():
    """Sets up logging to a file and console."""
    # Ensure logs directory exists
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def draw_face_landmarks(image, landmarks, connection_list=None, color=(0, 255, 0)):
    """Draws landmarks and optional connections on the image."""
    if landmarks:
        for lm in landmarks:
            cv2.circle(image, lm, 1, color, -1) # Draw individual points
        if connection_list:
            # For MediaPipe FaceMesh, connections are often managed by mp.solutions.drawing_utils
            # but if custom connections are passed, draw them.
            for connection in connection_list:
                p1 = landmarks[connection[0]]
                p2 = landmarks[connection[1]]
                cv2.line(image, p1, p2, color, 1)
    return image

def calculate_fps(frame_count, start_time):
    """Calculates and returns the current frames per second."""
    end_time = datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    if time_diff > 0:
        fps = frame_count / time_diff
    else:
        fps = 0.0
    return fps

def display_status(frame, status_text, status_color=(0, 255, 0), ear_value=None, fps_value=None):
    """Displays status, EAR, and FPS on the video frame."""
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    if ear_value is not None:
        cv2.putText(frame, f"EAR: {ear_value:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if fps_value is not None:
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

if __name__ == '__main__':
    setup_logging()
    logging.info("Helpers module test complete.")
