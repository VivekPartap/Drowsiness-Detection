import sys
import os
import cv2
import time
import threading
import logging
import collections
import numpy as np # Import numpy for isfinite check

# Ensure the current working directory is in sys.path for module imports
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"DEBUG (main.py): Current working directory: {os.getcwd()}")
print(f"DEBUG (main.py): sys.path before importing config: {sys.path}")
print(f"DEBUG (main.py): config.py exists: {os.path.exists('config.py')}")

import config # Import the config module directly

# --- Added Debugging for detector, ml, and alert directories ---
print(f"DEBUG (main.py): Checking detector directory at {os.path.join(current_dir, 'detector')}")
print(f"DEBUG (main.py): detector directory exists: {os.path.isdir(os.path.join(current_dir, 'detector'))}")
print(f"DEBUG (main.py): detector/__init__.py exists: {os.path.exists(os.path.join(current_dir, 'detector', '__init__.py'))}")
print(f"DEBUG (main.py): detector/face_mesh.py exists: {os.path.exists(os.path.join(current_dir, 'detector', 'face_mesh.py'))}")
print(f"DEBUG (main.py): Contents of detector directory: {os.listdir(os.path.join(current_dir, 'detector')) if os.path.isdir(os.path.join(current_dir, 'detector')) else 'N/A'}")

print(f"DEBUG (main.py): Checking ml directory at {os.path.join(current_dir, 'ml')}")
print(f"DEBUG (main.py): ml directory exists: {os.path.isdir(os.path.join(current_dir, 'ml'))}")
print(f"DEBUG (main.py): ml/__init__.py exists: {os.path.exists(os.path.join(current_dir, 'ml', '__init__.py'))}")
print(f"DEBUG (main.py): ml/model.py exists: {os.path.exists(os.path.join(current_dir, 'ml', 'model.py'))}")
print(f"DEBUG (main.py): Contents of ml directory: {os.listdir(os.path.join(current_dir, 'ml')) if os.path.isdir(os.path.join(current_dir, 'ml')) else 'N/A'}")

print(f"DEBUG (main.py): Checking alert directory at {os.path.join(current_dir, 'alert')}")
print(f"DEBUG (main.py): alert directory exists: {os.path.isdir(os.path.join(current_dir, 'alert'))}")
print(f"DEBUG (main.py): alert/__init__.py exists: {os.path.exists(os.path.join(current_dir, 'alert', '__init__.py'))}")
print(f"DEBUG (main.py): alert/alarm.py exists: {os.path.exists(os.path.join(current_dir, 'alert', 'alarm.py'))}")
print(f"DEBUG (main.py): Contents of alert directory: {os.listdir(os.path.join(current_dir, 'alert')) if os.path.isdir(os.path.join(current_dir, 'alert')) else 'N/A'}")
# --- End Added Debugging ---

from detector.face_mesh import FaceMeshDetector
from detector.eye_detector import EyeDetector
from ml.model import DrowsinessModel
from alert.alarm import AlarmSystem
from utils.ear import eye_aspect_ratio
from utils.helpers import setup_logging, draw_face_landmarks, display_status

# Check if running in Google Colab
IN_COLAB = 'COLAB_GPU' in os.environ

# Ensure necessary directories exist
os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.ASSETS_DIR, exist_ok=True)

# Setup global logging
setup_logging()

class VideoProcessor:
    """Manages video capture, frame processing, and drowsiness detection logic."""

    def __init__(self):
        self.cap = None
        self.running = False
        self.frame_thread = None
        self.latest_frame = None
        self.output_frame = None

        self.ear_consec_frames_counter = 0
        self.current_ear = 0.0
        self.drowsy_status = "Awake"
        self.start_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0
        self.ml_drowsiness_prob = 0.0

        # Initialize components
        self.face_mesh_detector = FaceMeshDetector()
        self.eye_detector = EyeDetector()
        self.ml_model = DrowsinessModel()
        self.alarm_system = AlarmSystem()

        logging.info("VideoProcessor initialized.")

    def start(self):
        """Starts the video capture and processing thread."""
        if self.running:
            logging.warning("Video processor is already running.")
            return False

        if config.VIDEO_INPUT_PATH and os.path.exists(config.VIDEO_INPUT_PATH):
            logging.info(f"Attempting to open video file: {config.VIDEO_INPUT_PATH}")
            self.cap = cv2.VideoCapture(config.VIDEO_INPUT_PATH)
        else:
            if config.VIDEO_INPUT_PATH and not os.path.exists(config.VIDEO_INPUT_PATH):
                logging.error(f"Video file not found at {config.VIDEO_INPUT_PATH}. Falling back to webcam.")
            logging.info(f"Attempting to open webcam at index {config.WEBCAM_INDEX}")
            self.cap = cv2.VideoCapture(config.WEBCAM_INDEX)

        if not self.cap.isOpened():
            error_msg = f"Failed to open video source. {'Check WEBCAM_INDEX' if not (config.VIDEO_INPUT_PATH and os.path.exists(config.VIDEO_INPUT_PATH)) else 'Check VIDEO_INPUT_PATH'}"
            logging.error(error_msg)
            self.running = False
            return False

        self.cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)

        self.running = True
        self.frame_thread = threading.Thread(target=self._process_frames)
        self.frame_thread.daemon = True
        self.frame_thread.start()
        logging.info("Video processing started.")
        return True

    def stop(self):
        """Stops the video capture and processing thread."""
        if not self.running:
            logging.warning("Video processor is not running.")
            return

        self.running = False
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.alarm_system.stop_alarm()
        logging.info("Video processing stopped.")

    def _process_frames(self):
        """Continuously processes frames from the webcam or video file."""
        self.start_time = time.time()
        self.frame_count = 0

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to grab frame or end of video stream. Exiting video processing.")
                self.running = False
                break

            self.frame_count += 1
            if time.time() - self.start_time >= 1.0:
                self.current_fps = self.frame_count / (time.time() - self.start_time)
                self.start_time = time.time()
                self.frame_count = 0

            img_height, img_width = frame.shape[:2]
            results, _ = self.face_mesh_detector.find_face_landmarks(frame)

            if results and results.face_landmarks:
                left_eye_landmarks, right_eye_landmarks = self.eye_detector.get_eye_landmarks(results, img_width, img_height)

                if left_eye_landmarks and right_eye_landmarks:
                    left_ear = eye_aspect_ratio(left_eye_landmarks)
                    right_ear = eye_aspect_ratio(right_eye_landmarks)
                    self.current_ear = (left_ear + right_ear) / 2.0

                    if not np.isfinite(self.current_ear) or self.current_ear < 0:
                        logging.warning(f"Invalid EAR value detected: {self.current_ear:.2f}. Skipping ML prediction for this frame.")
                        self.ml_drowsiness_prob = 0.0
                        self.ear_consec_frames_counter = 0
                    else:
                        self.ml_drowsiness_prob = self.ml_model.predict(self.current_ear, self.ear_consec_frames_counter)

                        if not IN_COLAB:
                            for p in left_eye_landmarks + right_eye_landmarks:
                                cv2.circle(frame, p, 1, (0, 255, 255), -1)

                        if self.ml_model.model and self.ml_drowsiness_prob >= config.ML_DROWSINESS_THRESHOLD:
                            self.drowsy_status = "Drowsy"
                            self.ear_consec_frames_counter += 1
                            logging.debug(f"ML Drowsy (Prob: {self.ml_drowsiness_prob:.2f}), EAR: {self.current_ear:.2f}, Consec Frames: {self.ear_consec_frames_counter})")
                        elif self.current_ear < config.EAR_THRESHOLD:
                            self.drowsy_status = "Drowsy"
                            self.ear_consec_frames_counter += 1
                            logging.debug(f"EAR Drowsy (EAR: {self.current_ear:.2f}), Consec Frames: {self.ear_consec_frames_counter})")
                        else:
                            self.drowsy_status = "Awake"
                            self.ear_consec_frames_counter = 0

                        if self.drowsy_status == "Drowsy" and self.ear_consec_frames_counter >= config.EAR_CONSEC_FRAMES:
                            self.alarm_system.trigger_alarm()

                else:
                    self.drowsy_status = "No Eyes Detected"
                    self.ear_consec_frames_counter = 0
                    self.current_ear = 0.0
                    self.ml_drowsiness_prob = 0.0
            else:
                self.drowsy_status = "No Face Detected"
                self.ear_consec_frames_counter = 0
                self.current_ear = 0.0
                self.ml_drowsiness_prob = 0.0
                self.alarm_system.stop_alarm()

            if not IN_COLAB:
                 display_status(frame, self.drowsy_status,
                                    config.COLOR_RED if self.drowsy_status == "Drowsy" else config.COLOR_GREEN,
                                    self.current_ear, self.current_fps)

            self.latest_frame = frame
            self.output_frame = frame.copy()

        logging.info("Video processing loop ended.")
        self.stop()

    def get_latest_frame(self):
        """Returns the most recently processed frame."""
        return self.latest_frame

    def get_current_status(self):
        """Returns the current drowsiness status, EAR, and FPS."""
        return self.drowsy_status, self.current_ear, self.current_fps

    def get_latest_ear(self):
        """Returns the latest calculated EAR value for graph."""
        return self.current_ear


def main():
    """Main function to set up and run the drowsiness detection system."""
    logging.info("Starting Drowsiness Detection System...")

    from ml.train_model import train_ml_model
    if not os.path.exists(os.path.join(config.MODELS_DIR, 'drowsiness_model.joblib')) or \
       not os.path.exists(os.path.join(config.MODELS_DIR, 'scaler.joblib')):
        logging.info("ML model not found, initiating training...")
        train_ml_model()
    else:
        logging.info("ML model already exists, skipping training.")

    processor = VideoProcessor()

    if IN_COLAB:
        logging.info("Running in Google Colab - GUI disabled. Processing video headlessly.")
        if not processor.start():
            logging.error("Failed to start video processor in Colab. Exiting.")
            return
        try:
            while processor.running:
                d_status, ear, fps = processor.get_current_status()
                logging.info(f"Headless Status: {d_status}, EAR: {ear:.2f}, FPS: {fps:.2f}")
                time.sleep(5)
        except KeyboardInterrupt:
            logging.info("Headless processing interrupted.")
        finally:
            processor.stop()

    else:
        from gui import DrowsinessApp

        app = DrowsinessApp(
            video_processor=processor,
            alarm_system=processor.alarm_system,
            on_start_callback=processor.start,
            on_stop_callback=processor.stop
        )

        logging.info("Launching GUI...")
        app.mainloop()
        logging.info("GUI closed.")

        processor.stop()

if __name__ == "__main__":
    main()
