import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODELS_DIR = os.path.join(BASE_DIR, 'ml')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True) # created by ml/train_model.py
# os.makedirs(ASSETS_DIR, exist_ok=True) # created manually for alarm.wav

LOG_FILE = os.path.join(LOGS_DIR, 'events.log')
MODEL_PATH = os.path.join(MODELS_DIR, 'drowsiness_model.joblib')
DATASET_PATH = os.path.join(MODELS_DIR, 'drowsiness_dataset.csv')
ALARM_PATH = os.path.join(ASSETS_DIR, 'alarm.wav')
MEDIAPIPE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')

# --- Eye Aspect Ratio (EAR) Parameters ---
EAR_THRESHOLD = 0.25  # Threshold for eye closure (lower means more closed)
EAR_CONSEC_FRAMES = 30 # Number of consecutive frames below threshold to trigger drowsiness

# --- ML Model Parameters ---
# If the model is not found, a synthetic dataset will be generated and a model trained.
# This threshold can be used as a fallback if the ML model is not available or if
# ML model confidence is low.
ML_DROWSINESS_THRESHOLD = 0.5 # Probability threshold for ML model classification

# --- GUI Parameters ---
WINDOW_TITLE = "AI-Powered Drowsiness Detection System"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
VIDEO_FEED_WIDTH = 640
VIDEO_FEED_HEIGHT = 480
FPS_UPDATE_INTERVAL_MS = 1000  # Update FPS display every 1000 ms (1 second)
GRAPH_UPDATE_INTERVAL_MS = 500  # Update live EAR graph every 500 ms
MAX_GRAPH_POINTS = 100 # Maximum data points to display on the live EAR graph

# --- Alert System ---
ALARM_DURATION_SECONDS = 3 # How long the alarm plays

# --- Video Capture ---
WEBCAM_INDEX = 0 # Usually 0 for default webcam
VIDEO_INPUT_PATH = 'assets/drowsiness_test.mp4' # Set to a video file path (e.g., 'path/to/video.mp4') to use a video file instead of webcam
FPS_TARGET = 30 # Target frames per second for video processing

# --- Colors (for GUI) ---
COLOR_GREEN = "#4CAF50" # Awake status
COLOR_RED = "#F44336"   # Drowsy status
COLOR_BLUE = "#2196F3"  # Info/default status
COLOR_DARK_GREY = "#333333"
COLOR_LIGHT_GREY = "#EEEEEE"

# --- MediaPipe Parameters (adjusted for tasks API expectations) ---
# STATIC_IMAGE_MODE is not directly used by FaceLandmarker.create_from_options
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.7 # For min_face_detection_confidence
MIN_TRACKING_CONFIDENCE = 0.7  # For min_tracking_confidence and min_face_presence_confidence

# --- Logging Parameters ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
