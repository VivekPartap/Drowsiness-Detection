"""
utils/config.py
Central configuration for the driver drowsiness detection system.
"""

from pathlib import Path


BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODEL_PATH: Path = BASE_DIR / "models" / "drowsiness_model.h5"
ALARM_PATH: Path = BASE_DIR / "alarm" / "alarm.wav"
LOG_PATH: Path = BASE_DIR / "logs" / "detection_logs.csv"

CAMERA_INDEX: int = 0

# Model / inference
IMG_SIZE: tuple[int, int] = (224, 224)
CONFIDENCE_THRESHOLD: float = 0.5
MIN_PREDICTION_CONFIDENCE: float = 0.65

LABELS: dict[int, str] = {
    0: "OPEN",
    1: "CLOSED",
}

# EAR drowsiness signal
EAR_THRESHOLD: float = 0.22
EAR_CONSEC_FRAMES: int = 20

# Temporal smoothing
SMOOTHING_WINDOW: int = 10
SMOOTHING_MIN_SAMPLES: int = 5

# Face reliability / head-pose context. Pose lowers reliability; it is not a
# drowsiness rule by itself.
HEAD_YAW_SOFT_DEGREES: float = 18.0
HEAD_YAW_HARD_DEGREES: float = 42.0
HEAD_PITCH_SOFT_DEGREES: float = 18.0
HEAD_PITCH_HARD_DEGREES: float = 45.0
HEAD_ROLL_SOFT_DEGREES: float = 15.0
HEAD_ROLL_HARD_DEGREES: float = 35.0
MIN_EYE_VISIBILITY: float = 0.55
MIN_RELIABILITY_FOR_CNN: float = 0.35
UNCERTAIN_RELIABILITY_THRESHOLD: float = 0.28

# Final decision
DROWSY_CNN_RATIO: float = 0.7
DROWSY_FRAME_LIMIT: int = EAR_CONSEC_FRAMES

# Alarm
ALARM_COOLDOWN_SECONDS: float = 5.0

# Display
WINDOW_TITLE: str = "Driver Drowsiness Detection"
FONT_SCALE: float = 0.65
FONT_THICKNESS: int = 2

# Overlay colour palette (BGR)
COLOR_GREEN: tuple[int, int, int] = (0, 220, 0)
COLOR_RED: tuple[int, int, int] = (0, 0, 220)
COLOR_YELLOW: tuple[int, int, int] = (0, 200, 220)
COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: tuple[int, int, int] = (0, 0, 0)
