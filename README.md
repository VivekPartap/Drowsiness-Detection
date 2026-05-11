# Driver Drowsiness Detection System

A production-style, real-time driver drowsiness detection system built with Python, OpenCV, MediaPipe, and TensorFlow/Keras.

---

## Overview

The system reads a live webcam feed, detects the driver's face using **MediaPipe Face Mesh**, extracts eye regions, and classifies each eye as **OPEN** or **CLOSED** using a pre-trained CNN model (`drowsiness_model.h5`).  Predictions are temporally smoothed and combined with an **Eye Aspect Ratio (EAR)** check to produce a robust drowsiness decision.  When drowsiness is detected, an audible alarm fires, a banner appears on-screen, and every frame is logged to a CSV file for review.

---

## System Architecture

```
Webcam Frame
     │
     ▼
Face Detection (MediaPipe Face Mesh)
     │
     ├── Eye Crop Extraction ──► CNN Inference ──► Temporal Smoother ──┐
     │                                                                   │
     └── EAR Calculation ─────────────────────────────────────────────► Decision Engine
                                                                         │
                              ┌──────────────────────────────────────────┤
                              ▼                          ▼               ▼
                         Alarm Manager            Overlay Renderer   CSV Logger
```

---

## Project Structure

```
driver_drowsiness_system/
├── models/
│   └── drowsiness_model.h5         ← place your model here
├── alarm/
│   ├── alarm.wav                   ← auto-generated if missing
│   └── alarm_manager.py
├── detection/
│   ├── face_detector.py
│   ├── eye_extractor.py
│   ├── ear_calculator.py
│   └── landmarks.py
├── inference/
│   ├── predictor.py
│   ├── temporal_smoothing.py
│   └── decision_engine.py
├── realtime/
│   ├── realtime_detection.py       ← main pipeline
│   ├── webcam.py
│   └── overlay.py
├── utils/
│   ├── config.py
│   ├── preprocessing.py
│   ├── fps.py
│   └── logger.py
├── logs/
│   └── detection_logs.csv          ← created automatically
├── main.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10 or later
- A working webcam
- The trained model file (see below)

---

## Installation

```bash
# 1. Clone / download the project
cd driver_drowsiness_system

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **macOS (Apple Silicon)** users: install `tensorflow-macos` and `tensorflow-metal` instead of `tensorflow`.

---

## Model Placement

Copy your trained model file into the `models/` directory:

```
models/drowsiness_model.h5
```

The model must:
- Accept input of shape **(1, 224, 224, 3)** (float32, values in [0, 1])
- Output a sigmoid probability (scalar) for the CLOSED class, **or** a softmax vector of shape (1, 2)
- Use class mapping: **0 = OPEN**, **1 = CLOSED**

---

## Running the System

### Real-time detection (main mode)

```bash
python main.py
```

Use a specific camera index (e.g. external USB webcam):

```bash
python main.py --camera 1
```

Press **q** or **ESC** inside the video window to stop.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit the detector |
| `ESC` | Quit the detector |

---

## On-Screen Overlay

| Item | Description |
|------|-------------|
| **FPS** | Rolling average frames-per-second |
| **Eye** | Smoothed eye state (OPEN / CLOSED) and CNN confidence |
| **EAR** | Current Eye Aspect Ratio and consecutive low-EAR frame count |
| **Decision** | ALERT (green) or DROWSY (red) |
| **ALARM !** | Shown while the alarm sound is playing |
| **DROWSY banner** | Large centred banner during drowsy episodes |
| Face box | Green when alert, red when drowsy |

---

## Alert Logic

Drowsiness is triggered by **either** of two independent signals:

1. **CNN signal** — the fraction of CLOSED predictions in the last `SMOOTHING_WINDOW` (10) frames exceeds `DROWSY_CNN_RATIO` (0.6).
2. **EAR signal** — the Eye Aspect Ratio stays below `EAR_THRESHOLD` (0.22) for `EAR_CONSEC_FRAMES` (20) consecutive frames.

Both thresholds are configurable in `utils/config.py`.

The alarm will not retrigger until `ALARM_COOLDOWN_SECONDS` (5 s) have elapsed.

---

## Detection Log

Every processed frame is appended to `logs/detection_logs.csv`:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO-8601 datetime with milliseconds |
| `frame_number` | Sequential frame index |
| `cnn_probability` | Raw CNN P(CLOSED) |
| `ear_value` | Eye Aspect Ratio |
| `smoothed_state` | Smoothed label (OPEN / CLOSED) |
| `final_decision` | ALERT or DROWSY |
| `alarm_triggered` | 1 if alarm fired this frame, else 0 |

---

## Configuration

All tunable parameters are in `utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | 0 | Webcam device index |
| `IMG_SIZE` | (224, 224) | CNN input resolution |
| `CONFIDENCE_THRESHOLD` | 0.5 | CNN classification threshold |
| `EAR_THRESHOLD` | 0.22 | Minimum open-eye EAR |
| `EAR_CONSEC_FRAMES` | 20 | Frames below EAR threshold to trigger |
| `SMOOTHING_WINDOW` | 10 | Rolling window size for majority vote |
| `DROWSY_CNN_RATIO` | 0.6 | Fraction of CLOSED frames to trigger |
| `ALARM_COOLDOWN_SECONDS` | 5.0 | Minimum gap between alarms |

---

## Troubleshooting

**`FileNotFoundError: Model file not found`**
→ Make sure `drowsiness_model.h5` is placed in the `models/` directory.

**`RuntimeError: Cannot open camera at index 0`**
→ Check that no other application is using the webcam.  Try `--camera 1` for an external camera.

**`pygame mixer unavailable`**
→ The alarm is disabled but the rest of the system works normally.  Install SDL2 system libraries if you need audio.

**Low FPS / lag**
→ Reduce `IMG_SIZE` to `(64, 64)` or `(96, 96)` in `config.py` to speed up inference.

**Too many false alarms**
→ Increase `EAR_CONSEC_FRAMES`, increase `SMOOTHING_WINDOW`, or raise `DROWSY_CNN_RATIO`.

**Eye not detected reliably**
→ Improve lighting.  MediaPipe Face Mesh works best with even frontal illumination.

---

## Future Improvements

- Multi-face support for fleet / shared-vehicle contexts
- Head-pose estimation to detect micro-sleep nodding
- Yawn detection using mouth landmarks
- ONNX / TFLite export for edge deployment
- REST API endpoint for integration with vehicle telematics
