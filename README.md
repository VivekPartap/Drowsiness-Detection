%%writefile README.md
# AI-Powered Drowsiness Detection System

This project implements an AI-powered system for real-time drowsiness detection using facial landmarks and a Machine Learning model. It features a Graphical User Interface (GUI) to display the video feed, Eye Aspect Ratio (EAR) graph, and detection status, along with an audio alarm system to alert the user of drowsiness.

## Features
-   **Real-time Drowsiness Detection:** Monitors eye closure using MediaPipe FaceLandmarker and a calculated Eye Aspect Ratio (EAR).
-   **Machine Learning Integration:** Employs a trained ML model (Logistic Regression or Random Forest) to enhance drowsiness detection accuracy.
-   **Audio Alarm System:** Triggers a sound alarm when prolonged drowsiness is detected.
-   **Interactive GUI:** Provides a user-friendly interface with:
    -   Live video feed.
    -   Real-time display of Drowsiness Status, EAR, and FPS.
    -   Live graph of Eye Aspect Ratio over time.
    -   Buttons to start/stop detection and enable/disable the alarm.
-   **Configurable Parameters:** Easily adjust thresholds, model paths, and GUI settings via `config.py`.
-   **Flexible Input:** Supports webcam or video file input.

## Project Structure

```
.
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings for the system
├── gui.py                     # Tkinter-based Graphical User Interface
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
├── alert/
│   ├── __init__.py
│   └── alarm.py               # Handles audio alarm playback
├── assets/
│   └── alarm.wav              # Audio file for the alarm (dummy provided)
├── detector/
│   ├── __init__.py
│   ├── face_mesh.py           # MediaPipe FaceLandmarker integration
│   └── eye_detector.py        # Extracts eye landmarks and calculates EAR
├── ml/
│   ├── __init__.py
│   ├── train_model.py         # Script to train and save the ML model
│   ├── model.py               # Loads and provides inference from the ML model
│   ├── drowsiness_model.joblib # (Generated) Trained ML model
│   ├── scaler.joblib          # (Generated) Scaler for ML model features
│   ├── face_landmarker.task   # (Downloaded) MediaPipe FaceLandmarker model
│   └── drowsiness_dataset.csv # (Generated) Synthetic dataset for ML training
├── utils/
│   ├── __init__.py
│   ├── ear.py                 # EAR calculation utility
│   └── helpers.py             # General utility functions (logging, drawing)
└── drowsiness_detection_model.keras # (Generated) Trained Keras model for image classification (if using advanced model)
```

## Setup and Installation

### 1. Clone the Repository (or Download Files)

If you're running this from a Colab notebook, you've likely received the files directly. If running locally, clone the repository:

```bash
git clone <repository_url>
cd <project_directory>
```

### 2. Create a Virtual Environment (Recommended)

It is highly recommended to create and activate a Python virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note on MediaPipe and OpenCV:**
Due to potential compatibility issues with `mediapipe`, `opencv-python`, and `protobuf`, specific versions are often necessary. The `requirements.txt` specifies `mediapipe==0.10.33` and `opencv-python`. If you encounter issues, especially `AttributeError: module 'mediapipe' has no attribute 'solutions'`, you might need to try a specific reinstallation sequence:

```bash
# Aggressive cleanup and reinstallation steps
pip uninstall opencv-python -y
pip uninstall mediapipe -y
pip uninstall protobuf -y
pip cache purge
pip install protobuf==3.20.3
pip install opencv-python
pip install mediapipe==0.10.33
pip install -r requirements.txt # To get other dependencies
```

### 4. Download MediaPipe Face Landmarker Model

The project requires the MediaPipe Face Landmarker model. It will be downloaded automatically if not found. Run the following command (or simply run `main.py` which will attempt the download):

```bash
python -c "import requests, os; from config import MEDIAPIPE_MODEL_PATH, MODELS_DIR; os.makedirs(MODELS_DIR, exist_ok=True); MEDIAPIPE_MODEL_URL = \"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task\"; if not os.path.exists(MEDIAPIPE_MODEL_PATH): print(f'Downloading MediaPipe Face Landmarker model to {MEDIAPIPE_MODEL_PATH}...'); response = requests.get(MEDIAPIPE_MODEL_URL, stream=True); response.raise_for_status(); with open(MEDIAPIPE_MODEL_PATH, 'wb') as f: [f.write(chunk) for chunk in response.iter_content(chunk_size=8192)]; print('Download complete!'); else: print(f'MediaPipe Face Landmarker model already exists at {MEDIAPIPE_MODEL_PATH}.')"
```

### 5. Prepare Alarm Sound File

A dummy `alarm.wav` file is included in the `assets/` directory. For a functional alarm, replace `assets/alarm.wav` with an actual `.wav` audio file of your choice. The current dummy file prevents errors but will not produce sound.

### 6. Train the ML Model (Optional, but Recommended)

The system includes an ML model for enhanced drowsiness detection. If `drowsiness_model.joblib` and `scaler.joblib` are not found in the `ml/` directory, `main.py` will automatically train a model using a synthetic dataset. You can also manually train it:

```bash
python ml/train_model.py
```

## Configuration (`config.py`)

The `config.py` file contains all adjustable parameters for the system. Key parameters you might want to modify include:

-   `WEBCAM_INDEX`: Set to `0` for the default webcam, or another index if you have multiple cameras.
-   `VIDEO_INPUT_PATH`: Set this to the path of a video file (e.g., `'assets/my_video.mp4'`) to process a video file instead of a live webcam feed. Set to `None` to use the webcam.
-   `EAR_THRESHOLD`: The Eye Aspect Ratio threshold below which an eye is considered closed.
-   `EAR_CONSEC_FRAMES`: The number of consecutive frames below `EAR_THRESHOLD` to trigger drowsiness.
-   `ML_DROWSINESS_THRESHOLD`: Probability threshold for the ML model to classify as drowsy.
-   `ALARM_PATH`: Path to your `alarm.wav` file.

## Running the Application

### In a Local Code Editor (e.g., VS Code, Cursor) with GUI

For local development environments, the GUI (Tkinter) should work out-of-the-box. Ensure your virtual environment is activated and dependencies are installed. Then, simply run the `main.py` script:

```bash
python main.py
```

-   **Using Webcam:** Ensure `VIDEO_INPUT_PATH` is set to `None` in `config.py`.
-   **Using a Video File:** Update `VIDEO_INPUT_PATH` in `config.py` to point to your video file (e.g., `VIDEO_INPUT_PATH = 'assets/your_video.mp4'`).

### In Google Colab (Without GUI)

Due to Colab's environment limitations, the GUI will not display directly. However, the backend processing logic can still run. `main.py` is designed to run headlessly in Colab by checking the `IN_COLAB` environment variable.

To run in Colab, ensure `VIDEO_INPUT_PATH` in `config.py` is set to a video file uploaded to your Colab environment or one downloaded into the `assets/` directory (webcam input is generally not supported for GUI applications in Colab).

```python
!python main.py
```

This will output logging information about the detection process to the Colab output.

## Troubleshooting

-   **`_tkinter.TclError: no display name and no $DISPLAY environment variable`**: This error typically occurs when running a Tkinter GUI application in an environment without a graphical display server. This is common in Colab or remote SSH sessions without X forwarding. If running locally, ensure your desktop environment is functional. If in Colab, the script should automatically detect the environment and run headlessly.
-   **`Failed to open video source`**: Check `WEBCAM_INDEX` in `config.py` (try `0`, `1`, etc.) or ensure `VIDEO_INPUT_PATH` points to a valid and accessible video file.
-   **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**: Refer to the MediaPipe and OpenCV reinstallation steps in Section 3.
-   **`ModuleNotFoundError: No module named '...'`**: Ensure all dependencies listed in `requirements.txt` are installed (`pip install -r requirements.txt`). For core modules, verify the project structure matches the imports (e.g., `ml/model.py` must exist if `from ml.model import ...` is used).
-   **`Alarm sound file not found`**: Ensure `assets/alarm.wav` exists and is a valid `.wav` file, or update `ALARM_PATH` in `config.py`.

Feel free to contribute, report issues, or suggest improvements!
