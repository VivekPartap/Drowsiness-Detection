import joblib
import os
import numpy as np
import logging

# Assuming config.py is in the base directory
try:
    from config import MODEL_PATH, ML_DROWSINESS_THRESHOLD, MODELS_DIR, LOG_FILE
except ImportError:
    print("Warning: config.py not found. Using default ML parameters.")
    MODEL_PATH = 'ml/drowsiness_model.joblib'
    ML_DROWSINESS_THRESHOLD = 0.5
    MODELS_DIR = 'ml'
    LOG_FILE = 'events.log' # Fallback for logging

# Setup basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )

class DrowsinessModel:
    """Manages loading and inference for the trained drowsiness detection ML model."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """Loads the pre-trained ML model and scaler from disk."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
            logging.info(f"ML model and scaler loaded successfully from {MODEL_PATH}")
        except FileNotFoundError:
            logging.error(f"ML model or scaler not found at {MODEL_PATH} or {os.path.join(MODELS_DIR, 'scaler.joblib')}. Please run `ml/train_model.py` first.")
        except Exception as e:
            logging.error(f"Error loading ML model or scaler: {e}")

    def predict(self, ear_value, consecutive_frames):
        """Predicts drowsiness based on EAR value and consecutive frames.

        Args:
            ear_value (float): The current Eye Aspect Ratio.
            consecutive_frames (int): Number of consecutive frames below EAR threshold.

        Returns:
            float: The probability of drowsiness (0 to 1).
        """
        if self.model is None or self.scaler is None:
            logging.warning("ML model or scaler not loaded. Returning 0 (Awake).")
            return 0.0

        # Prepare input for scaling and prediction
        features = np.array([[ear_value, consecutive_frames]])
        scaled_features = self.scaler.transform(features)

        # Predict probability of drowsiness (class 1)
        drowsiness_probability = self.model.predict_proba(scaled_features)[:, 1][0]

        return drowsiness_probability

    def is_drowsy(self, ear_value, consecutive_frames):
        """Determines if the subject is drowsy based on ML model prediction.

        Returns:
            bool: True if drowsy, False otherwise.
        """
        drowsiness_prob = self.predict(ear_value, consecutive_frames)
        return drowsiness_prob >= ML_DROWSINESS_THRESHOLD

if __name__ == '__main__':
    # Example usage (requires model and scaler to be trained and saved by train_model.py)
    logging.info("Testing DrowsinessModel. Make sure `ml/train_model.py` has been run.")

    # Ensure the ml directory exists for saving dummy model if needed for standalone test
    os.makedirs(MODELS_DIR, exist_ok=True)

    # For testing without actually training, create dummy model/scaler files.
    # In a real scenario, you'd run train_model.py first.
    dummy_model_path = MODEL_PATH
    dummy_scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')

    if not os.path.exists(dummy_model_path) or not os.path.exists(dummy_scaler_path):
        logging.warning("Dummy model/scaler not found. Creating placeholder files for test, but actual predictions will be meaningless. Run `ml/train_model.py` for real functionality.")
        # Create dummy files to allow DrowsinessModel to initialize without error
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        dummy_model = LogisticRegression()
        joblib.dump(dummy_model, dummy_model_path)
        dummy_scaler = StandardScaler()
        joblib.dump(dummy_scaler, dummy_scaler_path)

    drowsiness_detector = DrowsinessModel()

    if drowsiness_detector.model and drowsiness_detector.scaler:
        # Test with example EAR and consecutive frames
        test_ear_awake = 0.35 # High EAR (awake)
        test_consec_awake = 5 # Low consecutive frames
        prob_awake = drowsiness_detector.predict(test_ear_awake, test_consec_awake)
        status_awake = drowsiness_detector.is_drowsy(test_ear_awake, test_consec_awake)
        logging.info(f"Awake scenario (EAR={test_ear_awake:.2f}, CF={test_consec_awake}): Prob={prob_awake:.2f}, Drowsy={status_awake}")

        test_ear_drowsy = 0.20 # Low EAR (drowsy)
        test_consec_drowsy = 40 # High consecutive frames
        prob_drowsy = drowsiness_detector.predict(test_ear_drowsy, test_consec_drowsy)
        status_drowsy = drowsiness_detector.is_drowsy(test_ear_drowsy, test_consec_drowsy)
        logging.info(f"Drowsy scenario (EAR={test_ear_drowsy:.2f}, CF={test_consec_drowsy}): Prob={prob_drowsy:.2f}, Drowsy={status_drowsy}")
    else:
        logging.error("DrowsinessModel could not be initialized for testing.")
