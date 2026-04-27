import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import logging

# Assuming config.py is in the base directory
try:
    from config import MEDIAPIPE_MODEL_PATH, MAX_NUM_FACES, MIN_DETECTION_CONFIDENCE, LOG_FILE
except ImportError:
    print("Warning: config.py not found. Using default MediaPipe parameters.")
    MEDIAPIPE_MODEL_PATH = 'ml/face_landmarker.task'
    MAX_NUM_FACES = 1
    MIN_DETECTION_CONFIDENCE = 0.7
    LOG_FILE = 'events.log' # Fallback for logging

# Setup basic logging if not already configured (useful for standalone script execution)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )

class FaceMeshDetector:
    """Encapsulates MediaPipe FaceLandmarker for face and landmark detection using the tasks API."""

    def __init__(self):
        logging.info("Initializing MediaPipe FaceLandmarker...")
        try:
            # BaseOptions for configuring the model path
            base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)

            # FaceLandmarkerOptions for configuring the detection process
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for static frames
                num_faces=MAX_NUM_FACES,
                min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
                # min_tracking_confidence and min_face_presence_confidence are for VIDEO mode
            )
            # Create the FaceLandmarker object
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            logging.info("MediaPipe FaceLandmarker initialized.")
        except Exception as e:
            logging.error(f"Error initializing MediaPipe FaceLandmarker: {e}")
            self.landmarker = None


    def find_face_landmarks(self, image):
        """Processes an image to find face landmarks using the FaceLandmarker API.

        Args:
            image (np.array): The input image (BGR format from OpenCV).

        Returns:
            tuple: A tuple containing:
                - result (mediapipe.tasks.python.vision.FaceLandmarkerResult): The MediaPipe face landmarker results object.
                - image (np.array): The original image with converted color space (RGB). Returns None for result if landmarker not initialized.
        """
        if not self.landmarker:
            return None, None

        # Convert the BGR image to RGB and then to MediaPipe Image object.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image and find face landmarks.
        landmarker_result = self.landmarker.detect(mp_image)

        return landmarker_result, image_rgb

# Example usage (for testing, not part of the main application flow)
if __name__ == "__main__":
    import numpy as np # Ensure numpy is imported for the example
    logging.info("Running FaceMeshDetector standalone test...")
    detector = FaceMeshDetector()

    if detector.landmarker:
        # Create a dummy image for testing
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Dummy Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        results, _ = detector.find_face_landmarks(dummy_image)

        if results and results.face_landmarks:
            logging.info(f"Detected {len(results.face_landmarks)} face(s).")
            for face_landmarks in results.face_landmarks:
                logging.info(f"First landmark: x={face_landmarks[0].x}, y={face_landmarks[0].y}, z={face_landmarks[0].z}")
        else:
            logging.info("No faces detected in the dummy image or landmarker not initialized.")
    else:
        logging.error("FaceLandmarker could not be initialized for testing.")

    logging.info("FaceMeshDetector standalone test complete.")
