import cv2
import numpy as np

class EyeDetector:
    """Extracts eye landmarks from MediaPipe FaceMesh results."""

    def __init__(self):
        pass

    def get_eye_landmarks(self, face_mesh_results, img_width, img_height):
        """Extracts specific facial landmarks for EAR calculation from MediaPipe FaceMesh results.

        Args:
            face_mesh_results: The output from MediaPipe FaceLandmarker detector.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            tuple: A tuple containing (left_eye_landmarks, right_eye_landmarks) or (None, None).
        """
        # Check if any face landmarks were detected
        if not face_mesh_results or not face_mesh_results.face_landmarks:
            return None, None

        # Access the first detected face's landmarks
        face_landmarks = face_mesh_results.face_landmarks[0]

        # Indices for the left and right eyes based on MediaPipe FaceMesh documentation
        # Common indices for EAR calculation (p1, p2, p3, p4, p5, p6):
        # Left Eye: horizontal p1=33, p4=133; vertical p2=160, p3=158, p5=144, p6=153
        # Right Eye: horizontal p1=362, p4=263; vertical p2=387, p3=385, p5=373, p6=380

        LEFT_EYE_INDICES = [33, 160, 158, 133, 144, 153] # p1,p2,p3,p4,p5,p6 (approx)
        RIGHT_EYE_INDICES = [362, 387, 385, 263, 373, 380]

        left_eye_landmarks = []
        right_eye_landmarks = []

        for i in LEFT_EYE_INDICES:
            lm = face_landmarks[i]
            left_eye_landmarks.append((int(lm.x * img_width), int(lm.y * img_height)))

        for i in RIGHT_EYE_INDICES:
            lm = face_landmarks[i]
            right_eye_landmarks.append((int(lm.x * img_width), int(lm.y * img_height)))

        return left_eye_landmarks, right_eye_landmarks

if __name__ == '__main__':
    print("EyeDetector module: Extracts eye landmarks from face mesh results.")
