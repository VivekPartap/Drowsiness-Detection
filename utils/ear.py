import numpy as np

def euclidean_distance(ptA, ptB):
    """Calculates the Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

def eye_aspect_ratio(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR) for a given eye.

    Args:
        eye_landmarks (list): List of (x, y) coordinates for eye landmarks.
                              Expected order: [p1, p2, p3, p4, p5, p6]
                              Where p2, p3 are vertical and p5, p6 are horizontal.

    Returns:
        float: The Eye Aspect Ratio.
    """
    # Vertical eye landmark coordinates
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])

    # Horizontal eye landmark coordinates
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

if __name__ == '__main__':
    # Example usage
    # Mock eye landmarks (e.g., from MediaPipe)
    left_eye_mock = [(330, 200), (335, 190), (345, 192), (380, 205), (345, 218), (335, 215)]
    ear_val = eye_aspect_ratio(left_eye_mock)
    print(f"Sample EAR calculation: {ear_val:.2f}")
