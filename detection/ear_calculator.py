"""
detection/ear_calculator.py
Eye Aspect Ratio (EAR) helpers from facial landmarks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from detection.landmarks import EyeLandmarkIndices


def _euclidean(a: NDArray, b: NDArray) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


def compute_ear(landmarks_2d: NDArray, eye_indices: EyeLandmarkIndices) -> float:
    """Compute Eye Aspect Ratio for one eye."""
    p1 = landmarks_2d[eye_indices.p1]
    p2 = landmarks_2d[eye_indices.p2]
    p3 = landmarks_2d[eye_indices.p3]
    p4 = landmarks_2d[eye_indices.p4]
    p5 = landmarks_2d[eye_indices.p5]
    p6 = landmarks_2d[eye_indices.p6]

    vertical_a = _euclidean(p2, p6)
    vertical_b = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def average_ear(
    landmarks_2d: NDArray,
    left_eye: EyeLandmarkIndices,
    right_eye: EyeLandmarkIndices,
) -> float:
    """Compute the average EAR across both eyes."""
    return (compute_ear(landmarks_2d, left_eye) + compute_ear(landmarks_2d, right_eye)) / 2.0


def average_available_ear(
    landmarks_2d: NDArray,
    left_eye: EyeLandmarkIndices,
    right_eye: EyeLandmarkIndices,
    use_left: bool = True,
    use_right: bool = True,
) -> Optional[float]:
    """
    Compute EAR from whichever eyes are reliable enough.

    During a moderate head turn, one eye may be partially occluded. Using the
    visible eye preserves true drowsiness detection without treating weak
    landmark geometry as a confident signal.
    """
    ears: list[float] = []
    if use_left:
        ears.append(compute_ear(landmarks_2d, left_eye))
    if use_right:
        ears.append(compute_ear(landmarks_2d, right_eye))

    if not ears:
        return None
    return float(np.mean(ears))
