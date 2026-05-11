"""
utils/preprocessing.py
Image preprocessing helpers that replicate the exact pipeline used during training.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.config import IMG_SIZE


def resize_frame(frame: NDArray, size: tuple[int, int] = IMG_SIZE) -> NDArray:
    """Resize a BGR frame to *size* (width, height)."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def normalize(image: NDArray) -> NDArray:
    """Scale pixel values from [0, 255] to [0.0, 1.0]."""
    return image.astype(np.float32) / 255.0


def to_rgb(frame: NDArray) -> NDArray:
    """Convert a BGR OpenCV frame to RGB (as expected by Keras)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preprocess_eye(eye_crop: NDArray, size: tuple[int, int] = IMG_SIZE) -> NDArray:
    """
    Full preprocessing pipeline for a single eye crop.

    Steps
    -----
    1. Convert BGR → RGB
    2. Resize to model input size
    3. Normalise to [0, 1]
    4. Add batch dimension → shape (1, H, W, 3)

    Parameters
    ----------
    eye_crop : NDArray
        Raw BGR image crop extracted from the webcam frame.
    size : tuple[int, int]
        Target (width, height) matching the model's expected input.

    Returns
    -------
    NDArray
        Float32 array of shape (1, H, W, 3) ready for model.predict().
    """
    if eye_crop is None or eye_crop.size == 0:
        raise ValueError("Eye crop is empty or None.")

    rgb = to_rgb(eye_crop)
    resized = resize_frame(rgb, size)
    normalised = normalize(resized)
    batched = np.expand_dims(normalised, axis=0)   # (1, H, W, 3)
    return batched


def preprocess_frame(frame: NDArray, size: tuple[int, int] = IMG_SIZE) -> NDArray:
    """
    Preprocess a full frame (used when the model operates on full crops
    rather than isolated eye patches).
    Same pipeline as preprocess_eye.
    """
    return preprocess_eye(frame, size)
