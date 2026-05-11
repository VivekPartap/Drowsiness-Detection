"""
detection/eye_extractor.py
Extract and lightly align left/right eye crops from Face Mesh landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from detection.landmarks import (
    LEFT_EYE,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE,
    RIGHT_EYE_CONTOUR,
    EyeLandmarkIndices,
)


_PADDING: float = 0.35
_MIN_EYE_WIDTH = 10
_MIN_EYE_HEIGHT = 6


@dataclass
class EyeCrops:
    """Holds aligned image crops and crop boxes for both eyes."""

    left: Optional[NDArray]
    right: Optional[NDArray]
    left_bbox: Optional[tuple[int, int, int, int]] = None
    right_bbox: Optional[tuple[int, int, int, int]] = None

    @property
    def available_count(self) -> int:
        return int(self.left is not None) + int(self.right is not None)

    @property
    def both_available(self) -> bool:
        return self.available_count == 2

    @property
    def any_available(self) -> bool:
        return self.available_count > 0


def _contour_bbox(
    landmarks_2d: NDArray,
    contour_indices: list[int],
    frame_h: int,
    frame_w: int,
    padding: float = _PADDING,
) -> Optional[tuple[int, int, int, int]]:
    pts = landmarks_2d[contour_indices]
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    width = int(x_max - x_min)
    height = int(y_max - y_min)
    if width < _MIN_EYE_WIDTH or height < _MIN_EYE_HEIGHT:
        return None

    pad_x = max(2, int(width * padding))
    pad_y = max(2, int(height * padding))

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))

    if (x2 - x1) < _MIN_EYE_WIDTH or (y2 - y1) < _MIN_EYE_HEIGHT:
        return None
    return x1, y1, x2, y2


def _aligned_crop(
    frame_bgr: NDArray,
    landmarks_2d: NDArray,
    bbox: Optional[tuple[int, int, int, int]],
    eye_indices: EyeLandmarkIndices,
) -> Optional[NDArray]:
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    p1 = landmarks_2d[eye_indices.p1].astype(np.float32)
    p4 = landmarks_2d[eye_indices.p4].astype(np.float32)
    angle = degrees(atan2(p4[1] - p1[1], p4[0] - p1[0]))

    h, w = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    return cv2.warpAffine(
        crop,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def extract_eyes(frame_bgr: NDArray, landmarks_2d: NDArray) -> EyeCrops:
    """
    Extract padded, roll-aligned eye crops.

    If only one eye remains usable during a moderate head turn, the caller can
    still run single-eye inference with reduced reliability.
    """
    h, w = frame_bgr.shape[:2]
    left_bbox = _contour_bbox(landmarks_2d, LEFT_EYE_CONTOUR, h, w)
    right_bbox = _contour_bbox(landmarks_2d, RIGHT_EYE_CONTOUR, h, w)

    return EyeCrops(
        left=_aligned_crop(frame_bgr, landmarks_2d, left_bbox, LEFT_EYE),
        right=_aligned_crop(frame_bgr, landmarks_2d, right_bbox, RIGHT_EYE),
        left_bbox=left_bbox,
        right_bbox=right_bbox,
    )


def pick_best_eye(crops: EyeCrops) -> Optional[NDArray]:
    """Return whichever eye crop is available, preferring the left eye."""
    if crops.left is not None:
        return crops.left
    return crops.right
