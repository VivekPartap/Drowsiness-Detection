"""
realtime/webcam.py
Thin wrapper around cv2.VideoCapture with descriptive error messages.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.config import CAMERA_INDEX


class Webcam:
    """
    Manages a webcam capture session.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index (0 = system default).
    width : int, optional
        Requested frame width.  The actual width may differ if the camera
        does not support the requested resolution.
    height : int, optional
        Requested frame height.
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self._index = camera_index
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that the webcam is connected and not in use by another application."
            )

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ── Public ────────────────────────────────────────────────────────────────

    def read(self) -> Tuple[bool, Optional[NDArray]]:
        """
        Read the next frame from the webcam.

        Returns
        -------
        (success, frame)
            success – False if the camera stopped providing frames.
            frame   – BGR NDArray of shape (H, W, 3), or None on failure.
        """
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    @property
    def frame_width(self) -> int:
        """Actual capture width in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Actual capture height in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
