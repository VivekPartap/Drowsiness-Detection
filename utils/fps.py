"""
utils/fps.py
Lightweight rolling-average FPS calculator.
"""

import time
from collections import deque


class FPSCounter:
    """
    Compute frames-per-second using a rolling average over a fixed window.

    Parameters
    ----------
    window : int
        Number of recent frame timestamps to keep in the buffer.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record the timestamp of the current frame."""
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        """Return the smoothed FPS over the rolling window."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def __str__(self) -> str:
        return f"{self.fps:.1f} FPS"
