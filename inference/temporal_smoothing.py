"""
inference/temporal_smoothing.py
Confidence- and reliability-aware rolling smoother.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from utils.config import (
    MIN_PREDICTION_CONFIDENCE,
    MIN_RELIABILITY_FOR_CNN,
    SMOOTHING_MIN_SAMPLES,
    SMOOTHING_WINDOW,
)


@dataclass(frozen=True)
class _Sample:
    class_index: int
    confidence: float
    reliability: float

    @property
    def weight(self) -> float:
        return max(0.05, min(1.0, self.confidence * self.reliability))


class TemporalSmoother:
    """Maintain a weighted rolling window of reliable predictions."""

    def __init__(
        self,
        window: int = SMOOTHING_WINDOW,
        min_confidence: float = MIN_PREDICTION_CONFIDENCE,
        min_reliability: float = MIN_RELIABILITY_FOR_CNN,
        min_samples: int = SMOOTHING_MIN_SAMPLES,
        max_skipped_frames: int = 8,
    ) -> None:
        self._buffer: deque[_Sample] = deque(maxlen=window)
        self._min_confidence = min_confidence
        self._min_reliability = min_reliability
        self._min_samples = min_samples
        self._max_skipped_frames = max_skipped_frames
        self._skipped_frames = 0

    def update(self, class_index: int, confidence: float, reliability: float) -> bool:
        """
        Add a prediction when both CNN confidence and landmark reliability pass.

        Returns True if the prediction was accepted into the smoothing window.
        """
        if confidence < self._min_confidence or reliability < self._min_reliability:
            self.mark_unreliable()
            return False

        self._buffer.append(
            _Sample(
                class_index=1 if class_index == 1 else 0,
                confidence=float(confidence),
                reliability=float(reliability),
            )
        )
        self._skipped_frames = 0
        return True

    def mark_unreliable(self) -> None:
        """Record an unreliable frame without adding a false OPEN/CLOSED vote."""
        self._skipped_frames += 1
        if self._skipped_frames >= self._max_skipped_frames:
            self.reset()

    @property
    def smoothed_class(self) -> int:
        """Return the weighted majority class, defaulting to OPEN if sparse."""
        if not self.has_enough_samples:
            return 0
        return int(self.closed_ratio >= 0.5)

    @property
    def closed_ratio(self) -> float:
        """Weighted fraction of accepted predictions that are CLOSED."""
        if not self.has_enough_samples:
            return 0.0
        total_weight = sum(sample.weight for sample in self._buffer)
        if total_weight <= 0:
            return 0.0
        closed_weight = sum(
            sample.weight for sample in self._buffer if sample.class_index == 1
        )
        return closed_weight / total_weight

    @property
    def has_enough_samples(self) -> bool:
        return len(self._buffer) >= self._min_samples

    @property
    def accepted_count(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._skipped_frames = 0

    def __len__(self) -> int:
        return len(self._buffer)
