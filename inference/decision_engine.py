"""
inference/decision_engine.py
Three-state drowsiness decision logic: ALERT, UNCERTAIN, DROWSY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.config import (
    DROWSY_CNN_RATIO,
    DROWSY_FRAME_LIMIT,
    EAR_THRESHOLD,
    UNCERTAIN_RELIABILITY_THRESHOLD,
)


@dataclass
class DecisionResult:
    """Output of the decision engine for one frame."""

    is_drowsy: bool
    decision_label: str
    trigger_reason: str


class DecisionEngine:
    """Combine persistent CNN/EAR evidence with frame reliability context."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        ear_consec_limit: int = DROWSY_FRAME_LIMIT,
        drowsy_cnn_ratio: float = DROWSY_CNN_RATIO,
    ) -> None:
        self._ear_threshold = ear_threshold
        self._ear_consec_limit = ear_consec_limit
        self._drowsy_cnn_ratio = drowsy_cnn_ratio
        self._ear_consec_count = 0

    def evaluate(
        self,
        closed_ratio: float,
        ear: Optional[float],
        cnn_ready: bool,
        reliability: float,
        visible_eye_count: int,
        has_prediction: bool,
    ) -> DecisionResult:
        """
        Return ALERT, UNCERTAIN, or DROWSY.

        Low-reliability frames become UNCERTAIN. They do not become DROWSY, and
        they do not reset the whole pipeline unless no eye evidence is usable.
        """
        if visible_eye_count <= 0 or reliability < UNCERTAIN_RELIABILITY_THRESHOLD:
            self._ear_consec_count = 0
            return DecisionResult(False, "UNCERTAIN", "LOW RELIABILITY")

        if ear is None:
            self._ear_consec_count = 0
            ear_drowsy = False
        elif ear < self._ear_threshold:
            self._ear_consec_count += 1
            ear_drowsy = self._ear_consec_count >= self._ear_consec_limit
        else:
            self._ear_consec_count = 0
            ear_drowsy = False

        cnn_drowsy = cnn_ready and closed_ratio >= self._drowsy_cnn_ratio
        if cnn_drowsy or ear_drowsy:
            if cnn_drowsy and ear_drowsy:
                reason = "CNN + EAR"
            elif cnn_drowsy:
                reason = "CNN"
            else:
                reason = "EAR"
            return DecisionResult(True, "DROWSY", reason)

        if not has_prediction and ear is None:
            return DecisionResult(False, "UNCERTAIN", "NO EYE SIGNAL")
        if has_prediction and not cnn_ready:
            return DecisionResult(False, "UNCERTAIN", "SMOOTHING")
        if reliability < 0.55:
            return DecisionResult(False, "UNCERTAIN", "PARTIAL FACE")

        return DecisionResult(False, "ALERT", "-")

    @property
    def ear_consecutive_count(self) -> int:
        return self._ear_consec_count

    def reset(self) -> None:
        self._ear_consec_count = 0
