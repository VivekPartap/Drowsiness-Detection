"""
realtime/overlay.py
Drawing helpers for real-time diagnostic overlays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
from numpy.typing import NDArray

from utils.config import (
    COLOR_BLACK,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    FONT_SCALE,
    FONT_THICKNESS,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE = cv2.LINE_AA
_LINE_HEIGHT = 28


def _text_bg(
    frame: NDArray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    """Draw text on a black background for legibility."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, FONT_SCALE, FONT_THICKNESS)
    x, y = origin
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), COLOR_BLACK, -1)
    cv2.putText(frame, text, origin, _FONT, FONT_SCALE, color, FONT_THICKNESS, _LINE)


def draw_face_bbox(
    frame: NDArray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = COLOR_GREEN,
    label: Optional[str] = None,
) -> None:
    """Draw a clamped rectangle around the detected face."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, _LINE)
    if label:
        _text_bg(frame, label, (x1, max(0, y1 - 8)), color)


def draw_status_panel(
    frame: NDArray,
    eye_label: str,
    confidence: float,
    ear: Optional[float],
    fps: float,
    decision: str,
    alarm_active: bool,
    ear_consec: int = 0,
    reliability: float = 0.0,
    quality_label: str = "LOW",
    reason: str = "-",
) -> None:
    """Render the diagnostic panel in the top-left corner."""
    decision_color = {
        "DROWSY": COLOR_RED,
        "UNCERTAIN": COLOR_YELLOW,
        "ALERT": COLOR_GREEN,
    }.get(decision, COLOR_WHITE)
    eye_color = COLOR_YELLOW if eye_label in {"CLOSED", "LOW CONF"} else COLOR_GREEN

    lines: list[Tuple[str, Tuple[int, int, int]]] = [
        (f"FPS : {fps:.1f}", COLOR_WHITE),
        (f"Eye : {eye_label}  ({confidence * 100:.1f}%)", eye_color),
        (
            f"EAR : {'N/A' if ear is None else f'{ear:.3f}'}  [streak: {ear_consec}]",
            COLOR_WHITE,
        ),
        (f"Quality : {quality_label}  ({reliability:.2f})", COLOR_WHITE),
        (f"Decision : {decision}", decision_color),
    ]

    if decision == "UNCERTAIN" and reason != "-":
        lines.append((f"Reason : {reason}", COLOR_YELLOW))
    if alarm_active:
        lines.append(("ALARM !", COLOR_RED))

    y_start = 30
    for i, (text, color) in enumerate(lines):
        _text_bg(frame, text, (10, y_start + i * _LINE_HEIGHT), color)


def _draw_banner(frame: NDArray, message: str, color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(message, _FONT, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - th - 20), (w, y + 20), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, message, (x, y), _FONT, scale, COLOR_WHITE, thickness, _LINE)


def draw_uncertain_banner(frame: NDArray, reason: str = "") -> None:
    """Draw an UNCERTAIN state banner when signal quality is insufficient."""
    suffix = f" - {reason}" if reason and reason != "-" else ""
    _draw_banner(frame, f"UNCERTAIN{suffix}", (0, 150, 170))


def draw_drowsy_banner(frame: NDArray) -> None:
    """Draw a DROWSY DETECTED banner while final state is DROWSY."""
    _draw_banner(frame, "DROWSY DETECTED", (0, 0, 180))
