"""
detection/landmarks.py
MediaPipe Face Mesh landmark index constants for both eye regions.

MediaPipe Face Mesh provides 468 landmarks.  The indices below correspond
to the six-point EAR model popularised by Soukupová & Čech (2016) mapped
onto the Face Mesh topology.

References
----------
https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

from dataclasses import dataclass
from typing import NamedTuple


# ── Six-point eye contour indices ─────────────────────────────────────────────
# P1 (left corner) → P2 (upper-left) → P3 (upper-right)
# P4 (right corner) → P5 (lower-right) → P6 (lower-left)

class EyeLandmarkIndices(NamedTuple):
    p1: int  # horizontal left
    p2: int  # vertical upper-left
    p3: int  # vertical upper-right
    p4: int  # horizontal right
    p5: int  # vertical lower-right
    p6: int  # vertical lower-left


# Face Mesh indices for left eye (from subject's perspective)
LEFT_EYE = EyeLandmarkIndices(
    p1=362, p2=385, p3=387,
    p4=263, p5=373, p6=380,
)

# Face Mesh indices for right eye (from subject's perspective)
RIGHT_EYE = EyeLandmarkIndices(
    p1=33,  p2=160, p3=158,
    p4=133, p5=153, p6=144,
)

# Wider bounding-box contour points used for eye crop extraction
LEFT_EYE_CONTOUR: list[int] = [
    362, 382, 381, 380, 374, 373,
    390, 249, 263, 466, 388, 387,
    386, 385, 384, 398,
]

RIGHT_EYE_CONTOUR: list[int] = [
    33,  7,  163, 144, 145, 153,
    154, 155, 133, 173, 157, 158,
    159, 160, 161, 246,
]
