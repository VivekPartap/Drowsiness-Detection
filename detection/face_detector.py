"""
detection/face_detector.py
MediaPipe Face Mesh wrapper with lightweight landmark reliability scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from detection.landmarks import LEFT_EYE, RIGHT_EYE
from utils.config import (
    HEAD_PITCH_HARD_DEGREES,
    HEAD_PITCH_SOFT_DEGREES,
    HEAD_ROLL_HARD_DEGREES,
    HEAD_ROLL_SOFT_DEGREES,
    HEAD_YAW_HARD_DEGREES,
    HEAD_YAW_SOFT_DEGREES,
    MIN_EYE_VISIBILITY,
)


@dataclass(frozen=True)
class FacePose:
    """Approximate orientation plus reliability for the current face frame."""

    yaw: float
    pitch: float
    roll: float
    left_eye_visibility: float
    right_eye_visibility: float
    visible_eye_count: int
    pose_score: float
    eye_score: float
    frame_score: float
    reliability: float
    quality_label: str
    reason: str

    @property
    def both_eyes_visible(self) -> bool:
        return self.visible_eye_count == 2

    @property
    def has_visible_eye(self) -> bool:
        return self.visible_eye_count > 0


@dataclass
class FaceResult:
    """Container for a single detected face."""

    landmarks_2d: NDArray
    landmarks_norm: NDArray
    face_bbox: tuple[int, int, int, int]
    pose: FacePose


def _distance(a: NDArray, b: NDArray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _bbox_from_points(points: NDArray) -> tuple[int, int, int, int]:
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _axis_score(value: float, soft_limit: float, hard_limit: float) -> float:
    value = abs(value)
    if value <= soft_limit:
        return 1.0
    if value >= hard_limit:
        return 0.0
    return 1.0 - ((value - soft_limit) / (hard_limit - soft_limit))


def _edge_score(
    points: NDArray,
    frame_w: int,
    frame_h: int,
    margin: int = 3,
) -> float:
    x1, y1, x2, y2 = _bbox_from_points(points)
    if x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin:
        return 0.45
    return 1.0


def _quality_label(reliability: float) -> str:
    if reliability >= 0.78:
        return "HIGH"
    if reliability >= 0.48:
        return "MODERATE"
    return "LOW"


def _estimate_pose(landmarks_2d: NDArray, frame_w: int, frame_h: int) -> FacePose:
    """
    Estimate face orientation and landmark reliability from stable geometry.

    Pose is used as context only. Mild head turns lower reliability and
    prediction weight; they do not automatically block drowsiness detection.
    """
    left_outer = landmarks_2d[263].astype(np.float32)
    right_outer = landmarks_2d[33].astype(np.float32)
    left_cheek = landmarks_2d[454].astype(np.float32)
    right_cheek = landmarks_2d[234].astype(np.float32)
    nose_tip = landmarks_2d[1].astype(np.float32)
    chin = landmarks_2d[152].astype(np.float32)

    face_width = max(_distance(left_cheek, right_cheek), 1.0)
    mid_eyes = (left_outer + right_outer) * 0.5
    face_center = (left_cheek + right_cheek) * 0.5

    roll = degrees(atan2(left_outer[1] - right_outer[1], left_outer[0] - right_outer[0]))

    left_eye_width = _distance(landmarks_2d[LEFT_EYE.p1], landmarks_2d[LEFT_EYE.p4])
    right_eye_width = _distance(landmarks_2d[RIGHT_EYE.p1], landmarks_2d[RIGHT_EYE.p4])
    max_eye_width = max(left_eye_width, right_eye_width, 1.0)
    eye_width_sum = max(left_eye_width + right_eye_width, 1.0)

    nose_yaw = ((nose_tip[0] - face_center[0]) / face_width) * 85.0
    eye_yaw = ((right_eye_width - left_eye_width) / eye_width_sum) * 55.0
    yaw = (0.65 * nose_yaw) + (0.35 * eye_yaw)

    eye_to_chin = max(float(chin[1] - mid_eyes[1]), 1.0)
    nose_drop = float(nose_tip[1] - mid_eyes[1])
    pitch = ((nose_drop / eye_to_chin) - 0.42) * 85.0

    pose_score = min(
        _axis_score(yaw, HEAD_YAW_SOFT_DEGREES, HEAD_YAW_HARD_DEGREES),
        _axis_score(pitch, HEAD_PITCH_SOFT_DEGREES, HEAD_PITCH_HARD_DEGREES),
        _axis_score(roll, HEAD_ROLL_SOFT_DEGREES, HEAD_ROLL_HARD_DEGREES),
    )

    left_eye_points = landmarks_2d[[362, 263, 386, 374]]
    right_eye_points = landmarks_2d[[33, 133, 159, 145]]
    left_eye_visibility = min(
        left_eye_width / max_eye_width,
        _edge_score(left_eye_points, frame_w, frame_h),
    )
    right_eye_visibility = min(
        right_eye_width / max_eye_width,
        _edge_score(right_eye_points, frame_w, frame_h),
    )
    visible_eye_count = int(left_eye_visibility >= MIN_EYE_VISIBILITY) + int(
        right_eye_visibility >= MIN_EYE_VISIBILITY
    )

    if visible_eye_count == 2:
        eye_score = (left_eye_visibility + right_eye_visibility) * 0.5
    elif visible_eye_count == 1:
        eye_score = max(left_eye_visibility, right_eye_visibility) * 0.68
    else:
        eye_score = 0.0

    face_box = landmarks_2d[[10, 152, 234, 454, 33, 263, 1]]
    frame_score = _edge_score(face_box, frame_w, frame_h)
    if face_width < frame_w * 0.12:
        frame_score *= 0.65

    reliability = (0.45 * eye_score) + (0.35 * pose_score) + (0.20 * frame_score)
    if pose_score <= 0.05:
        reliability = min(reliability, 0.34)
    if visible_eye_count == 0:
        reliability = min(reliability, 0.24)

    reasons: list[str] = []
    if pose_score < 0.55:
        reasons.append("pose")
    if visible_eye_count < 2:
        reasons.append("partial eyes")
    if frame_score < 0.75:
        reasons.append("edge")

    return FacePose(
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        left_eye_visibility=float(left_eye_visibility),
        right_eye_visibility=float(right_eye_visibility),
        visible_eye_count=visible_eye_count,
        pose_score=float(pose_score),
        eye_score=float(eye_score),
        frame_score=float(frame_score),
        reliability=float(max(0.0, min(1.0, reliability))),
        quality_label=_quality_label(reliability),
        reason=", ".join(reasons) if reasons else "stable",
    )


class FaceDetector:
    """Detect a single face and return landmarks with reliability metadata."""

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: NDArray) -> Optional[FaceResult]:
        """Run face mesh detection on a single BGR frame."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        pixels_2d = np.column_stack(
            [
                (norm[:, 0] * w).astype(np.int32),
                (norm[:, 1] * h).astype(np.int32),
            ]
        )

        return FaceResult(
            landmarks_2d=pixels_2d,
            landmarks_norm=norm,
            face_bbox=_bbox_from_points(pixels_2d),
            pose=_estimate_pose(pixels_2d, w, h),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
