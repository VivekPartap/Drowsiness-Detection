"""
realtime/realtime_detection.py
End-to-end real-time drowsiness detection pipeline.
"""

from __future__ import annotations

import logging

import cv2

from alarm.alarm_manager import AlarmManager
from detection.ear_calculator import average_available_ear
from detection.eye_extractor import EyeCrops, extract_eyes
from detection.face_detector import FaceDetector, FacePose
from detection.landmarks import LEFT_EYE, RIGHT_EYE
from inference.decision_engine import DecisionEngine
from inference.predictor import EyeStatePredictor
from inference.temporal_smoothing import TemporalSmoother
from realtime.overlay import (
    draw_drowsy_banner,
    draw_face_bbox,
    draw_status_panel,
    draw_uncertain_banner,
)
from realtime.webcam import Webcam
from utils.config import MIN_EYE_VISIBILITY, MIN_RELIABILITY_FOR_CNN, WINDOW_TITLE
from utils.fps import FPSCounter
from utils.logger import DetectionLogger
from utils.preprocessing import preprocess_eye

logger = logging.getLogger(__name__)


class RealtimeDetection:
    """Orchestrates the complete real-time drowsiness detection pipeline."""

    def __init__(self, camera_index: int | None = None) -> None:
        from utils.config import CAMERA_INDEX

        cam_idx = camera_index if camera_index is not None else CAMERA_INDEX

        logger.info("Initialising subsystems...")
        self._webcam = Webcam(camera_index=cam_idx)
        self._face_detector = FaceDetector()
        self._predictor = EyeStatePredictor()
        self._smoother = TemporalSmoother()
        self._decision = DecisionEngine()
        self._alarm = AlarmManager()
        self._fps = FPSCounter()
        self._event_log = DetectionLogger()
        self._frame_count = 0
        logger.info("All subsystems ready.")

    @staticmethod
    def _visible_eye_flags(crops: EyeCrops, pose: FacePose) -> tuple[bool, bool]:
        left_visible = (
            crops.left is not None and pose.left_eye_visibility >= MIN_EYE_VISIBILITY
        )
        right_visible = (
            crops.right is not None and pose.right_eye_visibility >= MIN_EYE_VISIBILITY
        )
        return left_visible, right_visible

    def _run_cnn(self, crops: EyeCrops, use_left: bool, use_right: bool):
        left_input = preprocess_eye(crops.left) if use_left else None
        right_input = preprocess_eye(crops.right) if use_right else None
        return self._predictor.predict_average(left_input, right_input)

    def _process_frame(self, frame):
        """Execute one full detection cycle on a single frame."""
        self._frame_count += 1
        self._fps.tick()

        face_result = self._face_detector.detect(frame)

        eye_label = "NO FACE"
        probability = 0.0
        confidence = 0.0
        ear: float | None = None
        decision_label = "UNCERTAIN"
        decision_reason = "NO FACE"
        is_drowsy = False
        ear_consec = 0
        alarm_triggered = False
        reliability = 0.0
        quality_label = "LOW"
        visible_eye_count = 0
        has_prediction = False
        cnn_ready = False

        if face_result is not None:
            landmarks = face_result.landmarks_2d
            pose = face_result.pose
            crops = extract_eyes(frame, landmarks)
            reliability = pose.reliability
            quality_label = pose.quality_label

            use_left, use_right = self._visible_eye_flags(crops, pose)
            visible_eye_count = int(use_left) + int(use_right)
            can_run_cnn = (
                visible_eye_count > 0 and reliability >= MIN_RELIABILITY_FOR_CNN
            )

            if can_run_cnn:
                try:
                    prediction = self._run_cnn(crops, use_left, use_right)
                    probability = prediction.probability
                    confidence = prediction.confidence
                    has_prediction = self._smoother.update(
                        prediction.class_index,
                        prediction.confidence,
                        reliability,
                    )
                    cnn_ready = self._smoother.has_enough_samples
                    if has_prediction:
                        eye_label = (
                            "CLOSED"
                            if self._smoother.smoothed_class == 1
                            else "OPEN"
                        )
                    else:
                        eye_label = "LOW CONF"
                except Exception as exc:
                    logger.warning("CNN inference failed: %s", exc)
                    self._smoother.mark_unreliable()
                    eye_label = "CNN SKIPPED"
            else:
                self._smoother.mark_unreliable()
                eye_label = "LOW RELIABILITY" if visible_eye_count else "FACE PARTIAL"

            try:
                ear = average_available_ear(
                    landmarks,
                    LEFT_EYE,
                    RIGHT_EYE,
                    use_left=use_left,
                    use_right=use_right,
                )
            except Exception as exc:
                logger.warning("EAR computation failed: %s", exc)
                ear = None

            result = self._decision.evaluate(
                closed_ratio=self._smoother.closed_ratio,
                ear=ear,
                cnn_ready=cnn_ready,
                reliability=reliability,
                visible_eye_count=visible_eye_count,
                has_prediction=has_prediction,
            )
            decision_label = result.decision_label
            decision_reason = result.trigger_reason
            is_drowsy = result.is_drowsy
            ear_consec = self._decision.ear_consecutive_count

            if is_drowsy:
                alarm_triggered = self._alarm.trigger()

            box_color = {
                "DROWSY": (0, 0, 220),
                "UNCERTAIN": (0, 200, 220),
                "ALERT": (0, 220, 0),
            }[decision_label]
            draw_face_bbox(frame, face_result.face_bbox, color=box_color)
        else:
            self._decision.reset()
            self._smoother.reset()

        self._event_log.log(
            frame_number=self._frame_count,
            cnn_probability=probability,
            ear_value=ear,
            smoothed_state=eye_label,
            final_decision=decision_label,
            alarm_triggered=alarm_triggered,
        )

        draw_status_panel(
            frame=frame,
            eye_label=eye_label,
            confidence=confidence,
            ear=ear,
            fps=self._fps.fps,
            decision=decision_label,
            alarm_active=self._alarm.is_playing,
            ear_consec=ear_consec,
            reliability=reliability,
            quality_label=quality_label,
            reason=decision_reason,
        )

        if decision_label == "UNCERTAIN":
            draw_uncertain_banner(frame, decision_reason)
        elif is_drowsy:
            draw_drowsy_banner(frame)

        return frame, is_drowsy

    def run(self) -> None:
        """Start the real-time detection loop."""
        logger.info("Starting real-time detection. Press 'q' to quit.")

        try:
            while True:
                ok, frame = self._webcam.read()
                if not ok or frame is None:
                    logger.error("Webcam stopped providing frames. Exiting.")
                    break

                annotated, _ = self._process_frame(frame)
                cv2.imshow(WINDOW_TITLE, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("User requested exit.")
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Release all resources cleanly."""
        logger.info("Shutting down...")
        self._webcam.release()
        self._face_detector.close()
        self._alarm.close()
        self._event_log.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")
