"""
main.py
Entry point for the Driver Drowsiness Detection System.

Usage
-----
    python main.py              # use default camera (index 0)
    python main.py --camera 1   # use camera at index 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure the project root is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime.realtime_detection import RealtimeDetection


def _configure_logging() -> None:
    """Set up human-readable console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (0 = built-in, 1+ = external).",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    logger = logging.getLogger(__name__)
    logger.info("Driver Drowsiness Detection System starting …")
    logger.info("Camera index: %d", args.camera)
    logger.info("Press 'q' or ESC in the video window to quit.")

    try:
        detector = RealtimeDetection(camera_index=args.camera)
        detector.run()
    except FileNotFoundError as exc:
        logger.error("Startup error: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
