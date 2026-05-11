"""
utils/logger.py
Append-mode CSV logger for drowsiness detection events.
"""

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.config import LOG_PATH


_FIELDNAMES = [
    "timestamp",
    "frame_number",
    "cnn_probability",
    "ear_value",
    "smoothed_state",
    "final_decision",
    "alarm_triggered",
]


class DetectionLogger:
    """
    Thread-safe logger that appends one row per frame to a CSV file.

    The file is created (with headers) if it does not yet exist.

    Parameters
    ----------
    log_path : Path
        Destination CSV file path.  Defaults to LOG_PATH from config.
    """

    def __init__(self, log_path: Path = LOG_PATH) -> None:
        self._path = Path(log_path)
        self._lock = threading.Lock()
        self._ensure_file()

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create the CSV file with headers if it does not exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
                writer.writeheader()

    # ── Public ────────────────────────────────────────────────────────────────

    def log(
        self,
        frame_number: int,
        cnn_probability: float,
        ear_value: Optional[float],
        smoothed_state: str,
        final_decision: str,
        alarm_triggered: bool,
    ) -> None:
        """
        Append a single detection event row to the CSV.

        Parameters
        ----------
        frame_number : int
            Sequential frame index since the system started.
        cnn_probability : float
            Raw CNN probability for the CLOSED class (0–1).
        ear_value : float
            Eye Aspect Ratio computed for this frame.
        smoothed_state : str
            Label after temporal smoothing ("OPEN" / "CLOSED").
        final_decision : str
            Final decision label ("ALERT" / "DROWSY").
        alarm_triggered : bool
            Whether the alarm was fired this frame.
        """
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_number": frame_number,
            "cnn_probability": round(cnn_probability, 4),
            "ear_value": "" if ear_value is None else round(ear_value, 4),
            "smoothed_state": smoothed_state,
            "final_decision": final_decision,
            "alarm_triggered": int(alarm_triggered),
        }
        with self._lock:
            with self._path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
                writer.writerow(row)

    def close(self) -> None:
        """No-op; provided for symmetry with resource-managed objects."""
        pass
