"""
alarm/alarm_manager.py
Alarm management with cooldown to prevent continuous retriggering.

Falls back gracefully if pygame is unavailable or the WAV file is missing.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from utils.config import ALARM_COOLDOWN_SECONDS, ALARM_PATH

logger = logging.getLogger(__name__)


def _generate_alarm_wav(path: Path) -> None:
    """
    Write a simple 440 Hz sine-wave WAV file if no alarm file exists.
    This ensures the system works out-of-the-box without external assets.
    """
    import math
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    duration = 2       # seconds
    frequency = 880.0  # Hz – slightly higher than A4 to be more alerting
    num_samples = sample_rate * duration
    amplitude = 28000

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [
            int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            for i in range(num_samples)
        ]
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))

    logger.info("Generated fallback alarm WAV at %s", path)


class AlarmManager:
    """
    Plays an audio alarm when drowsiness is detected, with a configurable
    cooldown period to prevent the alarm from looping continuously.

    The alarm is played on a daemon thread so it never blocks the main
    video-processing loop.

    Parameters
    ----------
    alarm_path : Path
        Path to the WAV alarm file.
    cooldown_seconds : float
        Minimum time (seconds) between successive alarm triggers.
    """

    def __init__(
        self,
        alarm_path: Path = ALARM_PATH,
        cooldown_seconds: float = ALARM_COOLDOWN_SECONDS,
    ) -> None:
        self._alarm_path = Path(alarm_path)
        self._cooldown = cooldown_seconds
        self._last_triggered: float = 0.0
        self._is_playing: bool = False
        self._lock = threading.Lock()
        self._pygame_ready: bool = False

        self._ensure_alarm_file()
        self._init_pygame()

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_alarm_file(self) -> None:
        """Generate a fallback alarm WAV if the configured file is missing."""
        if not self._alarm_path.exists():
            logger.warning(
                "Alarm file not found at %s. Generating a fallback tone.",
                self._alarm_path,
            )
            try:
                _generate_alarm_wav(self._alarm_path)
            except Exception as exc:
                logger.error("Could not generate fallback alarm: %s", exc)

    def _init_pygame(self) -> None:
        """Attempt to initialise pygame mixer; mark unavailable on failure."""
        try:
            import pygame  # noqa: PLC0415
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._pygame = pygame
            self._pygame_ready = True
        except Exception as exc:  # pragma: no cover
            logger.warning("pygame mixer unavailable: %s. Alarm disabled.", exc)
            self._pygame_ready = False

    def _play_thread(self) -> None:
        """Background thread: load and play the alarm WAV once."""
        if not self._pygame_ready:
            return
        try:
            sound = self._pygame.mixer.Sound(str(self._alarm_path))
            sound.play()
            duration_ms = int(sound.get_length() * 1000)
            self._pygame.time.wait(duration_ms)
        except Exception as exc:
            logger.error("Failed to play alarm: %s", exc)
        finally:
            with self._lock:
                self._is_playing = False

    # ── Public ────────────────────────────────────────────────────────────────

    def trigger(self) -> bool:
        """
        Trigger the alarm if the cooldown period has elapsed.

        Returns
        -------
        bool
            True if the alarm was actually triggered this call.
        """
        now = time.monotonic()
        with self._lock:
            if self._is_playing:
                return False
            if (now - self._last_triggered) < self._cooldown:
                return False
            self._last_triggered = now
            self._is_playing = True

        thread = threading.Thread(target=self._play_thread, daemon=True)
        thread.start()
        return True

    def stop(self) -> None:
        """Stop any currently playing alarm sound."""
        if self._pygame_ready:
            try:
                self._pygame.mixer.stop()
            except Exception:
                pass
        with self._lock:
            self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """True while an alarm sound is currently playing."""
        with self._lock:
            return self._is_playing

    @property
    def cooldown_remaining(self) -> float:
        """Seconds until the next alarm can be triggered.  0.0 if ready."""
        elapsed = time.monotonic() - self._last_triggered
        remaining = self._cooldown - elapsed
        return max(0.0, remaining)

    def close(self) -> None:
        """Release pygame mixer resources."""
        self.stop()
        if self._pygame_ready:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass
