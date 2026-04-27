import os
import threading
import time
import logging
from playsound import playsound

# Assuming config.py is in the base directory
try:
    from config import ALARM_PATH, ALARM_DURATION_SECONDS, LOG_FILE
except ImportError:
    print("Warning: config.py not found. Using default alarm parameters.")
    ALARM_PATH = 'assets/alarm.wav' # Default path for alarm sound
    ALARM_DURATION_SECONDS = 3
    LOG_FILE = 'events.log' # Fallback for logging

# Setup basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )

class AlarmSystem:
    """Manages the audio alarm for drowsiness detection."""

    def __init__(self):
        self.alarm_thread = None
        self.stop_alarm_event = threading.Event()
        self.alarm_active = False
        if not os.path.exists(ALARM_PATH):
            logging.warning(f"Alarm sound file not found at {ALARM_PATH}. Alarm will not function.")
            self.alarm_enabled = False
        else:
            self.alarm_enabled = True
            logging.info(f"Alarm system initialized. Sound file: {ALARM_PATH}")

    def _play_alarm(self):
        """Internal method to play the alarm sound repeatedly for a duration."""
        logging.info("Alarm triggered and playing...")
        self.alarm_active = True
        start_time = time.time()
        while not self.stop_alarm_event.is_set() and (time.time() - start_time < ALARM_DURATION_SECONDS):
            try:
                playsound(ALARM_PATH, block=False)
                time.sleep(0.5)  # Small delay to allow sound to play and prevent CPU spike
            except Exception as e:
                logging.error(f"Error playing alarm sound: {e}")
                break # Exit loop if sound can't be played
        self.alarm_active = False
        logging.info("Alarm stopped.")

    def trigger_alarm(self):
        """Starts the alarm in a separate thread if not already active."""
        if not self.alarm_enabled:
            logging.warning("Alarm is not enabled (sound file not found).")
            return

        if not self.alarm_active:
            self.stop_alarm_event.clear()
            self.alarm_thread = threading.Thread(target=self._play_alarm)
            self.alarm_thread.daemon = True # Allow main program to exit even if thread is running
            self.alarm_thread.start()
            logging.info("Alarm trigger command sent.")
        else:
            logging.debug("Alarm already active, skipping trigger.")

    def stop_alarm(self):
        """Stops the currently playing alarm."""
        if self.alarm_active:
            self.stop_alarm_event.set()
            if self.alarm_thread and self.alarm_thread.is_alive():
                self.alarm_thread.join(timeout=1) # Wait for thread to finish playing
                if self.alarm_thread.is_alive():
                    logging.warning("Alarm thread did not terminate in time.")
            self.alarm_active = False
            logging.info("Alarm manually stopped.")
        else:
            logging.debug("Alarm not active, no need to stop.")

if __name__ == '__main__':
    # Example usage for testing
    logging.info("Testing AlarmSystem. Make sure 'assets/alarm.wav' exists.")

    # Create a dummy assets directory and file for testing if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    dummy_alarm_path = 'assets/alarm.wav'
    if not os.path.exists(dummy_alarm_path):
        # In a real scenario, you'd place a proper .wav file here.
        # For colab, we can't easily create a playable .wav file without external libraries
        # or pre-downloading. This is just to satisfy the os.path.exists check.
        # For actual testing, replace with a real sound file.
        print(f"NOTE: No actual alarm sound will play unless {dummy_alarm_path} is a valid .wav file.")
        # Creating a dummy file to pass the os.path.exists check, won't play sound.
        with open(dummy_alarm_path, 'w') as f:
            f.write("dummy sound data")

    alarm = AlarmSystem()

    if alarm.alarm_enabled:
        print("Triggering alarm for 5 seconds...")
        alarm.trigger_alarm()
        time.sleep(5)
        print("Stopping alarm...")
        alarm.stop_alarm()
        print("Test complete.")
    else:
        print("Alarm system disabled due to missing sound file.")
