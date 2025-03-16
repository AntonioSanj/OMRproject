import threading
import time


class PulseClock:
    def __init__(self, bpm):
        self.bpm = bpm
        self.pulse_duration = 60 / bpm  # Duration of a quarter pulse
        self.current_pulse = 0.0  # Start at 0.0 for precision
        self.running = False
        self.start_time = None  # Store start time
        self.lock = threading.Lock()

    def start(self, beatsPerMeasure):

        self.running = True
        self.start_time = time.time()  # Set start time
        while self.running:
            elapsed_time = time.time() - self.start_time
            pulse = elapsed_time / self.pulse_duration  # obtains the pulse

            with self.lock:
                self.current_pulse = round(pulse, 2)

            time.sleep(0.005)

    def get_pulse(self):
        """Returns the current pulse (can be decimal)."""
        with self.lock:
            return self.current_pulse

    def stop(self):
        """Stops the clock."""
        self.running = False

