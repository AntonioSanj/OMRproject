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

    def start(self, n):
        # starts the clock that keeps track of song pulses in 0.25 increments
        self.running = True
        self.start_time = time.time()  # Set start time
        last_printed_pulse = -1
        while self.running:
            elapsed_time = time.time() - self.start_time
            new_pulse = elapsed_time / self.pulse_duration  # Convert to pulse count

            with self.lock:
                self.current_pulse = round(new_pulse, 2)  # Round for better precision

            # Print '|' every `n` pulses
            if int(self.current_pulse) % n == 0 and int(self.current_pulse) != last_printed_pulse:
                print(" | ", end="", flush=True)  # Print without a newline
                last_printed_pulse = int(self.current_pulse)  # Update last printed pulse

            time.sleep(0.005)

    def get_pulse(self):
        """Returns the current pulse (can be decimal)."""
        with self.lock:
            return self.current_pulse

    def stop(self):
        """Stops the clock."""
        self.running = False

