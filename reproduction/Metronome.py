import threading
import time


class Metronome:
    def __init__(self, bpm):
        self.bpm = bpm
        self.beat_duration = 60 / bpm  # duration of a quarter beat
        self.current_beat = 0.0
        self.running = False
        self.start_time = None
        self.lock = threading.Lock()

    def start(self):

        self.running = True
        self.start_time = time.time()
        while self.running:
            elapsed_time = time.time() - self.start_time
            beat = elapsed_time / self.beat_duration  # obtains the beat

            with self.lock:
                self.current_beat = round(beat, 2)

            time.sleep(0.005)

    def get_pulse(self):
        # will return the current beat
        with self.lock:
            return self.current_beat

    def stop(self):
        """Stops the clock."""
        self.running = False

