class SoundDTO:
    # specifies how long a single sound file
    def __init__(self, sound, duration):
        self.sound = sound  # 'C4' 'E3#' 'A5b' 'rest'
        self.duration = duration

    def toString(self):
        return '(' + self.sound + ',' + str(self.duration) + ')'


class MultiSound:
    # may contain more than one soundDTO that will start to play at the same time
    def __init__(self, start, duration):
        self.sounds = []
        self.start = start
        self.duration = duration

    def toString(self):
        return '[' + ''.join([sound.toString() for sound in self.sounds]) + ']'


class Song:
    def __init__(self, mb, bpm):
        self.measuresUp = []
        self.measuresDown = []
        self.measureBeats = mb
        self.bpm = bpm

    def toString(self):
        return (
                f"Song: {self.measureBeats} beats per measure, {self.bpm} BPM\n\n"
                f"Upper Sounds:\n" + "\n".join(ms.toString() for ms in self.measuresUp) + "\n\n"
                f"Lower Sounds:\n" + "\n".join(ms.toString() for ms in self.measuresDown)
        )
