class SoundDTO:
    # specifies how long a single sound file
    def __init__(self, sound, duration):
        self.sound = sound  # 'C4' 'E3#' 'A5b' 'rest'
        self.duration = duration

    def toString(self):
        return '(' + self.sound + ',' + str(self.duration) + ')'


class MultiSound:
    # may contain more than one soundDTO that will start to play at the same time
    def __init__(self, sounds, start, duration):
        self.sounds = sounds
        self.start = start
        self.duration = duration

    def toString(self):
        return '[' + ''.join([sound.toString() for sound in self.sounds]) + ', ' + str(self.start) + ']'


class Song:
    def __init__(self, upperTrack, lowerTrack, mb, bpm):
        self.upperTrack = upperTrack
        self.lowerTrack = lowerTrack
        self.measureBeats = mb
        self.bpm = bpm

    def toString(self):
        return (
                f"Song: {self.measureBeats} beats per measure, {self.bpm} BPM\n\n"
                f"Upper Sounds: " + " ".join(ms.toString() for ms in self.upperTrack) + "\n"
                                                                                        f"Lower Sounds: " + " ".join(
            ms.toString() for ms in self.lowerTrack)
        )
