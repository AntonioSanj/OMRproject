
class Note:
    def __init__(self, pitch, octave):
        self.pitch = pitch
        self.octave = octave
        self.duration = 0
        self.accidental = 'n'    # 'n' for natural, 'b' for flat or '#' for sharp
        self.noteHead = (0, 0)
