
class Measure:
    def __init__(self, staveIndex, figures):
        self.staveIndex = staveIndex
        self.figures = figures
        self.duration = sum([fig.duration for fig in figures])


