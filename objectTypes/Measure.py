
class Measure:
    def __init__(self, staveIndex, sheetIndex, figures):
        self.staveIndex = staveIndex
        self.sheetIndex = sheetIndex
        self.figures = figures
        self.duration = sum([fig.duration for fig in figures])


