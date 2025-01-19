from constants import Clef


class Stave:
    def __init__(self, staveIndex):
        self.staveIndex = staveIndex
        self.lineHeights = []
        self.topLine = -1
        self.bottomLine = -1
        self.clef = Clef.UNDEFINED

    def setLineHeights(self, lineHeights):
        self.lineHeights = lineHeights

    def setTopLine(self, value):
        self.topLine = value

    def setBottomLine(self, value):
        self.bottomLine = value

    def setClef(self, clef):
        self.clef = clef

    def addLineHeight(self, value):
        self.lineHeights.append(value)

    def addLines(self, lines, sortTrigger=False):
        self.lineHeights.extend(lines)
        if sortTrigger:
            self.lineHeights.sort()

    def print(self):
        print(f"Stave {self.staveIndex}:")
        print(f"\tClef: {self.clef.value}")
        print(f"\tTopLine: {self.topLine}.")
        print(f"\tBottomLine: {self.bottomLine}.")
        print(f"\tLines at: {self.lineHeights}.")
