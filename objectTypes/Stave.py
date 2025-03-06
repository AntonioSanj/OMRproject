
class Stave:
    def __init__(self, staveIndex, meanGap):
        self.staveIndex = staveIndex
        self.meanGap = meanGap
        self.lineHeights = []
        self.topLine = -1
        self.bottomLine = -1
        self.figures = []

    def setLineHeights(self, lineHeights):
        self.lineHeights = lineHeights

    def setTopLine(self, value):
        self.topLine = value

    def setBottomLine(self, value):
        self.bottomLine = value

    def addLineHeight(self, value):
        self.lineHeights.append(value)

    def addLines(self, lines, sortTrigger=False):
        self.lineHeights.extend(lines)
        if sortTrigger:
            self.lineHeights.sort()

    def getHeightCenter(self):
        return self.topLine + (self.bottomLine - self.topLine) // 2

    def print(self):
        print(f"Stave {self.staveIndex}:")
        print(f"\tTopLine: {self.topLine}.")
        print(f"\tBottomLine: {self.bottomLine}.")
        print(f"\tLines at: {self.lineHeights}.")
