class Figure:
    def __init__(self, box, label, score, image=None):
        self.box = [int(coord) for coord in box]
        self.type = label
        self.score = score
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]
        self.image = image

    def toString(self):
        print(f"Figure {self.type} at {self.getCenter()}")

    def getCenter(self):
        return self.box[0] + self.width // 2, self.box[1] + self.height // 2


class NoteFigure(Figure):
    def __init__(self, box, label, score, image=None):
        super().__init__(box, label, score, image)
        self.noteHeads = []
        self.notes = []
        self.articulation = 'n'  # 'n' for natural, 's' for staccato
        self.isSignature = False

    @classmethod
    def fromFigure(cls, figure):
        # class convertion method
        return cls(figure.box, figure.type, figure.score, figure.image)


class ClefFigure(Figure):
    def __init__(self, box, label, score):
        super().__init__(box, label, score)
        self.signature = []

    @classmethod
    def fromFigure(cls, figure):
        # class convertion method
        return cls(figure.box, figure.type, figure.score)


class RestFigure(Figure):
    def __init__(self, box, label, score):
        super().__init__(box, label, score)
        self.duration = 0

    @classmethod
    def fromFigure(cls, figure):
        return cls(figure.box, figure.type, figure.score)
