class Figure:
    def __init__(self, box, label, score):
        self.box = [int(coord) for coord in box]
        self.type = label
        self.score = score
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]
        self.image = None
        self.noteHeads = []

    def toString(self):
        print(f"BOX: {self.box}\tTYPE: {self.type}")

    def getCenter(self):
        return self.box[0] + self.width // 2, self.box[1] + self.height // 2
