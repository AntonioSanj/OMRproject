class Figure:
    def __init__(self, box, label, score):
        self.box = [int(coord) for coord in box]
        self.type = label
        self.score = score

    def toString(self):
        print(f"BOX: {self.box}\tTYPE: {self.type}")
