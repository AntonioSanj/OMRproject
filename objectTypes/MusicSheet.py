
class MusicSheet:
    def __init__(self, index, path, image, staves):
        self.index = index
        self.path = path
        self.image = image
        self.staves = staves
        self.figures = []  # a list of all the figures in the stave

