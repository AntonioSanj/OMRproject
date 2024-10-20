from enum import Enum

IMAGE_WIDTH = 1960
IMAGE_HEIGHT = 2772
WHITE = (255, 255, 255)
IMAGE_CENTER = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


class Clef(Enum):
    G = 'G'
    F = 'F'
    UNDEFINED = 'UNDEFINED'
