from enum import Enum

# image processing
IMAGE_WIDTH = 1960
IMAGE_HEIGHT = 2772
WHITE = (255, 255, 255)
RED = (255, 0, 0)
IMAGE_CENTER = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
COLORS = [(237, 161, 223), (31, 255, 154), (255, 0, 0), (0, 0, 255)]


class Clef(Enum):
    G = 'G'
    F = 'F'
    C = 'C'
    UNDEFINED = '_'


outputVision = 'output/outputVision'

challengingSet = 'dataset/codingSet/challengingSet'
challengingSet2 = 'dataset/codingSet/challengingSet2'

myDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\img'
myDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann'
myDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\json'
