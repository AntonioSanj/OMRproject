from enum import Enum

# image data
IMAGE_WIDTH = 1960
IMAGE_HEIGHT = 2772
SLICE_WIDTH = 800
SLICE_HEIGHT = 500
IMAGE_CENTER = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
COLORS = [(237, 161, 223), (31, 255, 154), (255, 0, 0), (0, 0, 255)]

# recognition engine
NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


class Clef(Enum):
    G = 'G'
    F = 'F'
    C = 'C'
    UNDEFINED = '_'


# DIRECTORIES
outputVision = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\output\staveDetection_output'

# first approximation
myDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\img'
myDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann'
myDataCsv2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann2'
myDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\json'
myDataCoco = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\coco\coco_annotations.json'
myDataCoco2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\coco\coco_annotations2.json'
modelsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/models/'

# second approximation
mySlicedDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\img'
mySlicedDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\ann'
mySlicedDataCsv2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\ann2'
mySlicedDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\json'
mySlicedDataCoco = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\coco\coco_annotations.json'
mySlicedDataCoco2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\coco\coco_annotations2.json'
slicedModelsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/sliceModels/'

# fullsheets path
fullsheetsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/fullsheets/'

myFiguresDataSet = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/myFiguresDataSet/'
figureModels = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/models/'
