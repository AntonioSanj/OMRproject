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

# stave and figure detection
flatFigure = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/figureDetection/figure_templates/flat_figure.png'
sharpFigure = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/figureDetection/figure_templates/sharp_figure.png'
restDoubleFigure = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/figureDetection/figure_templates/restDouble_figure.png'
flatFigureMask = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/figureDetection/figure_templates/flat_figure_mask.png'
noteHeadTemplate = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/noteHeadDetection/notehead_template.png'
fourHeadTemplate = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/noteHeadDetection/four_head_template.png'
outputVision = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\output\staveDetection_output'

# first approximation F-RCNN
myDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\img'
myDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann'
myDataCsv2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann2'
myDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\json'
myDataCoco = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\coco\coco_annotations.json'
myDataCoco2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\coco\coco_annotations2.json'
modelsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/models/'

# second approximation F-RCNN
mySlicedDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\img'
mySlicedDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\ann'
mySlicedDataCsv2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\ann2'
mySlicedDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\json'
mySlicedDataCoco = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\coco\coco_annotations.json'
mySlicedDataCoco2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\coco\coco_annotations2.json'
slicedModelsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/sliceModels/'

# fullsheets path
fullsheetsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/fullsheets/'

# figure classifier
myFiguresDataSet = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/myFiguresDataSet/train/'
myFiguresDataSetTest = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/myFiguresDataSet/test/'
figureModels = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/models/'
figuresPerformanceDataJson = r'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/performance_data/performance_data.json'
figuresPerformanceDataJson2 = r'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/performance_data/performance_data2.json'



