# image data
IMAGE_WIDTH = 1960
IMAGE_HEIGHT = 2772
SLICE_WIDTH = 800
SLICE_HEIGHT = 500
IMAGE_CENTER = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
COLORS = [(237, 161, 223), (31, 255, 154), (255, 0, 0), (0, 0, 255)]

notePitchLabels = {
    1: 'C',
    2: 'D',
    3: 'E',
    4: 'F',
    5: 'G',
    6: 'A',
    7: 'B'
}

noteDurations = {
    'one': 1,
    'half': 0.5,
    'double': 2,
    'quarter': 0.25,
    'restHalf': 0.5,
    'four': 4,
    'restOne': 1,
    'restDouble': 2,
}

classColors = {
    'gClef': 'red',
    'fClef': 'green',
    'one': 'blue',
    'half': 'yellow',
    'double': 'pink',
    'quarter': 'orange',
    'restHalf': 'brown',
    'four': 'magenta',
    'restOne': 'cyan',
    'restDouble': 'purple',
    'flat': 'lime',
    'sharp': 'lime',
    'bar': 'teal',
    'dot': 'lime',
}

# DIRECTORIES

# stave and figure detection
flatFigureTemplates = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/resources/figure_templates/flat'
sharpFigureTemplates = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/resources/figure_templates/sharp'
restDoubleTemplates = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/resources/figure_templates/restDouble'

flatFigureMask = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/resources/figure_templates/flat_mask.png'
FLAT_FIGURE_HEAD_HEIGHT = 27
noteHeadTemplate = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/noteHeadDetection/notehead_template.png'
fourHeadTemplate = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/vision/noteHeadDetection/four_head_template.png'
outputVision = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\output\staveDetection_output'
fastRCNNOutput = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/output/fastRCNN_output'

# first approximation F-RCNN
myDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\img'
myDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann'
myDataCsv2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\ann2'
myDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\json'
myDataCoco = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\coco\coco_annotations.json'
myDataCoco2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\myDataSet\coco\coco_annotations2.json'
modelsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/models/'
frcnnPerformanceFull = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/performance_data/full'
frcnnPerformanceSlice = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/performance_data/sliced'


# second approximation F-RCNN
mySlicedDataImg = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\img'
mySlicedDataCsv = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\ann'
mySlicedDataCsv2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\ann2'
mySlicedDataJson = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\json'
mySlicedDataCoco = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\coco\coco_annotations.json'
mySlicedDataCoco2 = r'C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\mySlicedDataSet\coco\coco_annotations2.json'
slicedModelsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/sliceModels/'


# performance data
figClassPerformanceDataDir = "C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/performance_data"

# fullsheets path
fullsheetsDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/fullsheets/'

# figure classifier
myFiguresDataSet = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/myFiguresDataSet/train/'
myFiguresDataSetTest = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/dataset/myFiguresDataSet/test/'
figureModels = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/models/'
figuresPerformanceDataJson = r'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/performance_data/performance_data.json'
figuresPerformanceDataJson2 = r'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/figureClassification/performance_data/performance_data2.json'

# sound files
soundFilesDir = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/resources/sounds'

