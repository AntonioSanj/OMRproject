from constants import *
from vision.figureDetection.figureDetection import extractFigureLocations
from vision.figureDetection.pointDetection import getPointModifications

# extractFigureLocations(fullsheetsDir + 'this_is_me1.png', sharpFigure, 0.65, True, True)
extractFigureLocations(fullsheetsDir + 'roar1.png', flatFigure, 0.6, templateMask_path=flatFigureMask, show=True,
                       print_points=True)
# extractFigureLocations(myDataImg + '/image_13.png', restDoubleFigure, 0.8, show=True, print_points=True)
getPointModifications(myDataImg + '/image_13.png', True, True)
