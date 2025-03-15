from constants import *
from vision.figureDetection.figureDetection import extractFigureLocations
from vision.figureDetection.pointDetection import getPointModifications

#extractFigureLocations(fullsheetsDir + '/this_is_me1.png', sharpFigureTemplates, 0.65, show=True, print_points=True)
#extractFigureLocations(fullsheetsDir + '/roar1.png', flatFigureTemplates, 0.6, templateMask_path=flatFigureMask, show=True, print_points=True)
#extractFigureLocations(myDataImg + '/image_13.png', restDoubleTemplates, 0.8, show=True, print_points=True)
extractFigureLocations(fullsheetsDir + '/the_chesire_cat.png', restDoubleTemplates, 0.8, show=True, print_points=True)
getPointModifications(myDataImg + '/image_13.png', True, True)
