import cv2

from constants import *
from staveDetection import getStaves
import os


def testStaveDetection(sourceDir, outputDir, printData=False, debug=False):
    for filename in os.listdir(sourceDir):
        filePath = os.path.join(sourceDir, filename)
        image, _ = getStaves(filePath, 0, printData=printData, debug=debug, show=True)
        # cv2.imwrite(outputDir + '/output_' + filename, image)

    return


# getStaves(fullsheetsDir + 'roar1.png', 0, debug=True)
testStaveDetection(myDataImg, outputVision, debug=False)
testStaveDetection(fullsheetsDir, outputVision, debug=False)
