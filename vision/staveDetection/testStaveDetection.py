import cv2

from constants import *
from staveDetection import getStaves
import os


def testProcessImage(sourceDir, outputDir, show=False, printData=False, maxData=-1, debug=False):
    # check sourceDir exists
    if not os.path.exists(sourceDir):
        raise FileNotFoundError(f"Source directory '{sourceDir}' does not exist.")

    # check outputDir exists
    if not os.path.exists(outputDir):
        raise FileNotFoundError(f"Output directory '{outputDir}' does not exist.")

    # clear output folder
    for filename in os.listdir(outputDir):
        file_path = os.path.join(outputDir, filename)
        os.remove(file_path)

    count = 0

    for filename in os.listdir(sourceDir):

        # stop processing when maxData is reached
        if maxData != -1 and count > maxData - 1:
            print(f"STAVE DETECTION FINISHED. REACHED MAX-DATA={maxData}")
            return

        # get file path
        filePath = os.path.join(sourceDir, filename)

        # process image
        image, _ = getStaves(filePath, show, printData, debug)

        # save image
        cv2.imwrite(outputDir + '/output_' + filename, image)

        count += 1
        print(f"({count}):\t", filename, "\t\tStaves detected")

    print(f"STAVE DETECTION FINISHED. {count} IMAGES PROCESSED")
    return


testProcessImage(myDataImg, outputVision, True, True)
