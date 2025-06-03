import os
import random

from PIL import Image

from constants import myFiguresDataSet
from vision.noteHeadDetection.noteHeadDetector import getNoteHeads, getNoteHeadsFour


def testNoteHeadDetectionFolder(folderDir, noteType='', debug=False):
    for file in os.listdir(folderDir):
        if file.lower().endswith('.png'):
            testNoteHeadDetection(os.path.join(folderDir, file), noteType=noteType, fileName=file, debug=debug)


def testNoteHeadDetectionRandomPicks(folderDir, noteType='', debug=False, num_images=10):
    image_files = [file for file in os.listdir(folderDir) if file.lower().endswith('.png')]

    if num_images < len(image_files):
        image_files = random.sample(image_files, num_images)  # select random images

    for file in image_files:
        testNoteHeadDetection(os.path.join(folderDir, file), noteType=noteType, fileName=file, debug=debug)


def testNoteHeadDetection(imagePath, noteType='', fileName='Note head detection',debug=False):
    img = Image.open(imagePath).convert("RGB")
    getNoteHeads(img, noteType=noteType, show=True, fileName=fileName, debug=debug)


def testGetFourHeadCenters(imagePath, fileName='', debug=False):
    img = Image.open(imagePath).convert("RGB")
    getNoteHeadsFour(img, show=True, fileName=fileName, debug=debug)


def testGetFourHeadCentersRandomPick(folderDir, num_images=10, debug=False):
    image_files = [file for file in os.listdir(folderDir) if file.lower().endswith('.png')]

    if num_images < len(image_files):
        image_files = random.sample(image_files, num_images)  # select random images

    for file in image_files:
        testGetFourHeadCenters(os.path.join(folderDir, file), fileName=file, debug=debug)


# testNoteHeadDetection(myFiguresDataSet + '/double/figure_108.png', 'double')
# testNoteHeadDetection(myFiguresDataSet + '/quarter/figure_103.png')
# testNoteHeadDetection(myFiguresDataSet + '/quarter/figure_113.png', debug=True)
# testNoteHeadDetectionFolder(myFiguresDataSet + '/double', 'double')
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/half', 'half', 10)
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/one', 'one', True, 10)
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/double', 'double', 10)
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/quarter', 'quarter', 10)
testGetFourHeadCentersRandomPick(myFiguresDataSet + '/four', 10, debug=True)
