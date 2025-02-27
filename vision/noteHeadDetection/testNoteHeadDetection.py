import os
import random

from PIL import Image

from constants import myFiguresDataSet
from vision.noteHeadDetection.noteHeadDetector import getNoteHead, getFourHeadCenters


def testNoteHeadDetectionFolder(folderDir, noteType=''):
    for file in os.listdir(folderDir):
        if file.lower().endswith('.png'):
            testNoteHeadDetection(os.path.join(folderDir, file), noteType=noteType, fileName=file)


def testNoteHeadDetectionRandomPicks(folderDir, noteType='', num_images=10):
    image_files = [file for file in os.listdir(folderDir) if file.lower().endswith('.png')]

    if num_images < len(image_files):
        image_files = random.sample(image_files, num_images)  # select random images

    for file in image_files:
        testNoteHeadDetection(os.path.join(folderDir, file), noteType=noteType, fileName=file)


def testNoteHeadDetection(imagePath, noteType='', fileName='Note head detection'):
    img = Image.open(imagePath).convert("RGB")
    getNoteHead(img, noteType=noteType, show=True, fileName=fileName)


def testGetFourHeadCenters(imagePath, fileName=''):
    img = Image.open(imagePath).convert("RGB")
    getFourHeadCenters(img, show=True, fileName=fileName)


def testGetFourHeadCentersRandomPick(folderDir, num_images=10):
    image_files = [file for file in os.listdir(folderDir) if file.lower().endswith('.png')]

    if num_images < len(image_files):
        image_files = random.sample(image_files, num_images)  # select random images

    for file in image_files:
        testGetFourHeadCenters(os.path.join(folderDir, file), fileName=file)


# testNoteHeadDetection(myFiguresDataSet + '/double/figure_108.png', 'double')
# testNoteHeadDetection(myFiguresDataSet + '/quarter/figure_103.png')
# testNoteHeadDetection(myFiguresDataSet + '/quarter/figure_113.png')
# testNoteHeadDetectionFolder(myFiguresDataSet + '/double', 'double')
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/half', 'half', 10)
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/one', 'one', 10)
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/double', 'double', 10)
# testNoteHeadDetectionRandomPicks(myFiguresDataSet + '/quarter', 'quarter', 10)
testGetFourHeadCentersRandomPick(myFiguresDataSet + '/four', 10)
