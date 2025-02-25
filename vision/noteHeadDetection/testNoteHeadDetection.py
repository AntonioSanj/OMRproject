import os
from PIL import Image

from constants import myFiguresDataSet
from vision.noteHeadDetection.noteHeadDetector import getNoteHead


def testNoteHeadDetectionFolder(folderDir):
    for file in os.listdir(folderDir):
        if file.lower().endswith('.png'):
            testNoteHeadDetection(os.path.join(folderDir, file), file)


def testNoteHeadDetection(imagePath, fileName="Note head detection"):
    img = Image.open(imagePath).convert("RGB")
    getNoteHead(img, True, fileName)


testNoteHeadDetection(myFiguresDataSet + '/one/figure_17.png')
testNoteHeadDetection(myFiguresDataSet + '/double/figure_4.png')
testNoteHeadDetection(myFiguresDataSet + '/quarter/figure_2.png')
testNoteHeadDetectionFolder(myFiguresDataSet + '/quarter')
