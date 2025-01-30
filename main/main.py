from PIL import Image

from constants import *
from mainFunctions import obtainSliceHeights, getPredictions, startModel, showPredictions, mergeFigures, \
    translateToFullSheet, filterOutBorderFigures, saveFigures
from vision.staveDetection.staveDetection import getStaves

imagePath = myDataImg + '/image_1.png'

_, staves = getStaves(imagePath)

image = Image.open(imagePath).convert("RGB")

# verify number of staves is even
if len(staves) % 2 != 0:
    raise ValueError(f"Stave detetection went wrong. Staves detected: {len(staves)}")

i = 0
x_increment = int(SLICE_WIDTH / 3)

model, device = startModel(slicedModelsDir + 'fasterrcnn_epoch_9.pth', 10)

fullSheetFigures = []

while i < (len(staves)):
    print(f"ANALYSING STAVE PAIR {int(i / 2) + 1} ", end="")
    sliceTop, sliceBottom = obtainSliceHeights(staves[i], staves[i + 1])
    j = 0
    while j < IMAGE_WIDTH - SLICE_WIDTH:
        print("::::::", end="")

        slicedImage = image.crop((j, sliceTop, j + SLICE_WIDTH, sliceBottom))

        sliceFigures = getPredictions(slicedImage, model, 0.15, device)

        sliceFigures = filterOutBorderFigures(sliceFigures, 30)

        # showPredictions(slicedImage, sliceFigures)

        sliceFigures = mergeFigures(sliceFigures)

        sliceFigures = translateToFullSheet(sliceFigures, j, sliceTop)

        fullSheetFigures = fullSheetFigures + sliceFigures

        j = j + x_increment

    i = i + 2

    print("  COMPLETED")

fullSheetFigures = mergeFigures(fullSheetFigures, 0.3)

showPredictions(image, fullSheetFigures)

saveFigures(image, fullSheetFigures, myFiguresDataSet, 0)
