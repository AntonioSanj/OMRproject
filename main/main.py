from PIL import Image

from constants import *
from mainFunctions import obtainSliceHeights, getPredictions, startModel, showPredictions
from vision.staveDetection.staveDetection import getStaves

imagePath = myDataImg + r'\image_13.png'

print(imagePath)
_, staves = getStaves(imagePath)

image = Image.open(imagePath).convert("RGB")

# verify number of staves is even
if len(staves) % 2 != 0:
    raise ValueError(f"Stave detetection went wrong. Staves detected: {len(staves)}")

i = 0
x_increment = int(SLICE_WIDTH/3)

model, device = startModel(slicedModelsDir + 'fasterrcnn_epoch_6.pth', 10)

while i < (len(staves)):
    print("STAVE PAIR", i / 2 + 1)
    sliceTop, sliceBottom = obtainSliceHeights(staves[i], staves[i + 1])
    j = 0
    while j < IMAGE_WIDTH - SLICE_WIDTH:
        print("j:", j)
        slicedImage = image.crop((j, sliceTop, j + SLICE_WIDTH, sliceBottom))

        figures = getPredictions(slicedImage, model, 0.15, device)

        showPredictions(slicedImage, figures)

        j = j + x_increment
    i = i + 10
