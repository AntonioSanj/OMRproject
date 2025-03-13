from PIL import Image

from constants import *
from mainFunctions import obtainSliceHeights, getPredictions, startModel, mergeFigures, \
    translateToFullSheet, filterOutBorderFigures, startFiguresModel, classifyFigures, \
    getNoteHeadCenters, detectTemplateFigures, distributeFiguresInStaves, detectMeasureBarLines, detectPoints, \
    handleCorrections, showPredictionsStaves, assignNotes, getKeySignatures, assignObjectTypes, applyAccidentals, \
    applyKeySignature, assignNoteDurations, applyDots, convertToMeasures, adjustMeasuresToBeat, \
    showPredictionMeasures, createSong
from utils.plotUtils import showImage
from vision.staveDetection.staveDetection import getStaves

imagePath = fullsheetsDir + '/thinking_out_loud1.png'

staveLinesImage, staves = getStaves(imagePath)

showImage(staveLinesImage, 'Staves found')

image = Image.open(imagePath).convert("RGB")

# verify number of staves is even
if len(staves) % 2 != 0:
    raise ValueError(f"Stave detection went wrong. Staves detected: {len(staves)}")

i = 0
x_increment = int(SLICE_WIDTH / 3)

model, device = startModel(slicedModelsDir + 'fasterrcnn_epoch_9.pth', 10)

figures = []

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

        figures = figures + sliceFigures

        j = j + x_increment

    i = i + 2

    print("  COMPLETED")

figures = mergeFigures(figures, 0.3)

# saveFigures(image, fullSheetFigures, myFiguresDataSet, 0)
print('Running figure classification..... ', end='')

figureClassificationModel = startFiguresModel(figureModels + 'figure_classification_model.pth')

figures = classifyFigures(figures, figureClassificationModel, image)

print('\t\tCOMPLETED')

figures = assignObjectTypes(figures)

figures = getNoteHeadCenters(figures)

figures = detectTemplateFigures(imagePath, figures)

figures = detectMeasureBarLines(imagePath, figures)

figures = detectPoints(imagePath, figures)

staves = distributeFiguresInStaves(figures, staves)

staves = handleCorrections(staves)

staves = assignNotes(staves)

staves = getKeySignatures(staves)

staves = applyKeySignature(staves)

staves = applyAccidentals(staves)

staves = assignNoteDurations(staves)

staves = applyDots(staves)

showPredictionsStaves(image, staves, 'types')
showPredictionsStaves(image, staves, 'notes')
showPredictionsStaves(image, staves, 'duration')

# --------------------------------------------------------------------------------------
# END OF IMAGE RECOGNITION
# STARTING REPRODUCTION PROCESS
# --------------------------------------------------------------------------------------

measures, measureBeats = convertToMeasures(staves)

print(measureBeats)

showPredictionMeasures(image, measures)

measures = adjustMeasuresToBeat(measures, measureBeats)

showPredictionMeasures(image, measures)

song = createSong(measures, measureBeats, 100)

print(song.toString())
