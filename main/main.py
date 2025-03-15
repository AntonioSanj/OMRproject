from PIL import Image

from constants import *
from mainFunctions import obtainSliceHeights, getPredictions, startModel, mergeFigures, \
    translateToFullSheet, filterOutBorderFigures, startFiguresModel, classifyFigures, \
    getNoteHeadCenters, detectTemplateFigures, distributeFiguresInStaves, detectMeasureBarLines, detectPoints, \
    handleCorrections, showPredictionsStaves, assignNotes, getKeySignatures, assignObjectTypes, applyAccidentals, \
    applyKeySignature, assignNoteDurations, applyDots, adjustMeasuresToBeat, \
    showPredictionMeasures, createSong, convertToTracks, initSheetsWithStaves, showPredictionsFigures
from reproduction.playSong import playSong
from utils.plotUtils import showImage


def readAndPlay(sheetPaths, bpm, show=False):
    print(f'Reading {len(sheetPaths)} sheets:\n' + ''.join(sh + '\n' for sh in sheetPaths))

    sheets = initSheetsWithStaves(sheetPaths)

    model, device = startModel(slicedModelsDir + 'fasterrcnn_epoch_9.pth', 10)
    x_increment = int(SLICE_WIDTH / 3)

    for sheet in sheets:
        # verify number of staves is even
        if len(sheet.staves) % 2 != 0:
            raise ValueError(f"Stave detection went wrong. Staves detected: {len(sheet.staves)}")

        i = 0
        figures = []

        while i < (len(sheet.staves)):
            print(f"Analysing stave pair {int(i / 2) + 1} ", end="")
            sliceTop, sliceBottom = obtainSliceHeights(sheet.staves[i], sheet.staves[i + 1])
            j = 0
            while j < IMAGE_WIDTH - SLICE_WIDTH:
                print("::::::", end="")

                slicedImage = sheet.image.crop((j, sliceTop, j + SLICE_WIDTH, sliceBottom))

                sliceFigures = getPredictions(slicedImage, model, 0.15, device)

                sliceFigures = filterOutBorderFigures(sliceFigures, 30)

                # showPredictions(slicedImage, sliceFigures)

                sliceFigures = mergeFigures(sliceFigures)

                sliceFigures = translateToFullSheet(sliceFigures, j, sliceTop)

                figures = figures + sliceFigures

                j = j + x_increment

            i = i + 2

            print("  COMPLETED")

        # include teh figures to the sheet, at this point we don't know the stave of each figure
        sheet.figures = mergeFigures(figures, 0.3)

    # saveFigures(image, fullSheetFigures, myFiguresDataSet, 0)
    print('Running figure classification..... ', end='')

    figureClassificationModel = startFiguresModel(figureModels + 'figure_classification_model.pth')

    sheets = classifyFigures(sheets, figureClassificationModel)

    print('\t\tCOMPLETED')

    sheets = assignObjectTypes(sheets)

    sheets = getNoteHeadCenters(sheets)

    print('Detecting template figures......', end='')

    sheets = detectTemplateFigures(sheets)

    sheets = detectMeasureBarLines(sheets)

    sheets = detectPoints(sheets)

    print('\t\tCOMPLETED')
    print('Computing dependencies.....', end='')

    sheets = distributeFiguresInStaves(sheets)

    sheets = handleCorrections(sheets)

    sheets = assignNotes(sheets)

    sheets = getKeySignatures(sheets)

    sheets = applyKeySignature(sheets)

    sheets = applyAccidentals(sheets)

    sheets = assignNoteDurations(sheets)

    sheets = applyDots(sheets)

    print('\t\t\t\tCOMPLETED')

    if show:
        showPredictionsStaves(sheets, 'types')
        showPredictionsStaves(sheets, 'notes')
        showPredictionsStaves(sheets, 'duration')

    # --------------------------------------------------------------------------------------
    # END OF IMAGE RECOGNITION
    # STARTING REPRODUCTION PROCESS
    # --------------------------------------------------------------------------------------

    print('Wrapping data for reproduction...', end='')

    tracks, measureBeats = convertToTracks(sheets)

    if show:
        showPredictionMeasures(sheets, tracks)

    tracks = adjustMeasuresToBeat(tracks, measureBeats)

    song = createSong(tracks, measureBeats, bpm)

    print('\t\tCOMPLETED\n')

    print(song.toString())

    if show:
        showPredictionMeasures(sheets, tracks)

    playSong(song)
    print('\n\n')


readAndPlay([fullsheetsDir + '/roar1.png', fullsheetsDir + '/roar2.png'], 90)
readAndPlay([myDataImg + '/image_9.png', myDataImg + '/image_10.png'], 70)
readAndPlay([fullsheetsDir + '/thinking_out_loud1.png', fullsheetsDir + '/thinking_out_loud2.png'], 80)
