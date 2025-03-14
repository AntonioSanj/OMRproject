import math

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional as F

from constants import *
from learning.FasterRCNN.getModel import get_model
from objectTypes.Figure import Figure, ClefFigure, NoteFigure, RestFigure, Accidental, Dot
from objectTypes.Measure import Measure
from objectTypes.Note import Note
from objectTypes.SoundDTO import SoundDTO, Song, MultiSound
from utils.plotUtils import showImage
from vision.figureDetection.figureDetection import extractFigureLocations
from vision.figureDetection.pointDetection import getPointModifications
from vision.measureBarDetection.measureBarDetector import getMeasureBars
from vision.noteHeadDetection.noteHeadDetector import getNoteHeads, getNoteHeadsFour


def obtainSliceHeights(stave1, stave2):
    topLine = stave1.topLine
    bottomLine = stave2.bottomLine

    staveAreaHeight = bottomLine - topLine
    spacing = math.trunc((SLICE_HEIGHT - staveAreaHeight) / 2)
    topSpacing = spacing
    if staveAreaHeight % 2 != 0:
        bottomSpacing = round(spacing + 0.5)
    else:
        bottomSpacing = spacing

    slice_topLine = topLine - topSpacing
    slice_bottomLine = bottomLine + bottomSpacing

    return slice_topLine, slice_bottomLine


def startModel(modelDir, num_classes):
    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(modelDir))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, device


def getPredictions(slicedImage, model, threshold, device):
    image_tensor = F.to_tensor(slicedImage).unsqueeze(0)  # Convert image to tensor and add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    COCO_CLASSES = {0: "Background", 1: "One", 2: "Double", 3: "Four", 4: "Half", 5: "Quarter", 6: "GClef", 7: "FClef",
                    8: "RestOne", 9: "RestHalf"}

    figures = []

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            figure = Figure(box, COCO_CLASSES.get(label), score)
            figures.append(figure)

    return figures


def filterOutBorderFigures(figures, borderSeparation=30):
    res = []
    for figure in figures:
        if figure.box[0] > borderSeparation and figure.box[2] < SLICE_WIDTH - borderSeparation \
                and figure.box[1] > borderSeparation and figure.box[3] < SLICE_HEIGHT - borderSeparation:
            res.append(figure)
    return res


def iou(box1, box2):
    # box1 and box2 are [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection
    inter_width = max(0, x2 - x1 + 1)
    inter_height = max(0, y2 - y1 + 1)
    inter_area = inter_width * inter_height

    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the union area
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def mergeFigures(figures, iou_threshold=0.5):
    merged = []
    # while figures is not empty
    while figures:
        # take first figure
        current = figures.pop(0)
        # add it to merged
        merged.append(current)

        # filter out figures that overlap with the current one
        figures = [
            fig for fig in figures
            if iou(current.box, fig.box) <= iou_threshold
        ]

    return merged


def translateToFullSheet(figures, offset_x=0, offset_y=0):
    for figure in figures:
        figure.box[0] = figure.box[0] + offset_x
        figure.box[1] = figure.box[1] + offset_y
        figure.box[2] = figure.box[2] + offset_x
        figure.box[3] = figure.box[3] + offset_y
    return figures


def saveFigures(image, figures, output_dir, start_from=0):
    for i, figure in enumerate(figures):
        crop = image.crop((figure.box[0], figure.box[1], figure.box[2], figure.box[3]))

        crop.save(f"{output_dir}figure_{start_from + i + 1}.png", "PNG")


def startFiguresModel(model_dir, num_classes=9):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Ensure it matches the saved model
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))  # Load weights
    model.eval()
    return model


def classifyFigure(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_class = output.max(1)

    # Define class names (Ensure these match your training data)
    class_names = ['double', 'fClef', 'four', 'gClef', 'half', 'one', 'quarter', 'restHalf', 'restOne']

    prediction = class_names[predicted_class.item()]

    return prediction


def classifyFigures(figures, model, image):
    for figure in figures:
        figure_img = image.crop((figure.box[0], figure.box[1], figure.box[2], figure.box[3]))
        prediction = classifyFigure(figure_img, model)
        figure.image = figure_img
        figure.type = prediction

    return figures


def showPredictionsFigures(image, figures):
    imageCopy = image.copy()
    draw = ImageDraw.Draw(imageCopy)

    font = ImageFont.truetype("arial.ttf", 20)

    for figure in figures:
        box = figure.box
        draw.rectangle(box, outline="red", width=1)
        draw.text((box[0], box[1] - 20), figure.type, fill="red", font=font)

        for noteHead in figure.noteHeads:
            x, y = noteHead
            draw.point((x, y), fill="magenta")

    showImage(imageCopy, 'Predicted figures')


def isClef(figure):
    return figure.type in ['gClef', 'fClef']


def isNote(figure):
    return figure.type in ['one', 'double', 'four', 'half', 'quarter']


def isAccidental(figure):
    return figure.type in ['sharp', 'flat']


def isRest(figure):
    return figure.type in ['restOne', 'restDouble', 'restHalf']


def assignObjectTypes(figures):
    for i, figure in enumerate(figures):
        if isClef(figure):
            figures[i] = ClefFigure.fromFigure(figure)
        elif isNote(figure):
            figures[i] = NoteFigure.fromFigure(figure)
        elif isRest(figure):
            figures[i] = RestFigure.fromFigure(figure)
    return figures


def getNoteHeadCenters(figures):
    # assign to each figure its note head centers relatively to the big image
    for i, figure in enumerate(figures):  # Keep track of the index
        x1, y1, _, _ = figure.box

        if figure.type == 'double':
            heads = getNoteHeads(figure.image, 'double')
            figure.noteHeads = [(x + x1, y + y1) for (x, y) in heads]

        elif figure.type == 'four':
            heads = getNoteHeadsFour(figure.image)
            figure.noteHeads = [(x + x1, y + y1) for (x, y) in heads]

        elif figure.type in ['one', 'half', 'quarter']:
            heads = getNoteHeads(figure.image)
            figure.noteHeads = [(x + x1, y + y1) for (x, y) in heads]

    return figures


def detectTemplateFigures(imagePath, figures):
    sharpLocations = extractFigureLocations(imagePath, sharpFigure, 0.6)
    flatLocations = extractFigureLocations(imagePath, flatFigure, 0.6, templateMask_path=flatFigureMask)
    restDoubleLocations = extractFigureLocations(imagePath, restDoubleFigure, 0.8)

    for location in sharpLocations:
        figure = Accidental(location, 'sharp', 1)
        figure.noteHead = (location[0] + figure.width // 2, location[1] + figure.height // 2)
        figures.append(figure)

    for location in flatLocations:
        figure = Accidental(location, 'flat', 1)
        figure.noteHead = (location[0] + figure.width // 2, location[1] + FLAT_FIGURE_HEAD_HEIGHT)
        figures.append(figure)

    for location in restDoubleLocations:
        figure = RestFigure(location, 'restDouble', 1)
        figures.append(figure)

    return figures


def detectMeasureBarLines(imagePath, figures):
    measureBarLines = getMeasureBars(imagePath)

    for line in measureBarLines:
        figure = Figure((line[0] - 5, line[1], line[2] + 5, line[3]), 'bar', 1)
        figures.append(figure)
    return figures


def detectPoints(imagePath, figures):
    points = getPointModifications(imagePath)

    for point in points:
        figure = Dot((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), 'dot', 1)
        figures.append(figure)
    return figures


def distributeFiguresInStaves(figures, staves):
    # assign each figure to the closest stave based on the box center
    for figure in figures:

        _, y = figure.getCenter()

        if figure.type != 'bar':
            closest_stave = min(staves, key=lambda stave: abs(y - stave.getHeightCenter()))
            closest_stave.figures.append(figure)

        else:
            # measure bars will be split in their two staves
            sorted_staves = sorted(staves, key=lambda stave: abs(y - stave.getHeightCenter()))
            top_stave, bottom_stave = sorted_staves[:2]  # get the two closest staves

            # create new figures for the part of the measure bar in that stave
            top_stave.figures.append(
                Figure((figure.box[0], top_stave.topLine - 7, figure.box[2], top_stave.bottomLine + 7), 'bar', 1))
            bottom_stave.figures.append(
                Figure((figure.box[0], bottom_stave.topLine - 7, figure.box[2], bottom_stave.bottomLine + 7), 'bar', 1))

    # sort figures in based on the starting x of the box
    for stv in staves:
        stv.figures.sort(key=lambda fig: fig.getCenter()[0])

    return staves


def showPredictionsStaves(image, staves, labeling='type', coloring=None):
    imageCopy = image.copy()
    draw = ImageDraw.Draw(imageCopy)

    font = ImageFont.truetype("arial.ttf", 20)
    colors = ["red", "blue", "green", "orange"]

    for i, stave in enumerate(staves):

        color = colors[i % len(colors)]

        for figure in stave.figures:

            if coloring == 'type':
                color = classColors[figure.type]

            box = figure.box
            draw.rectangle(box, outline=color, width=1)

            tagText = ''
            if labeling == 'notes':
                if isNote(figure):
                    tagText = "".join(
                        notePitchLabels[note.pitch] + str(note.octave) +
                        (note.accidental if note.accidental != 'n' else '')
                        for note in figure.notes)
                if isAccidental(figure):
                    tagText = notePitchLabels[figure.note.pitch] + str(figure.note.octave)

            elif labeling == 'types':
                tagText = figure.type

            elif labeling == 'duration':
                if isNote(figure):
                    tagText = "".join(
                        str(note.duration) +
                        (figure.articulation if figure.articulation != 'n' else '') + ' '
                        for note in figure.notes)
                if isRest(figure):
                    tagText = str(figure.duration)

            draw.text((box[0], box[1] - 20), tagText, fill=color, font=font)

            if isNote(figure):
                for noteHead in figure.noteHeads:
                    x, y = noteHead
                    draw.point((x, y), fill="magenta")
                    draw.point((x, y - 1), fill="magenta")
                    draw.point((x, y + 1), fill="magenta")
                    draw.point((x - 1, y), fill="magenta")
                    draw.point((x + 1, y), fill="magenta")

    showImage(imageCopy, 'Predictions in staves')


def overlapRatio(box1, box2):
    # evaluates how much of box 1 intersects with box 2
    fig1_x1, fig1_y1, fig1_x2, fig1_y2 = box1
    fig2_x1, fig2_y1, fig2_x2, fig2_y2 = box2

    # calculate intersection area
    overlapWidth = max(0, min(fig1_x2, fig2_x2) - max(fig1_x1, fig2_x1))
    overlapHeight = max(0, min(fig1_y2, fig2_y2) - max(fig1_y1, fig2_y1))
    overlapArea = overlapWidth * overlapHeight

    box1Area = (fig1_x2 - fig1_x1) * (fig1_y2 - fig1_y1)

    return (overlapArea / box1Area) if box1Area > 0 else 0


def isPartOfSignature(figure, stave):
    figuresToLeft = [
        fig2 for fig2 in stave.figures
        if fig2.getCenter()[0] < figure.getCenter()[0]  # fig2 is to the left
    ]

    figuresToLeft.sort(key=lambda fig: fig.getCenter()[0], reverse=True)  # order figures from right to left

    for fig2 in figuresToLeft:
        if isClef(fig2):
            return True
        if isNote(fig2):
            return False


def handleCorrections(staves):
    for stave in staves:
        if stave.staveIndex == 0:
            # filter out the bpm and rhythm figures if any on top of stave 1
            stave.figures[:] = [
                fig2 for fig2 in stave.figures
                if abs(fig2.getCenter()[1] - stave.getHeightCenter()) < 100
            ]

        for figure in stave.figures:

            # sometimes key signature is detected by the fastRCNN as other figure types
            if isAccidental(figure):

                # check if the accidental is part of the key signature
                isSignature = isPartOfSignature(figure, stave)

                if isSignature:
                    # filter out figures that overlap too much with the key signature accidentals
                    stave.figures[:] = [
                        fig2 for fig2 in stave.figures
                        if fig2.type == figure.type
                           or overlapRatio(figure.box, fig2.box) < 0.5
                    ]

            # fClef figures has to dots that might be detected as 'dot' figures, filter those out
            if figure.type == 'fClef':
                stave.figures[:] = [
                    fig2 for fig2 in stave.figures
                    if fig2.type != 'dot'  # filter out dots
                       or fig2.getCenter()[0] < figure.getCenter()[0]  # that are to the right of the fClef
                       or overlapRatio(fig2.box, figure.box) < 0.5  # where dot area overlaps more than 0.5 with fClef
                ]
    return staves


def mapNoteFromLine(n, pitchOffset=0, octaveOffset=4):
    #                             line           |   n   ->  line in the half step scale
    #                             ....           | ...
    #                              --            |  15
    #                |--           --            |  13   ->  A in gClef (reference point with pitchOffset=6 in the octaveOffset=5)
    #                |    ---------------------  |  11
    #               /|    ---------------------  |   9
    #  stave lines <||    ---------------------  |   7
    #               \|    ---------------------  |   5
    #                |    ---------------------  |   3
    #                |--           --            |   1   ->  C in gClef (reference point with pitchOffset=0 and octaveOffset=4)
    #                              --            |  -1
    #                             ....           | ...

    n = n + pitchOffset

    if n < 0:
        pitch = (7 + (n % 7)) % 7 or 7  # if 0
        octave = n // 7  # decreases every 7 steps
    else:
        pitch = (n - 1) % 7 + 1  # pitch is integers from 1 to 7
        octave = (n - 1) // 7  # increments every 7 steps

    return Note(int(pitch), int(octave) + octaveOffset)  # return a note object


def getClef(figure, stave):
    # given a figure in a stave looks for the closest clef to its left
    # returns the clef type

    clefsToLeft = [
        fig for fig in stave.figures
        if isClef(fig)
           and fig.getCenter()[0] < figure.getCenter()[0]
    ]  # list of clefs to the left

    # get the closest clef
    closestClef = max(clefsToLeft, key=lambda clef: clef.getCenter()[0])

    return closestClef


def getNote(figureHead_y, staveLines, clefType, meanGap):
    # find the closest lines above and below
    # lower line is the minimum of the lines below
    # upper line is the maximum of lines above
    lower_line = min([line for line in staveLines if line > figureHead_y], default=None)
    upper_line = max([line for line in staveLines if line <= figureHead_y], default=None)

    # note head is inside the stave
    if lower_line is not None and upper_line is not None:
        # compute the midline (notes can be in the space between lines)
        midline = (lower_line + upper_line) / 2
        closestLine = min([lower_line, upper_line, midline], key=lambda x: abs(x - figureHead_y))

        if closestLine == midline:
            # line from gClef C. (C is line 1, D is 1.5, E is 2...)
            lineFromC = staveLines.index(lower_line) + 1 + 0.5

            # convert to half steps scale (assuming gClef):
            # 1.0 -> 1(C), 1.5 -> 2(D), 2 -> 3(E), 2.5 -> 4(F), 4.5 -> 8(C in the next octave)...
            halfStepFromC = int(2 * lineFromC - 1)

            if clefType == 'gClef':
                note = mapNoteFromLine(halfStepFromC)
            elif clefType == 'fClef':
                # gClef C is equivalent to E in octave 2 in fClef
                note = mapNoteFromLine(halfStepFromC, 2, 2)

        else:
            # line from gClef C. (C is line 1)
            lineFromC = staveLines.index(closestLine) + 1

            # convert to half steps scale (assuming gClef):
            # 1.0 -> 1(C), 1.5 -> 2(D), 2 -> 3(E), 2.5 -> 4(F)...
            halfStepFromC = 2 * lineFromC - 1

            if clefType == 'gClef':
                note = mapNoteFromLine(halfStepFromC)
            elif clefType == 'fClef':
                # gClef C is equivalent to E in octave 2 in fClef
                note = mapNoteFromLine(halfStepFromC, 2, 2)

    # head is below the stave
    elif lower_line is None:
        # head is still in the C line, just 1 or 2 pixels below (assuming gClef)
        if abs(upper_line - figureHead_y) < meanGap // 4:
            if clefType == 'gClef':
                note = Note(1, 4)  # C is pitch 1 and octave 4 in gClef (C4)
            elif clefType == 'fClef':
                note = Note(3, 2)  # E2 is the equivalent in fClef to C4 in gClef
        else:
            dist = abs(figureHead_y - upper_line)
            halfStepsBelow = round(dist / (meanGap / 2))  # count half-steps
            # line from gClef C
            lineFromC = -0.5 * halfStepsBelow + 1
            # half step from gClef C
            halfStepFromC = 2 * lineFromC - 1

            if clefType == 'gClef':
                note = mapNoteFromLine(halfStepFromC)
            elif clefType == 'fClef':
                # gClef C is equivalent to E in octave 2 in fClef
                note = mapNoteFromLine(halfStepFromC, 2, 2)

    # head is above the stave
    elif upper_line is None:
        # head is still in the A line, just 1 or 2 pixels above (assuming gClef)
        if abs(lower_line - figureHead_y) < meanGap // 4:
            if clefType == 'gClef':
                note = Note(6, 5)  # A is pitch 6, octave 5 in gClef (A5)
            elif clefType == 'fClef':
                note = Note(1, 4)  # C4 is the equivalent in fClef to A5 in gClef
        else:
            dist = abs(lower_line - figureHead_y)  # distance from A
            halfStepsAbove = round(dist / (meanGap / 2))  # count half-steps
            # line from gClef A
            lineFromA = 0.5 * halfStepsAbove
            # half step scale from gClef A
            halfStepFromA = 2 * lineFromA

            if clefType == 'gClef':
                note = mapNoteFromLine(halfStepFromA, 6, 5)  # starting point is A5 in gClef scale (A is note 6)
            elif clefType == 'fClef':
                note = mapNoteFromLine(halfStepFromA, 1, 4)  # C4 in fClef is equivalent to A5 in gClef
    return note


def assignNotes(staves):
    for stave in staves:
        # bottom line which has the highest value in y must be index 0
        staveLines = sorted(stave.lineHeights, reverse=True)
        for figure in stave.figures:
            if isNote(figure):

                clefType = getClef(figure, stave).type  # obtain clef for later pitch assignation

                for (figureHead_x, figureHead_y) in figure.noteHeads:
                    note = getNote(figureHead_y, staveLines, clefType, stave.meanGap)

                    note.noteHead = (figureHead_x, figureHead_y)
                    figure.notes.append(note)

                # order the notes increasingly with octave as the first criteria
                figure.notes.sort(key=lambda noteObj: (noteObj.octave, noteObj.pitch))

            if isAccidental(figure):
                clefType = getClef(figure, stave).type  # obtain clef for later pitch assignation

                note = getNote(figure.noteHead[1], staveLines, clefType, stave.meanGap)
                note.noteHead = (figure.noteHead[0], figure.noteHead[1])
                figure.note = note

    return staves


def getKeySignatures(staves):
    for stave in staves:
        previousSignature = ['n', 'n', 'n', 'n', 'n', 'n', 'n']  # Default all naturals
        for i, figure in enumerate(stave.figures):
            if isClef(figure):

                # all naturals by default (index 0 -> C, index 6 -> B)
                signatureAccidentals = ['n', 'n', 'n', 'n', 'n', 'n', 'n']

                j = i + 1  # Start from the first figure to the right

                hasSignature = False
                while j < len(stave.figures) and isAccidental(stave.figures[j]):
                    hasSignature = True
                    accNote = stave.figures[j].note.pitch
                    stave.figures[j].isSignature = True  # mark that the accidental is part of the signature

                    if stave.figures[j].type == 'sharp':
                        signatureAccidentals[accNote - 1] = '#'  # Modify based on pitch
                    elif stave.figures[j].type == 'flat':
                        signatureAccidentals[accNote - 1] = 'b'

                    j += 1

                # no signature found -> inherit from previous
                if not hasSignature:
                    signatureAccidentals = previousSignature.copy()

                figure.signature = signatureAccidentals
                previousSignature = signatureAccidentals  # update the new previous

    return staves


def applyKeySignature(staves):
    for stave in staves:
        for figure in stave.figures:

            if isNote(figure):

                clefSignature = getClef(figure, stave).signature

                for note in figure.notes:
                    note.accidental = clefSignature[note.pitch - 1]

    return staves


def applyAccidentals(staves):
    for stave in staves:
        for i, figure in enumerate(stave.figures):
            if isAccidental(figure) and not figure.isSignature:
                # figures to the right
                for j in range(i + 1, len(stave.figures)):
                    next_figure = stave.figures[j]

                    if isNote(next_figure):
                        for note in next_figure.notes:
                            if note.pitch == figure.note.pitch:
                                if figure.type == 'flat':
                                    note.accidental = 'b'
                                elif figure.type == 'sharp':
                                    note.accidental = '#'

                    # accidentals only apply to the measure if not in signature
                    if next_figure.type == 'bar':
                        break

    return staves


def assignNoteDurations(staves):
    for stave in staves:
        for figure in stave.figures:
            if isNote(figure):
                if len(figure.notes) > 0:
                    for note in figure.notes:
                        note.duration = noteDurations[figure.type]
                    # take the max of the notes, will be used for adjusting measures to beat
                    figure.duration = max([note.duration for note in figure.notes])
                else:
                    # note heads not detected
                    figure.duration = noteDurations[figure.type]

            if isRest(figure):
                figure.duration = noteDurations[figure.type]
    return staves


def doDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def doAngle(p1, p2):
    # assumed center of the circle is p1
    # assumed p2 is in the circumference line
    #        90
    #         |
    #  180 -- · -- 0
    #         |
    #        -90

    # y-axis is inverted because y values in python are inverted
    angle = math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0]) * (180 / math.pi)
    return angle


def getBestDot(point, stave, maxDist=50):
    # dot must be over the note head
    #      .
    #  <0>-.  -> here is very exaggerated but this point is the closest
    #  <0>--     for both notes and the other point will be unused and later removed
    #  |----
    #  |----
    #

    dots = [dot for dot in stave.figures
            if dot.type == 'dot'
            and doDistance(dot.getCenter(), point) <= maxDist
            and (-10 < doAngle(point, dot.getCenter()) < 120 or -120 < doAngle(point, dot.getCenter()) < -60)]

    if len(dots) > 0:  # 301, 1551, 315, 1565
        closestDot = min(dots, key=lambda d: doDistance(d.getCenter(), point))
    else:
        closestDot = None

    return closestDot


def applyDots(staves):
    for stave in staves:
        for figure in stave.figures:
            if isNote(figure):
                for note in figure.notes:
                    bestDot = getBestDot(note.noteHead, stave, 50)
                    if bestDot is not None:
                        angle = doAngle(note.noteHead, bestDot.getCenter())
                        if -45 < angle < 50:
                            # point is to the right -> extend duration
                            note.duration += note.duration * 0.5
                            bestDot.used = True
                        elif 60 < angle < 120 or -120 < angle < -60:
                            # point is on top or below -> staccato articulation
                            figure.articulation = 's'
                            bestDot.used = True

                # update global duration
                figure.duration = max([note.duration for note in figure.notes])

            if isRest(figure):
                bestDot = getBestDot(figure.getCenter(), stave, 60)
                if bestDot is not None:
                    angle = doAngle(figure.getCenter(), bestDot.getCenter())
                    if -50 < angle < 50:
                        figure.duration = figure.duration + figure.duration * 0.5
                        bestDot.used = True

    # remove unused dots, they might be points on 'i' letters or other random places
    for stave in staves:
        stave.figures = [figure for figure in stave.figures if not (figure.type == 'dot' and not figure.used)]

    return staves


def convertToMeasures(staves):
    measures = []

    # distribute figures in measures
    for staveIndex, stave in enumerate(staves):
        currentMeasureFigures = []
        for figure in stave.figures:
            if figure.type == 'bar':
                if currentMeasureFigures:  # add if there are figures in the measure
                    measures.append(Measure(staveIndex, currentMeasureFigures))
                currentMeasureFigures = []
            elif isNote(figure):
                currentMeasureFigures.append(figure)
            elif isRest(figure):
                currentMeasureFigures.append(figure)

        # get last measure if it doesn't end with a bar line
        if currentMeasureFigures:
            measures.append(Measure(staveIndex, currentMeasureFigures))

    # Find the most common measure duration
    durations = [measure.duration for measure in measures]
    measureDuration = max(set(durations), key=durations.count)

    return measures, measureDuration


def adjustMeasuresToBeat(measures, realBeats):
    for measure in measures:

        beatDiff = measure.duration - realBeats

        if beatDiff != 0:
            if beatDiff > 0:  # more duration than expected -> cut down
                # decrease duration
                while beatDiff > 0 and measure.figures:
                    last_fig = measure.figures[-1]
                    if last_fig.duration > beatDiff:
                        last_fig.duration -= beatDiff
                        beatDiff = 0
                        if isNote(last_fig):
                            for note in last_fig.notes:
                                note.duration -= beatDiff
                                if note.duration < 0:
                                    note.duration = 0
                    else:
                        beatDiff -= last_fig.duration
                        measure.figures.pop()

            elif beatDiff < 0:  # less duration than expected -> extend
                measure.figures[-1].duration += -beatDiff

    return measures


def showPredictionMeasures(image, measures):
    imageCopy = image.copy()
    draw = ImageDraw.Draw(imageCopy)

    font = ImageFont.truetype("arial.ttf", 20)

    colors = ["red", "blue", "green", "orange"]

    for mi, measure in enumerate(measures):
        color = colors[mi % len(colors)]
        for fi, figure in enumerate(measure.figures):
            box = figure.box
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], box[1] - 20), f"{mi + 1}, {fi + 1}: {figure.duration}", fill=color, font=font)

    showImage(imageCopy, 'Predictions Measures')

    return


def createSong(measures, beats, bpm):
    song = Song([], [], beats, bpm)

    for measure in measures:
        startPulse = 0
        for figure in measure.figures:

            multiSound = MultiSound([], startPulse, figure.duration)

            if isNote(figure):
                if len(figure.notes) > 0:
                    for note in figure.notes:
                        # turn flats to their sharp equivalents
                        if note.accidental == 'b':
                            if note.pitch == 1:
                                sound = notePitchLabels[7] + str(note.octave - 1)
                            elif note.pitch == 4:
                                sound = notePitchLabels[3] + str(note.octave)
                            else:
                                sound = notePitchLabels[note.pitch - 1] + str(note.octave) + '#'
                        elif note.accidental == '#':
                            if note.pitch == 7:
                                sound = notePitchLabels[1] + str(note.octave + 1)
                            elif note.pitch == 3:
                                sound = notePitchLabels[4] + str(note.octave)
                            else:
                                sound = notePitchLabels[note.pitch] + str(note.octave) + '#'
                        else:  # natural note
                            sound = notePitchLabels[note.pitch] + str(note.octave)

                        soundDTO = SoundDTO(sound, note.duration)
                        multiSound.sounds.append(soundDTO)
                else:
                    # no noteHeads in the figure
                    soundDTO = SoundDTO('rest', figure.duration)
                    multiSound.sounds.append(soundDTO)

            elif isRest(figure):
                soundDTO = SoundDTO('rest', figure.duration)
                multiSound.sounds.append(soundDTO)

            # assign track
            if measure.staveIndex % 2 == 0:
                song.upperTrack.append(multiSound)
            else:
                song.lowerTrack.append(multiSound)

            startPulse += figure.duration

    return song
