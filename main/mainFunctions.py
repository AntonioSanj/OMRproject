import math
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional as F
from constants import *
from learning.FasterRCNN.getModel import get_model
from objectTypes.Figure import Figure
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


def getNoteHeadCenters(figures):
    # assign to each figure its note head centers relatively to the big image
    for figure in figures:
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
    sharpLocations = extractFigureLocations(imagePath, sharpFigure, 0.65)
    flatLocations = extractFigureLocations(imagePath, flatFigure, 0.6, templateMask_path=flatFigureMask)
    restDoubleLocations = extractFigureLocations(imagePath, restDoubleFigure, 0.8)

    for location in sharpLocations:
        figure = Figure(location, 'sharp', 1)
        figure.noteHeads = [(location[0] + figure.width // 2, location[1] + figure.height // 2)]
        figures.append(figure)

    for location in flatLocations:
        figure = Figure(location, 'flat', 1)
        figure.noteHeads = [(location[0] + figure.width // 2, location[1] + FLAT_FIGURE_HEAD_HEIGHT)]
        figures.append(figure)

    for location in restDoubleLocations:
        figure = Figure(location, 'restDouble', 1)
        figure.noteHeads = [(location[0] + figure.width // 2, location[1] + figure.height // 2)]
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
        figure = Figure((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), 'dot', 1)
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
        stv.figures.sort(key=lambda fig: fig.box[0])

    return staves


def showPredictionsStaves(image, staves, notes=False):
    imageCopy = image.copy()
    draw = ImageDraw.Draw(imageCopy)

    font = ImageFont.truetype("arial.ttf", 20)

    colors = ["red", "blue", "green", "orange"]

    for i, stave in enumerate(staves):

        color = colors[i % len(colors)]

        for figure in stave.figures:
            box = figure.box
            draw.rectangle(box, outline=color, width=1)

            if notes:
                tagText = "".join(noteLabels[pitch] + str(octave) for (pitch, octave) in figure.notes)
            else:
                tagText = figure.type

            draw.text((box[0], box[1] - 20), tagText, fill=color, font=font)

            for noteHead in figure.noteHeads:
                x, y = noteHead
                draw.point((x, y), fill="magenta")

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


def existsFigureAtTheLeft(figure, stave, types, distanceToSearch=999999):
    # checks if figure <figure> has in the stave <stave> any other figures of type in <types>
    # within a distance to the left of <distanceToSearch>
    return any(
        fig2.type in types and  # fig2 is the searched type
        fig2.getCenter()[0] <= figure.getCenter()[0] and  # fig2 is to the left
        (figure.getCenter()[0] - fig2.getCenter()[0]) <= distanceToSearch  # fig2 is inside the distance threshold
        for fig2 in stave.figures
    )


def handleCorrections(staves):
    for stave in staves:
        for figure in stave.figures:
            # sometimes clef armor is detected by the fastRCNN
            if figure.type == 'flat':

                # check if there is a gClef or fClef to the left in a distance
                clefToTheLeft = existsFigureAtTheLeft(figure, stave, ['gClef', 'fClef'], 150)

                if clefToTheLeft:
                    # filter out figures that overlap too much with the flat figure
                    stave.figures[:] = [
                        fig for fig in stave.figures
                        if fig is figure or overlapRatio(figure.box, fig.box) < 0.9
                    ]
    return staves


def mapNote(n, pitchOffset=0, octaveOffset=4):
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
        octave = int(n // 7)  # decreases every 7 steps
    else:
        pitch = (n - 1) % 7 + 1  # pitch is integers from 1 to 7
        octave = int((n - 1) // 7)  # increments every 7 steps

    return pitch, octave + octaveOffset


def getClef(figure, stave):
    # given a figure in a stave looks for the closest clef to its left
    # returns the clef type

    clefsToLeft = [
        fig for fig in stave.figures
        if fig.type in ['gClef', 'fClef'] and
        fig.getCenter()[0] < figure.getCenter()[0]
    ]  # list of clefs to the left

    if len(clefsToLeft) == 0:
        return 'gClef'  # if no clefs found return gClef (most common)

    # get the closest clef
    closestClef = max(clefsToLeft, key=lambda clef: clef.getCenter()[0])

    return closestClef.type


def assignNotes(staves):
    for stave in staves:
        # bottom line which has the highest value in y must be index 0
        staveLines = sorted(stave.lineHeights, reverse=True)
        for figure in stave.figures:
            if figure.type in ['double', 'one', 'half', 'quarter', 'four', 'sharp', 'flat']:

                clef = getClef(figure, stave)  # obtain clef for later pitch assignation

                for (_, figureHead_y) in figure.noteHeads:

                    # find the closest lines above and below
                    # lower line is the minimum of the lines below
                    # upper line is the maximum of lines above
                    lower_line = min([line for line in staveLines if line >= figureHead_y], default=None)
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

                            if clef == 'gClef':
                                note = mapNote(halfStepFromC)
                            elif clef == 'fClef':
                                # gClef C is equivalent to E in octave 2 in fClef
                                note = mapNote(halfStepFromC, 2, 2)

                        else:
                            # line from gClef C. (C is line 1)
                            lineFromC = staveLines.index(closestLine) + 1

                            # convert to half steps scale (assuming gClef):
                            # 1.0 -> 1(C), 1.5 -> 2(D), 2 -> 3(E), 2.5 -> 4(F)...
                            halfStepFromC = 2 * lineFromC - 1

                            if clef == 'gClef':
                                note = mapNote(halfStepFromC)
                            elif clef == 'fClef':
                                # gClef C is equivalent to E in octave 2 in fClef
                                note = mapNote(halfStepFromC, 2, 2)

                    # head is below the stave
                    elif lower_line is None:
                        # head is still in the C line, just 1 or 2 pixels below (assuming gClef)
                        if abs(upper_line - figureHead_y) < stave.meanGap // 4:
                            if clef == 'gClef':
                                note = 1, 4  # C is pitch 1 and octave 4 in gClef (C4)
                            elif clef == 'fClef':
                                note = 3, 2  # E2 is the equivalent in fClef to C4 in gClef
                        else:
                            dist = abs(figureHead_y - upper_line)
                            halfStepsBelow = round(dist / (stave.meanGap / 2))  # count half-steps
                            # line from gClef C
                            lineFromC = -0.5 * halfStepsBelow + 1
                            # half step from gClef C
                            halfStepFromC = 2 * lineFromC - 1

                            if clef == 'gClef':
                                note = mapNote(halfStepFromC)
                            elif clef == 'fClef':
                                # gClef C is equivalent to E in octave 2 in fClef
                                note = mapNote(halfStepFromC, 2, 2)

                    # head is above the stave
                    elif upper_line is None:
                        # head is still in the A line, just 1 or 2 pixels above (assuming gClef)
                        if abs(lower_line - figureHead_y) < stave.meanGap // 4:
                            if clef == 'gClef':
                                note = 6, 5  # A is pitch 6, octave 5 in gClef (A5)
                            elif clef == 'fClef':
                                note = 1, 4  # C4 is the equivalent in fClef to A5 in gClef
                        else:
                            dist = abs(lower_line - figureHead_y)  # distance from A
                            halfStepsAbove = round(dist / (stave.meanGap / 2))  # count half-steps
                            # line from gClef A
                            lineFromA = 0.5 * halfStepsAbove
                            # half step scale from gClef A
                            halfStepFromA = 2 * lineFromA

                            if clef == 'gClef':
                                note = mapNote(halfStepFromA, 6, 5)  # starting point is A5 in gClef scale (A is note 6)
                            elif clef == 'fClef':
                                note = mapNote(halfStepFromA, 1, 4)  # C4 in fClef is equivalent to A5 in gClef

                    figure.notes.append(note)

                # order the notes increasingly with octave as the first criteria
                figure.notes.sort(key=lambda noteTuple: (noteTuple[1], noteTuple[0]))

    return staves
