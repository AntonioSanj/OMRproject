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


def showPredictions(image, figures):
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

    # Display using matplotlib
    plt.figure(figsize=(10, 16))
    plt.imshow(imageCopy)
    plt.axis('off')  # Turn off the axis for better visibility
    plt.show()


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
        figures.append(figure)

    for location in flatLocations:
        figure = Figure(location, 'flat', 1)
        figures.append(figure)

    for location in restDoubleLocations:
        figure = Figure(location, 'restDouble', 1)
        figures.append(figure)

    return figures


def distributeFiguresInStaves(figures, staves):
    return
