import math

import torch
from PIL import ImageDraw
from matplotlib import pyplot as plt

from constants import SLICE_HEIGHT, SLICE_WIDTH
from learning.FasterRCNN.getModel import get_model
from torchvision.transforms import functional as F

from objectTypes.Figure import Figure


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
        if figure.box[0] > borderSeparation and figure.box[2] < SLICE_WIDTH - borderSeparation:
            res.append(figure)
    return res


def showPredictions(image, figures):
    imageCopy = image.copy()
    draw = ImageDraw.Draw(imageCopy)

    for figure in figures:
        box = figure.box
        draw.rectangle(box, outline="red", width=1)

    # Display using matplotlib
    plt.figure(figsize=(16, 10))
    plt.imshow(imageCopy)
    plt.axis('off')  # Turn off the axis for better visibility
    plt.show()


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
