import math

import torch
from PIL import ImageDraw
from matplotlib import pyplot as plt

from constants import SLICE_HEIGHT
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


def showPredictions(slicedImage, figures):
    slicedImageCopy = slicedImage.copy()
    draw = ImageDraw.Draw(slicedImageCopy)

    for figure in figures:
        box = figure.box
        draw.rectangle(box, outline="red", width=1)

    # Display using matplotlib
    plt.imshow(slicedImageCopy)
    plt.axis('off')  # Turn off the axis for better visibility
    plt.show()

