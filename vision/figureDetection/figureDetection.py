import os

import cv2
import numpy as np

from utils.plotUtils import showImage
from vision.figureDetection.figureDetectionUtils import filterClosePoints
from vision.visionUtils import loadImageGrey, createKernelFromImage
from constants import *


def extractFigureLocations(image_path, templatesDir, threshold=0.7, templateMask_path=None, show=False,
                           print_points=False):
    img, imgGrey = loadImageGrey(image_path)
    _, binary = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    points = []

    for templateName in os.listdir(templatesDir):
        template = os.path.join(templatesDir, templateName)
        kernel = createKernelFromImage(template)

        if templateMask_path is None:
            result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED)
        else:
            kernelMask = createKernelFromImage(templateMask_path)
            result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED, mask=kernelMask)

        locations = np.where(result >= threshold)  # (y_coords, x_coords)

        # Convert to list of (x, y) coordinates
        for x, y in zip(locations[1], locations[0]):
            points.append((x, y, kernel.shape[1], kernel.shape[0]))

    points = filterClosePoints(points, 10)

    boxLocations = []
    for x, y, w, h in points:
        box = (x, y, x + w, y + h)
        boxLocations.append(box)

    if print_points:
        print(f'{len(points)} FIGURES FOUND:\n{boxLocations}')

    if show:
        for box in boxLocations:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        showImage(img)

    return boxLocations
