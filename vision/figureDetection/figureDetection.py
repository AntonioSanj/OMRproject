import cv2
import numpy as np

from utils.plotUtils import showImage
from vision.figureDetection.figureDetectionUtils import filterClosePoints
from vision.visionUtils import loadImageGrey, createKernelFromImage
from constants import *


def extractFigureLocations(image_path, figure_path, threshold=0.7, templateMask_path=None, show=False, print_points=False):

    img, imgGrey = loadImageGrey(image_path)
    _, binary = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = createKernelFromImage(figure_path)

    if templateMask_path is None:
        result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED)
    else:
        kernelMask = createKernelFromImage(templateMask_path)
        result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED, mask=kernelMask)

        showImage(result)

    locations = np.where(result >= threshold)  # (y_coords, x_coords)

    # Convert to list of (x, y) coordinates
    points = list(zip(locations[1], locations[0]))

    points = filterClosePoints(points, 5)

    boxLocations = []
    for point in points:
        box = (point[0], point[1], point[0] + kernel.shape[1], point[1] + kernel.shape[0])
        boxLocations.append(box)

    if show:
        for box in boxLocations:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        showImage(img)

    if print_points:
        print(f'{len(points)} FIGURES FOUND:\n{boxLocations}')

    return boxLocations
