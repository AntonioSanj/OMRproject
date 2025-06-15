import cv2
import numpy as np

from constants import *
from utils.plotUtils import showImage
from vision.figureDetection.figureDetectionUtils import filterClosePoints
from vision.visionUtils import loadImageGrey


def getPointModifications(image_path, show=False, print_points=False):
    img, imgGrey = loadImageGrey(image_path)
    _, binary = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED)

    threshold = 0.7
    locations = np.where(result >= threshold)  # (y_coords, x_coords)

    # Convert to list of (x, y) coordinates
    points = [(x, y, kernel.shape[1], kernel.shape[0]) for x, y in zip(locations[1], locations[0])]

    points = filterClosePoints(points, 5)

    boxLocations = []
    for x, y, w, h in points:
        box = (x, y, x + w, y + h)
        boxLocations.append(box)

    # points = [(x + kernel.shape[1] // 2, y + kernel.shape[0] // 2) for (x, y) in points]

    if print_points:
        print(f'{len(points)} POINTS FOUND:\n{points}')

    if show:
        for (x, y, w, h) in points:
            cv2.rectangle(binary, (x, y), (x + w, y + h), 255, 2)
        showImage(binary)
    return boxLocations



