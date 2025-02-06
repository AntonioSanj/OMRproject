import cv2
import numpy as np

from utils.plotUtils import showImage
from vision.figureDetection.figureDetectionUtils import filterClosePoints
from vision.imageUtils import loadImageGrey
from constants import *


def createKernel(image_path):
    img, imgGrey = loadImageGrey(image_path)  # Load the image in grayscale
    _, kernel = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return kernel


def extractFigureLocations(image_path, figure_path, threshold=0.7, show=False, print_points=False):
    img, imgGrey = loadImageGrey(image_path)
    _, binary = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = createKernel(figure_path)

    result = cv2.matchTemplate(binary, kernel, cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= threshold)  # (y_coords, x_coords)

    # Convert to list of (x, y) coordinates
    points = list(zip(locations[1], locations[0]))

    points = filterClosePoints(points, 5)

    if print_points:
        print(f'{len(points)} FIGURES FOUND:\n{points}')

    if show:
        for point in points:
            cv2.rectangle(binary, point, (point[0] + kernel.shape[1], point[1] + kernel.shape[0]), 255, 2)
        showImage(binary)
    return


extractFigureLocations(fullsheetsDir + 'this_is_me1.png', sharpFigure, 0.65, True, True)
extractFigureLocations(fullsheetsDir + 'walk_on_by1.png', flatFigure, 0.7, True, True)
