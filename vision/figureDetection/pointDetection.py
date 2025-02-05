import cv2
import numpy as np

from utils.plotUtils import showImage
from vision.imageUtils import loadImageGrey, thresh
from constants import *


def getPointModifications(image_path, meanGap):
    img, imgGrey = loadImageGrey(image_path)
    _, binary = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    showImage(binary)

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

    # Define threshold for matches
    threshold = 0.8  # Adjust based on your needs
    locations = np.where(result >= threshold)
    # Draw rectangles on the matches
    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    w, h = kernel.shape[::-1]

    for pt in zip(*locations[::-1]):  # Swap x and y
        cv2.rectangle(img_color, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)

    showImage(img_color, "Matched Points")
    return


getPointModifications(myDataImg + '/image_13.png', 10)
