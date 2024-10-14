import cv2
import numpy as np

from constants import IMAGE_SIZE, WHITE, IMAGE_CENTER


def loadImageGrey(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


def thresh(grayImage, threshold):
    _, umbralized_image = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY)
    return umbralized_image


def cannyEdges(image, dilate_kernel_radius):
    # compute edges
    edges = cv2.Canny(image, 50, 200, apertureSize=3)

    if dilate_kernel_radius != 0:
        # dilate canny edges for easing line detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_radius, dilate_kernel_radius))
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def rotateAdjustImage(edges, original):
    # detect lines of pentagram
    minLineLength = 1000
    maxLineGap = 50
    angle = np.pi / 180
    lines = cv2.HoughLinesP(edges, 1, angle, threshold=10, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # store line angles
    angles = []

    if lines is not None:
        for line in lines:
            # obtain line points
            x1, y1, x2, y2 = line[0]

            # compute angle
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = 0
            angle = np.arctan(slope) * (180.0 / np.pi)
            # store angles
            angles.append(angle)

    # rotate original image by the mean angle
    matrix = cv2.getRotationMatrix2D(IMAGE_CENTER, np.mean(angles), 1.0)
    rotated_image = cv2.warpAffine(original, matrix, (IMAGE_SIZE[0], IMAGE_SIZE[1]), borderValue=WHITE)

    return rotated_image


def adjustImageSize(img):
    resized_image = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    return resized_image


def getHorizontalLines(draw_image, edges, threshold, minLineLength, maxLineGap):
    angle = np.pi / 180
    lines = cv2.HoughLinesP(edges, 1, angle, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # draw obtained lines
    linesImage = draw_image.copy()

    if lines is not None:
        for line in lines:
            # obtain line points
            x1, y1, x2, y2 = line[0]
            # draw line
            cv2.line(linesImage, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return linesImage, lines


def getLineHeights(lines, minGap):
    heights = []
    for line in lines:
        # obtain line height points
        _, y1, _, y2 = line[0]
        if y1 == y2:
            heights.append(y1)

    # order heights
    heights.sort()

    lineHeights = []
    for i in range(len(heights) - 1):
        gap = heights[i + 1] - heights[i]
        if gap > minGap:
            lineHeights.append(heights[i])

    # add last line
    lineHeights.append(heights[len(heights)-1])
    print(lineHeights)
    return lineHeights
