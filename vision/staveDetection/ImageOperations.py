import cv2
import numpy as np

from constants import IMAGE_WIDTH, IMAGE_HEIGHT, WHITE, IMAGE_CENTER, COLORS, RED

from objectTypes.Stave import Stave
from utils.plotUtils import showImage


def loadImageGrey(image_path):
    # load image
    img = cv2.imread(image_path)

    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


def thresh(grayImage, threshold):
    # apply threshold to image
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


def rotateAdjustImage(edges, original, showLines=False):
    # detect lines of pentagram
    minLineLength = 1000
    maxLineGap = 50
    angle = np.pi / 180
    lines = cv2.HoughLinesP(edges, 1, angle, threshold=10, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if showLines:
        lineShow = original.copy()
        for line in lines:
            # obtain line points
            x1, y1, x2, y2 = line[0]
            cv2.line(lineShow, (x1, y1), (x2, y2), (255, 0, 0), 2)

        showImage(lineShow, 'All lines detected')
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
            # remove vertical lines
            if 45 > angle > -45:
                # store angles
                angles.append(angle)

    # rotate original image by the mean angle
    matrix = cv2.getRotationMatrix2D(IMAGE_CENTER, np.mean(angles), 1.0)
    rotated_image = cv2.warpAffine(original, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT), borderValue=WHITE)

    return rotated_image


def adjustImageSize(img):
    resized_image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return resized_image


def getHorizontalLines(edges, og, threshold, minLineLength, maxLineGap):
    angle = np.pi / 180
    lines = cv2.HoughLinesP(edges, 1, angle, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    og2 = og.copy()
    for line in lines:
        # obtain line points
        x1, y1, x2, y2 = line[0]
        cv2.line(og2, (x1, y1), (x2, y2), RED, 1)

    return lines, og2


def chooseCandidate(candidates, op="min"):
    n = len(candidates)
    value = 0
    if op == "min":
        value = min(candidates)
    elif op == "max":
        value = max(candidates)
    elif op == "med":
        if n % 2 == 1:  # Odd length
            value = candidates[n // 2]
        else:  # even length
            value = candidates[n // 2 - 1]
    elif op == "mean":
        value = round(sum(candidates) / len(candidates))
    return value


def getLineHeights(lines, minGap):
    # lines:    array of line heights
    # minGap:   minimum gap between lines in the returning array
    # trimMeanFactor:   when computing the mean of gaps how many elements are
    #                   removed from each side of the sorted gap array list

    # gap above the actual first stave line and gap below actual last stave line must be computed
    heights = [0, IMAGE_HEIGHT]

    for line in lines:
        # obtain line height points
        _, y1, _, y2 = line[0]
        if y1 == y2:
            heights.append(y1)

    # order heights
    heights.sort()

    lineHeights = []
    gapList = []
    candidates = []

    # select only lines distanced enough
    for i in range(len(heights) - 1):
        candidates.append(heights[i])
        gap = heights[i + 1] - heights[i]
        if gap > minGap:
            lineHeights.append(chooseCandidate(candidates))
            candidates = []

    # add last line
    lineHeights.append(heights[len(heights) - 1])

    # compute gaps of the remaining lines
    for i in range(len(lineHeights) - 1):
        gap = lineHeights[i + 1] - lineHeights[i]
        gapList.append(gap)

    # lineHeight[i] has a gap above of gapList[i-1] and a gap below of gapList[i]
    # print("(", len(lineHeights), ") ", "LineHeights: ", lineHeights)
    # print("(", len(gapList), ") ", "gapList: ", gapList)

    meanGap = np.bincount(gapList).argmax()  # most repeated value

    return lineHeights, meanGap


def isGapAboveMeanGap(lineIndex, lineHeights, meanGap, tolerance):
    # is the gap above the current line the size of meanGap?
    return abs(abs(lineHeights[lineIndex - 1] - lineHeights[lineIndex]) - meanGap) <= tolerance


def isGapBelowMeanGap(lineIndex, lineHeights, meanGap, tolerance):
    # is the gap below the current line the size of meanGap?
    return abs(abs(lineHeights[lineIndex + 1] - lineHeights[lineIndex]) - meanGap) <= tolerance


def consolidateLines(lineHeights, meanGap, tolerance):
    # eliminates lines that do not correspond to the stave lines. Those are the lines
    # which both of their gaps differ too much from the meanGap, that means not
    # having a stave line nor above nor below.

    # tolerance: how much can a gap differ from meanGap for not being removed

    i = 1
    while i < len(lineHeights) - 1:
        if (not isGapAboveMeanGap(i, lineHeights, meanGap, tolerance) and
                not isGapBelowMeanGap(i, lineHeights, meanGap, tolerance)):

            lineHeights.pop(i)

        else:
            i += 1

    # print("(", len(lineHeights), ") ", "CONSOLIDATED: ", lineHeights)
    return lineHeights


def getLinesAbove(lineHeight, meanGap, extraLines):
    newLines = []
    i = 1
    while i <= extraLines:
        newLines.append(lineHeight - meanGap * i)
        i += 1
    return newLines


def getLinesBelow(lineHeight, meanGap, extraLines):
    newLines = []
    i = 1
    while i <= extraLines:
        newLines.append(lineHeight + meanGap * i)
        i += 1
    return newLines


def generateStaves(lineHeights, meanGap, extraLines=2):
    staves = []

    i = 1
    staveIndex = 0
    currentStave = Stave(staveIndex)

    while i < len(lineHeights) - 1:

        # add line to currentStave
        currentStave.addLineHeight(lineHeights[i])

        # if meanGap is below, not above, that that means it is the first line of a stave
        if (not isGapAboveMeanGap(i, lineHeights, meanGap, 5) and
                isGapBelowMeanGap(i, lineHeights, meanGap, 5)):
            # set stave top line
            currentStave.setTopLine(lineHeights[i])

        # meanGap is above, not below, that is last line of a stave
        if (isGapAboveMeanGap(i, lineHeights, meanGap, 5) and
                not isGapBelowMeanGap(i, lineHeights, meanGap, 5)):
            # set stave bottom line
            currentStave.setBottomLine(lineHeights[i])

            # store current stave before resetting values
            staves.append(currentStave)
            staveIndex += 1
            currentStave = Stave(staveIndex)

        i += 1

    # generate extra line heights
    for i in range(len(staves)):
        staves[i].addLines(getLinesAbove(staves[i].topLine, meanGap, extraLines))
        staves[i].addLines(getLinesBelow(staves[i].bottomLine, meanGap, extraLines), True)

    return staves


def printStaves(staves):
    for i in range(len(staves)):
        staves[i].print()
        print("")
    return


def drawLineHeights(staves, image, thickness=1):
    for stave in staves:
        for lh in stave.lineHeights:
            cv2.line(image, (0, lh), (IMAGE_WIDTH, lh), COLORS[stave.staveIndex % len(COLORS)], thickness)
    return image
