from constants import *
from utils.plotUtils import showImage
from vision.visionUtils import loadImageGrey, thresh
from vision.measureBarDetection.measureBarDetectionOperations import getVerticalLines, sobelFilter, filterVerticalEdges, \
    mergeLines, drawLines


def getMeasureBars(imagePath, show=False):
    img, gray = loadImageGrey(imagePath)

    thresh_image = thresh(gray, 160)

    vertical_edges = sobelFilter(thresh_image)

    thresh_edges = thresh(vertical_edges, 160)

    filtered_vertical_edges = filterVerticalEdges(thresh_edges)

    lines = getVerticalLines(filtered_vertical_edges, 100, 20, 10)

    lines = mergeLines(lines)

    lineTuples = [tuple(arr.flatten()) for arr in lines]

    # highest point first (x1, y1, x2, y2) ensure y1 < y2
    lineTuples = [
        (x1, y1, x2, y2) if y1 < y2 else (x1, y2, x2, y1) for x1, y1, x2, y2 in lineTuples
    ]

    if show:
        linesImage = drawLines(lines, img, thickness=5)
        print('Number of measure bars found:', len(lines))
        showImage(linesImage)

    return lineTuples
