from constants import *
from utils.plotUtils import showImage
from vision.visionUtils import loadImageGrey, thresh
from vision.measureBarDetection.measureBarDetectionOperations import getVerticalLines, sobelFilter, filterVerticalEdges, \
    mergeLines, drawLines


def getMeasureBars(imagePath, show=False):
    img, gray = loadImageGrey(imagePath)

    thresh_image = thresh(gray, 160)
    # showImage(thresh_image, 'threshold')

    vertical_edges = sobelFilter(thresh_image)
    # showImage(vertical_edges, 'vertical edges')

    thresh_edges = thresh(vertical_edges, 160)

    filtered_vertical_edges = filterVerticalEdges(thresh_edges)
    # showImage(filtered_vertical_edges, 'filtered edges')

    lines = getVerticalLines(filtered_vertical_edges, 100, 20, 10)

    lines = mergeLines(lines)

    linesImage = drawLines(lines, img, thickness=5)

    if show:
        print('Number of measure bars found:', len(lines))
        showImage(linesImage)

    lineTuples = [tuple(arr.flatten()) for arr in lines]
    print(lineTuples)
    return

