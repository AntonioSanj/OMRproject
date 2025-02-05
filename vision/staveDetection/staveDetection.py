from vision.imageUtils import *
from vision.staveDetection.staveOperations import *
from utils.plotUtils import *


def getStaves(imagePath, show=False, printData=False, debug=False):
    img, gray = loadImageGrey(imagePath)

    thresh_image = thresh(gray, 160)
    # showImage(thresh_image, 'threshold')

    twisted_edges = cannyEdges(thresh_image, 3)
    # showImage(twisted_edges, 'edges')

    rotatedImage = rotateAdjustImage(twisted_edges, img)
    # showImage(rotatedImage, 'rotate')

    horizontal_edges = cannyEdges(rotatedImage, 3)
    # showImage(horizontal_edges, 'horizontal edges')

    lines, linesImage = getHorizontalLines(horizontal_edges, rotatedImage, 1000, 700, 120)

    if debug:
        showImage(linesImage, 'Detected lines')

    lineHeights, meanGap = getLineHeights(lines, 3)

    consolidatedLines = consolidateLines(lineHeights, meanGap, 5)

    staves = generateStaves(consolidatedLines, meanGap, 0)

    result = drawLineHeights(staves, rotatedImage, 2)

    if printData:
        printStaves(staves)

    if show:
        compareToOg(result, gray)

    return result, staves
