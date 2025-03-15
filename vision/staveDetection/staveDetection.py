from vision.visionUtils import *
from vision.staveDetection.staveOperations import *
from utils.plotUtils import *


def getStaves(imagePath, sheetIndex, show=False, printData=False, debug=False):
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

    # one extra line is added up and below (C and A assuming gClef)
    staves = generateStaves(consolidatedLines, meanGap, sheetIndex, 1)

    result = drawLineHeights(staves, rotatedImage, 2)

    if printData:
        printStaves(staves)

    if show:
        showCompareImages(result, gray)

    return result, staves
