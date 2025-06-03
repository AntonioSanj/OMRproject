from vision.visionUtils import *
from vision.staveDetection.staveOperations import *
from utils.plotUtils import *


def getStaves(imagePath, sheetIndex, printData=False, debug=False):
    img, gray = loadImageGrey(imagePath)
    showImage(img) if debug else None

    horizontal_edges = cannyEdges(img, 3)
    showImage(horizontal_edges, 'horizontal edges') if debug else None

    lines, linesImage = getHorizontalLines(horizontal_edges, img, 1000, 700, 120)

    showImage(linesImage, 'Detected lines') if debug else None

    lineHeights, meanGap = getLineHeights(lines, 3)

    consolidatedLines = consolidateLines(lineHeights, meanGap, 5)
    drawHeights(consolidatedLines, img) if debug else None

    # one extra line is added up and below (C and A assuming gClef)
    staves = generateStaves(consolidatedLines, meanGap, sheetIndex, 1)

    result = drawStaves(staves, img, 1)
    showImage(result) if debug else None

    if printData:
        printStaves(staves)

    return result, staves
