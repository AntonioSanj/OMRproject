from ImageOperations import *
from utils import *

img, gray = loadImageGrey('sample_images/twisted.png')

thresh_image = thresh(gray, 160)
#showImage(thresh_image, 'threshold')

twisted_edges = cannyEdges(thresh_image, 3)
#showImage(twisted_edges, 'edges')

rotatedImage = rotateAdjustImage(twisted_edges, img)
#showImage(rotatedImage, 'rotate')

horizontal_edges = cannyEdges(rotatedImage, 3)
#showImage(horizontal_edges, 'horizontal edges')

linesImage, lines = getHorizontalLines(rotatedImage, horizontal_edges, 1000, 1000, 50)
#showImage(linesImage, 'lines')

lineHeights = getLineHeights(lines, 3)


showImage(linesImage, 'Pentagram detection')
#compareToOg(linesImage, gray)
