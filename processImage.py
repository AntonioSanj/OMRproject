from ImageOperations import *
from utils import *

img, gray = loadImageGrey('sample_images/twisted.png')

twisted_edges = cannyEdges(gray, True)

showImage(twisted_edges)

rotatedImage = rotateAdjustImage(twisted_edges, img)

showImage(rotatedImage)

horizontal_edges = cannyEdges(rotatedImage, True)

showImage(horizontal_edges)

linesImage = getHorizontalLines(rotatedImage, horizontal_edges)

showImage(linesImage)

resized_image = adjustImageSize(linesImage)

showImage(resized_image, 'Pentagram detection')
compareToOg(resized_image, gray)
