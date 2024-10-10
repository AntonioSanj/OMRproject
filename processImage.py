from ImageOperations import *
from utils import *

img, gray = loadImageGrey('sample_images/twisted.png')

thresh_image = thresh(gray, 160)
showImage(thresh_image, 'threshold')

twisted_edges = cannyEdges(thresh_image, 3)
showImage(twisted_edges, 'edges')

rotatedImage = rotateAdjustImage(twisted_edges, img)
showImage(rotatedImage, 'rotate')

horizontal_edges = cannyEdges(rotatedImage, 3)
showImage(horizontal_edges, 'horizontal edges')

linesImage = getHorizontalLines(rotatedImage, horizontal_edges, 20, 1000, 50)
showImage(linesImage, 'lines')

resized_image = adjustImageSize(linesImage)

showImage(resized_image, 'Pentagram detection')
compareToOg(resized_image, gray)
