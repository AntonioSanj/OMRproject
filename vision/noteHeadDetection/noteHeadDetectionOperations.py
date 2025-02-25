import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import flood

from utils.plotUtils import showImage, showCompareImages


def closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def fillWhiteHoles(image):
    image = cv2.bitwise_not(image)
    image = binary_fill_holes(image).astype(np.uint8)
    image = cv2.bitwise_not(image)

    return image


def floodRegion(image, x, y):
    pixel = (y, x)
    mask = flood(image, pixel, connectivity=2)  # Generate flood mask
    return mask


def cleanLines(image):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]).astype(np.uint8)
    noLines = closing(image, kernel)
    return noLines


def getQuaverBarArea(image):
    height, width = image.shape

    # remove both vertical and horizontal lines from image
    noLinesImage = cleanLines(image)

    quaverBarArea = np.zeros(image.shape)

    # traverse left and right borders
    for y in range(1, height - 1):
        # check for black pixels in the border
        # if it is not painted yet in the result paint it
        if noLinesImage[y, 0] == 0 and quaverBarArea[y, 0] == 0:
            mask = floodRegion(noLinesImage, 0, y)
            quaverBarArea[mask] = 255.

        if noLinesImage[y, width - 1] == 0 and quaverBarArea[y, width - 1] == 0:
            mask = floodRegion(noLinesImage, width - 1, y)
            quaverBarArea[mask] = 255.

    return quaverBarArea.astype(bool)


def findLocalMax(headLocations, windowHeight=20, threshold=0.4):
    height, width = headLocations.shape

    # mask two store the local maxima points
    mask = np.zeros_like(headLocations, dtype=np.uint8)

    midLine = width // 2

    # iterate through image from top to bottom in two separate windows
    # as heads can be either on the right or left
    for y in range(0, height, windowHeight // 2):
        # make sure we are getting the final rows
        y_end = min(y + windowHeight, height)

        # extract windows
        leftWindow = headLocations[y:y_end, :midLine]
        rightWindow = headLocations[y:y_end, midLine:]

        # get maximum in each window
        _, maxValueLeft, _, maxLocationLeft = cv2.minMaxLoc(leftWindow)
        _, maxValueRight, _, maxLocationRight = cv2.minMaxLoc(rightWindow)

        # evaluate threshold
        # translate coordinates to global image
        # paint new max point in the mask
        if maxValueLeft > threshold:
            point = (maxLocationLeft[0], y + maxLocationLeft[1])
            mask[point[1], point[0]] = maxValueLeft * 255

        if maxValueRight > threshold:
            point = (midLine + maxLocationRight[0], y + maxLocationRight[1])
            mask[point[1], point[0]] = maxValueRight * 255

    return mask


def findEnclosedWhiteRegions(binary_image, max_area=20):
    # Find contours of all white regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary_image, dtype=np.uint8)

    # Loop through all contours
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter out small enclosed areas
        if 0 < area <= max_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)  # Fill the mask with detected regions

    return mask.astype(bool)


def cleanCorners(image):
    # corners are black so we invert the image
    inverse = cv2.bitwise_not(image)
    # then remove small white areas (previously black)
    mask = findEnclosedWhiteRegions(inverse, 20).astype(bool)
    # apply the mask
    image[mask] = 255.

    return image


def consolidatePoints(noteHeadPoints, threshold=10):
    pointCoords = np.argwhere(noteHeadPoints > 0)
    pointValues = noteHeadPoints[noteHeadPoints > 0]
    # point => ((x_pos, y_pos), value)
    points = [((x, y), value) for (y, x), value in zip(pointCoords, pointValues)]
    print(points)

    filteredPoints = []
    for (x, y), value in points:
        foundClosePoint = False
        # search for close points in the result list
        for i, ((fpx, fpy), fp_value) in enumerate(filteredPoints):

            distance = np.sqrt((fpx - x)**2 + (fpy - y)**2)

            if distance < threshold:
                # close point is found
                if fp_value < value:
                    # replace the with the larger value
                    filteredPoints[i] = ((x, y), value)

                foundClosePoint = True

        # no close points found => add point to result
        if not foundClosePoint:
            filteredPoints.append(((x, y), value))

    return filteredPoints


def drawPoints(noteHeadPoints, image, filename="Note head detection"):

    imgArray = np.array(image)

    for (x, y), _ in noteHeadPoints:
        imgArray[y, x] = (255, 0, 255)

    imgCopy = Image.fromarray(imgArray)

    showImage(imgCopy, filename)
    return

