import cv2


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
