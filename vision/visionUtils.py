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


def cannyEdges(image, dilate_kernel_radius):
    # compute edges
    edges = cv2.Canny(image, 50, 200, apertureSize=3)

    if dilate_kernel_radius != 0:
        # dilate canny edges for easing line detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_radius, dilate_kernel_radius))
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def createKernelFromImage(image_path):
    img, imgGrey = loadImageGrey(image_path)  # Load the image in grayscale
    _, kernel = cv2.threshold(imgGrey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return kernel
