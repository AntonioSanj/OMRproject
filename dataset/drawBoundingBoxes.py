import csv
import os
import cv2

from constants import *
from utils.plotUtils import showImage


def drawAllBoundingBoxes(imagesPath, csvAnnPath):
    if not os.path.exists(imagesPath):
        raise FileNotFoundError(f"Image directory '{imagesPath}' does not exist.")
    if not os.path.exists(csvAnnPath):
        raise FileNotFoundError(f"Csv directory '{csvAnnPath}' does not exist.")

    for file_name in os.listdir(csvAnnPath):
        if file_name.endswith(".csv"):  # process only csv files
            csv_file_path = os.path.join(csvAnnPath, file_name)
            # get image path
            image_file_name = file_name.replace(".csv", ".png")
            image_file_path = os.path.join(imagesPath, image_file_name)

            # draw bounding boxes and show image
            image = drawBoundingBoxes(image_file_path, csv_file_path)
            showImage(image, image_file_name)

    return


def drawBoundingBoxes(imagePath, csvPath):
    # check image and csv exist
    if not os.path.exists(imagePath):
        raise FileNotFoundError(f"Image file '{imagePath}' does not exist.")
    if not os.path.exists(csvPath):
        raise FileNotFoundError(f"CSV file '{csvPath}' does not exist.")

    # load image
    image = cv2.imread(imagePath)

    # load data from the csv
    with open(csvPath, "r") as file:
        reader = csv.reader(file)
        next(reader)  # skip first line
        for row in reader:
            classTitle, x1, y1, x2, y2 = row
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # convert coordinates to integers

            # draw boxes and write the class
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(image, classTitle, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return image


drawAllBoundingBoxes(myDataImg, myDataCsv)
