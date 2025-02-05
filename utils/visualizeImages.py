import os

from constants import myDataImg
from utils.plotUtils import showImage
from vision.imageUtils import loadImageGrey


def showImages(sourcePath):
    # Check if the source folder exists
    if not os.path.exists(sourcePath):
        raise FileNotFoundError(f"The source folder '{sourcePath}' does not exist.")

    # Supported image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # Get a list of all files in the source folder
    files = os.listdir(sourcePath)

    for filename in files:
        # Check if the file is an image
        if filename.lower().endswith(image_extensions):
            # Create full paths for old and new file names
            imagePath = os.path.join(sourcePath, filename)
            print("SHOWING IMAGE: ", imagePath)
            image, _ = loadImageGrey(imagePath)
            showImage(image)


showImages(myDataImg)
