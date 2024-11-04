import os
import shutil
from constants import *


def labelImages(sourcePath, destinationPath, isMask=False):
    # Check if the source folder exists
    if not os.path.exists(sourcePath):
        raise FileNotFoundError(f"The source folder '{sourcePath}' does not exist.")

    # Check if the source folder exists
    if not os.path.exists(destPath):
        raise FileNotFoundError(f"The destination folder '{destPath}' does not exist.")

    # Supported image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # Clean all images in the destination folder
    for filename in os.listdir(destinationPath):
        file_path = os.path.join(destinationPath, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f'Deleted: "{filename}" from "{destinationPath}"')
            except PermissionError:
                print(f'PermissionError: Could not delete "{filename}". It may be in use by another process.')
                # Optionally: Retry logic can be added here if needed

    # Get a list of all files in the source folder
    files = os.listdir(sourcePath)

    # Counter for renaming
    counter = 1

    for filename in files:
        # Check if the file is an image
        if filename.lower().endswith(image_extensions):
            # Create the new name (e.g., image_1.jpg)
            if isMask:
                new_name = f"image_{counter}_inst{os.path.splitext(filename)[1]}"
            else:
                new_name = f"image_{counter}{os.path.splitext(filename)[1]}"


            # Create full paths for old and new file names
            old_file = os.path.join(sourcePath, filename)
            new_file = os.path.join(destinationPath, new_name)

            # Copy the file to the new folder
            shutil.copy2(old_file, new_file)
            print(f'Copied: "{filename}" to "{new_name}" in "{destPath}"')

            # Increment the counter
            counter += 1


# Specify the source folder and destination folder here
srcPath = denseDataSetImages
destPath = denseDataSetImagesLabeled
src2 = denseDataSetAnnotations
dest2 = denseDataSetAnnotationsLabeled

# Call the function
labelImages(srcPath, destPath)
labelImages(src2, dest2)
