import os
import shutil

from constants import reducedSetImages, denseDataSetImages, denseDataSetAnnotations, reducedSetAnnotations


def copyImages(sourceDir, outputDir, maxData=-1):
    # Check if sourceDir exists
    if not os.path.exists(sourceDir):
        raise FileNotFoundError(f"Source directory '{sourceDir}' does not exist.")

    # Check if outputDir exists
    if not os.path.exists(outputDir):
        raise FileNotFoundError(f"Output directory '{outputDir}' does not exist.")

    # Clear output folder
    for filename in os.listdir(outputDir):
        file_path = os.path.join(outputDir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Make copies of elements in new folder
    count = 0

    for filename in os.listdir(sourceDir):
        source_path = os.path.join(sourceDir, filename)
        dest_path = os.path.join(outputDir, filename)

        # Stop processing when maxData is reached
        if maxData != -1 and count >= maxData:
            print(f"PROCESSING FINISHED SUCCESSFULLY. REACHED MAX-DATA={maxData}")
            return

        if os.path.isfile(source_path):  # Ensure only files are copied
            shutil.copy2(source_path, dest_path)
            count += 1
            print(f"({count}):\t", filename, "\t\tmoved successfully")

    print("PROCESSING COMPLETED SUCCESSFULLY.")
    return


# Call the function with the specified directories and maxData limit
copyImages(denseDataSetImages, reducedSetImages, 50)
copyImages(denseDataSetAnnotations, reducedSetAnnotations, 50)
