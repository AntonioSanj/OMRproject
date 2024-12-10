import csv
import json
import os

from constants import *


def getAllBoundingBoxes(srcDir, outputDir):
    # get bounding boxes for each file in srcDir

    # check if source and output directories exist
    if not os.path.exists(srcDir):
        raise FileNotFoundError(f"Source directory '{srcDir}' does not exist.")
    if not os.path.exists(outputDir):
        raise FileNotFoundError(f"Output directory '{outputDir}' does not exist.")

    # clear output directory
    for file_name in os.listdir(outputDir):
        file_path = os.path.join(outputDir, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"removed {file_name}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

    # generate csv for each json file
    for file_name in os.listdir(srcDir):
        if file_name.endswith(".json"):  # process only json files
            json_file_path = os.path.join(srcDir, file_name)

            try:
                getBoundingBoxes(json_file_path, outputDir)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
    return


def getBoundingBoxes(jsonPath, outputDir):

    if not os.path.exists(outputDir):
        raise FileNotFoundError(f"Output directory '{outputDir}' does not exist.")

    # extract base name of json file ,
    json_filename = os.path.basename(jsonPath)
    # remove extensions png and json
    base_name = json_filename.replace(".png.json", "").replace(".json", "")
    # and add csv extension
    csv_filename = base_name + ".csv"

    # output path for csv file
    csv_file = os.path.join(outputDir, csv_filename)

    # load json data
    with open(jsonPath, "r") as file:
        data = json.load(file)

    # config csv data
    csv_data = [["classTitle", "x1", "y1", "x2", "y2"]]

    for obj in data.get("objects", []):
        class_title = obj.get("classTitle", "")
        points_exterior = obj.get("points", {}).get("exterior", [])
        if len(points_exterior) == 2:  # two points
            x1, y1 = points_exterior[0]
            x2, y2 = points_exterior[1]
            csv_data.append([class_title, x1, y1, x2, y2])

    # write to csv file
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"{csv_filename} created")


getAllBoundingBoxes(myDataJson, myDataCsv)
