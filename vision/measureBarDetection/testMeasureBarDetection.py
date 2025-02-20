import os
from constants import *

from vision.measureBarDetection.measureBarDetector import getMeasureBars


def testMeasureBarDetection(img_dir):
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path}")

        getMeasureBars(file_path, True)


testMeasureBarDetection(myDataImg)

getMeasureBars(fullsheetsDir + 'thinking_out_loud1.png', True)
