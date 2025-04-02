from constants import fullsheetsDir, myDataImg
from main.main import readSheets

readSheets([fullsheetsDir + '/the_chesire_cat.png'], 124, True)
readSheets([myDataImg + '/pinkpanther1.png', fullsheetsDir + '/pinkpanther2.png'], 117, True)
readSheets([myDataImg + '/image_9.png', myDataImg + '/image_10.png'], 70)
readSheets([fullsheetsDir + '/thinking_out_loud1.png', fullsheetsDir + '/thinking_out_loud2.png'], 80)