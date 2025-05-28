from constants import fullsheetsDir, myDataImg
from readSheets.readSheetsMain import readSheets
from reproduction.playSong import playSong

# song = readSheets([fullsheetsDir + '/the_chesire_cat.png'], 124, True, show=True)
# playSong(song)
# readSheets([myDataImg + '/pinkpanther1.png', fullsheetsDir + '/pinkpanther2.png'], 117, True)
# readSheets([myDataImg + '/image_9.png', myDataImg + '/image_10.png'], 70)
song = readSheets([fullsheetsDir + '/thinking_out_loud1.png', fullsheetsDir + '/thinking_out_loud2.png'], 80)
playSong(song)
