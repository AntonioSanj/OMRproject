from objectTypes.Sound import Sound, MultiSound, Song
from reproduction.playSong import playSong

# right hand track
upperTrack = [MultiSound([Sound('rest', 1)], 0, 1),
              MultiSound([Sound('rest', 0.5)], 1, 0.5),
              MultiSound([Sound('F4#', 0.25)], 1.5, 0.25),
              MultiSound([Sound('A4', 0.25)], 1.75, 0.25),
              MultiSound([Sound('B4', 0.5)], 2, 0.5),
              MultiSound([Sound('A4', 0.5)], 2.5, 0.5),
              MultiSound([Sound('F4#', 0.5)], 3, 0.5),
              MultiSound([Sound('E4', 0.25)], 3.5, 0.25),
              MultiSound([Sound('D4', 0.25)], 3.75, 0.25),
              MultiSound([Sound('E4', 0.25)], 0, 0.25),
              MultiSound([Sound('F4#', 0.25)], 0.25, 0.25),
              MultiSound([Sound('B3', 0.25)], 0.5, 0.25),
              MultiSound([Sound('D4', 0.25)], 0.75, 0.25),
              MultiSound([Sound('D4', 3.0)], 1, 3.0)]

# left hand track
lowerTrack = [MultiSound([Sound('D3', 1.5), Sound('A3', 1.5)], 0, 1.5),
              MultiSound([Sound('F3#', 0.5), Sound('A3', 0.5)], 1.5, 0.5),
              MultiSound([Sound('F3#', 2), Sound('A3', 2)], 2, 2),
              MultiSound([Sound('G3', 1.5)], 0, 1.5),
              MultiSound([Sound('A2', 0.5)], 1.5, 0.5),
              MultiSound([Sound('A2', 1)], 2, 1),
              MultiSound([Sound('G3', 1)], 3, 1)]

song = Song(upperTrack, lowerTrack, 4, 80)

playSong(song)
