from objectTypes.SoundDTO import SoundDTO, MultiSound, Song
from reproduction.playSong import playSong

# right hand track
upperTrack = [MultiSound([SoundDTO('rest', 1)], 0, 1),
              MultiSound([SoundDTO('rest', 0.5)], 1, 0.5),
              MultiSound([SoundDTO('F4#', 0.25)], 1.5, 0.25),
              MultiSound([SoundDTO('A4', 0.25)], 1.75, 0.25),
              MultiSound([SoundDTO('B4', 0.5)], 2, 0.5),
              MultiSound([SoundDTO('A4', 0.5)], 2.5, 0.5),
              MultiSound([SoundDTO('F4#', 0.5)], 3, 0.5),
              MultiSound([SoundDTO('E4', 0.25)], 3.5, 0.25),
              MultiSound([SoundDTO('D4', 0.25)], 3.75, 0.25),
              MultiSound([SoundDTO('E4', 0.25)], 0, 0.25),
              MultiSound([SoundDTO('F4#', 0.25)], 0.25, 0.25),
              MultiSound([SoundDTO('B3', 0.25)], 0.5, 0.25),
              MultiSound([SoundDTO('D4', 0.25)], 0.75, 0.25),
              MultiSound([SoundDTO('D4', 3.0)], 1, 3.0)]

# left hand track
lowerTrack = [MultiSound([SoundDTO('D3', 1.5), SoundDTO('A3', 1.5)], 0, 1.5),
              MultiSound([SoundDTO('F3#', 0.5), SoundDTO('A3', 0.5)], 1.5, 0.5),
              MultiSound([SoundDTO('F3#', 2), SoundDTO('A3', 2)], 2, 2),
              MultiSound([SoundDTO('G3', 1.5)], 0, 1.5),
              MultiSound([SoundDTO('A2', 0.5)], 1.5, 0.5),
              MultiSound([SoundDTO('A2', 1)], 2, 1),
              MultiSound([SoundDTO('G3', 1)], 3, 1)]

song = Song(upperTrack, lowerTrack, 4, 80)

playSong(song)
