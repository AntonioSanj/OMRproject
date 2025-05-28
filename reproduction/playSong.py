import os
import threading
import time
import pygame

from constants import soundFilesDir
from reproduction.Metronome import Metronome


def playSong(song):
    # Initialize the pygame mixer
    pygame.mixer.init()
    pygame.mixer.set_num_channels(64)
    print("Playing song...")

    # Create and start the global clock
    clock = Metronome(song.bpm)
    clock_thread = threading.Thread(target=clock.start)
    clock_thread.start()

    # create threads for both tracks so they play in parallel
    upper_thread = threading.Thread(target=play_track, args=(song.upperTrack, clock))
    lower_thread = threading.Thread(target=play_track, args=(song.lowerTrack, clock))

    upper_thread.start()
    lower_thread.start()

    upper_thread.join()
    lower_thread.join()

    # Stop the clock once the song finishes
    clock.stop()
    clock_thread.join()

    return


def play_track(track, clock):
    for multisound in track:
        while True:
            current_beat = clock.get_pulse()
            if current_beat >= multisound.start:  # allow some error margin
                break  # play the multisound
            time.sleep(0.001)  # keep waiting

        play_multisound(multisound)


def play_multisound(multisound):
    for sound in multisound.sounds:
        play_sound(sound.sound, sound.duration)


def play_sound(soundName, duration):
    soundPath = os.path.join(soundFilesDir, f"{soundName}.wav")

    if not os.path.exists(soundPath):
        soundPath = os.path.join(soundFilesDir, "rest.wav")

    sound = pygame.mixer.Sound(soundPath)

    sound.play(maxtime=int(duration * 1000))
