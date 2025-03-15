import os
import threading
import time
import pygame

from constants import soundFilesDir
from reproduction.PulseClock import PulseClock


def playSong(song):
    # Initialize the pygame mixer
    pygame.mixer.init()
    print("Playing song...")

    # Create and start the global clock
    clock = PulseClock(song.bpm)
    clock_thread = threading.Thread(target=clock.start)
    clock_thread.start()

    # Create threads for both tracks so they play in parallel
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
            current_pulse = clock.get_pulse()
            if current_pulse >= multisound.start:
                break  # play the multisound
            time.sleep(0.001)  # keep waiting

        play_multisound(multisound)


def play_multisound(multisound):
    for sound_dto in multisound.sounds:
        threading.Thread(target=play_sound, args=(sound_dto.sound, sound_dto.duration), daemon=True).start()


def play_sound(soundName, duration):
    soundPath = os.path.join(soundFilesDir, f"{soundName}.wav")

    if not os.path.exists(soundPath):
        soundPath = os.path.join(soundFilesDir, "rest.wav")

    sound = pygame.mixer.Sound(soundPath)
    sound.play(maxtime=int(duration * 1000))
