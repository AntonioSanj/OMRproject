import os
import threading
import time
import pygame

from constants import soundFilesDir


def playSong(song):
    # Initialize the pygame mixer
    pygame.mixer.init()
    print("Playing song...")

    # Create threads for both tracks so they play in parallel
    upper_thread = threading.Thread(target=play_track, args=(song.upperTrack, song.bpm))
    lower_thread = threading.Thread(target=play_track, args=(song.lowerTrack, song.bpm))

    upper_thread.start()
    lower_thread.start()

    upper_thread.join()
    lower_thread.join()

    return


def play_track(track, bpm):
    start_time = time.time()

    for multisound in track:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Convert start (in beats) to seconds
        start_seconds = multisound.start * (60 / bpm)

        wait_time = start_seconds - elapsed_time

        if wait_time > 0:
            time.sleep(wait_time)  # Wait until the correct pulse time

        play_multisound(multisound, bpm)


def play_multisound(multisound, bpm):
    threads = []

    for sound_dto in multisound.sounds:
        duration_in_seconds = (sound_dto.duration / bpm) * 60  # Convert beats to seconds

        # Start a thread to play each sound independently
        thread = threading.Thread(target=play_sound, args=(sound_dto.sound, duration_in_seconds))
        threads.append(thread)
        thread.start()

    # Ensure all sounds in the MultiSound finish before continuing
    for thread in threads:
        thread.join()


def play_sound(soundName, duration):
    """Loads and plays a sound file for the specified duration."""
    soundPath = soundFilesDir + '/' + soundName + '.wav'
    if os.path.exists(soundPath):
        sound = pygame.mixer.Sound(soundPath)
        sound.play()
        time.sleep(duration)  # Keep playing for the duration
        sound.stop()
    else:
        soundPath = soundFilesDir + '/' + 'rest.wav'
        sound = pygame.mixer.Sound(soundPath)
        sound.play()
        time.sleep(duration)  # Keep playing for the duration
        sound.stop()
