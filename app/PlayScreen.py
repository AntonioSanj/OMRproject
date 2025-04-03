import os
import sys
import threading

from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen
from kivymd.toast import toast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "main"))
sys.path.append(os.path.join(project_root, "reproduction"))
from main.main import readSheets
from reproduction.playSong import playSong


class PlayScreen(Screen):
    def __init__(self, **kw):
        super().__init__()
        self.playThread = None
        self.readThread = None
        self.is_playing = False
        self.selected_files = None
        self.swing = False
        self.bpm = 100
        self.song = None

    def start_playing(self, selected_files, bpm, swing):
        print(f"Playing files: {selected_files}")
        print(f"BPM: {bpm}, Swing: {swing}")
        self.selected_files = selected_files
        self.bpm = bpm
        self.swing = swing

        toast('Reading sheets in background', background=[0.2, 0.2, 0.2, 0.2], duration=2)
        self.readThread = threading.Thread(target=self.readSongData, daemon=True)
        self.readThread.start()

        self.ids.carousel.clear_widgets()

        file_name = os.path.basename(self.selected_files[0])
        self.ids.file_name_label.text = file_name

        self.ids.carousel.unbind(index=self.update_file_name)
        self.ids.carousel.bind(index=self.update_file_name)

        for file_path in self.selected_files:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.ids.carousel.add_widget(Image(source=file_path, fit_mode="scale-down"))

    def carousel_previous(self):
        carousel = self.ids.carousel
        if carousel.index > 0:
            carousel.load_previous()

    def carousel_next(self):
        carousel = self.ids.carousel
        if carousel.index < len(carousel.slides) - 1:
            carousel.load_next()

    def update_file_name(self, instance, index):
        if index is not None and 0 <= index < len(self.selected_files):
            file_name, _ = os.path.splitext(os.path.basename(self.selected_files[index]))
            self.ids.file_name_label.text = file_name

    def readSongData(self):
        self.song = readSheets(self.selected_files, int(self.bpm), self.swing)
        Clock.schedule_once(lambda dt: toast('Song is ready!', background=[0.2, 0.2, 0.2, 0.2], duration=2))

    def play(self):
        if self.song is None:
            toast('Song not ready yet', background=[0.2, 0.2, 0.2, 0.2], duration=2)
        elif self.is_playing:
            toast('Song is already playing', background=[0.2, 0.2, 0.2, 0.2], duration=2)
        else:
            self.playThread = threading.Thread(target=self.playBackSong, daemon=True)
            self.playThread.start()

    def playBackSong(self):
        self.is_playing = True
        playSong(self.song)
        self.is_playing = False

    def go_back(self):
        if self.is_playing:
            toast('Cannot exit while song playing', background=[0.2, 0.2, 0.2, 0.2], duration=2)
        elif self.song is None:
            toast('Cannot exit while reading sheets', background=[0.2, 0.2, 0.2, 0.2], duration=2)
        else:
            self.manager.current = "main"
