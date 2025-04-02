import os
import sys

from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "main"))
from main.main import readAndPlay


class PlayScreen(Screen):
    def __init__(self, **kw):
        super().__init__()
        self.selected_files = None
        self.swing = None
        self.bpm = None

    def start_playing(self, selected_files, bpm, swing):
        print(f"Playing files: {selected_files}")
        print(f"BPM: {bpm}, Swing: {swing}")
        self.selected_files = selected_files
        self.bpm = bpm
        self.swing = swing

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
        if 0 <= index < len(self.selected_files):
            file_name, _ = os.path.splitext(os.path.basename(self.selected_files[index]))
            self.ids.file_name_label.text = file_name

    def play(self):
        print(self.selected_files)

    def go_back(self):
        self.manager.current = "main"
