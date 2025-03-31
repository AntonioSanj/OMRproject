import os
import sys
from os.path import expanduser

from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.card import MDSeparator
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel

from app.MainScreen import mainScreen
from app.PlayScreen import playScreen

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "main"))
from main.main import readAndPlay

KV = """
ScreenManager:
    MainScreen:
    PlayScreen:
""" + mainScreen + playScreen


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = MDFileManager(
            select_path=self.select_files,
            exit_manager=self.close_file_manager,
            preview=True,
            selector="multi"
        )

    def open_file_manager(self):
        self.file_manager.show(expanduser("~/Desktop/UDC/QUINTO/TFG/src_code/dataset/fullsheets"))

    def select_files(self, paths):
        if paths:
            cleaned_paths = [os.path.abspath(path) for path in paths]
            self.update_file_list(cleaned_paths)
            self.file_manager.close()

    def close_file_manager(self, *args):
        self.file_manager.close()

    def update_file_list(self, files):
        self.selected_files = files
        file_list = self.ids.file_list
        file_list.clear_widgets()
        play_button = self.ids.play_button

        for index, file_path in enumerate(self.selected_files):
            file_name = os.path.basename(file_path)

            # Create a horizontal layout for each item (file name + delete button)
            item_layout = MDBoxLayout(orientation="horizontal", adaptive_height=True, padding=(10, 5))

            # File label (simulating OneLineListItem without built-in separator issues)
            file_label = MDLabel(text=f"{index + 1}. {file_name}", size_hint_x=0.9)
            file_label.full_path = file_path
            item_layout.add_widget(file_label)

            # Delete button
            btn_delete = MDIconButton(icon="trash-can", on_release=lambda x, path=file_path: self.delete_item(path))

            # Move Up Button (Only if not first item)

            btn_up = MDIconButton(
                icon="arrow-up",
                on_release=lambda x, idx=index: self.move_item(idx, "up")
            )
            if index > 0:
                item_layout.add_widget(btn_up)

            # Move Down Button (Only if not last item)

            btn_down = MDIconButton(
                icon="arrow-down",
                on_release=lambda x, idx=index: self.move_item(idx, "down")
            )
            if index < len(self.selected_files) - 1:
                item_layout.add_widget(btn_down)

            item_layout.add_widget(btn_delete)

            # Add row and separator to the file list
            file_list.add_widget(item_layout)
            file_list.add_widget(MDSeparator())

        # Show play button if files exist
        play_button.opacity = 1 if files else 0
        play_button.disabled = not bool(files)

    def move_item(self, index, direction):
        if direction == "up" and index > 0:
            self.selected_files[index], self.selected_files[index - 1] = self.selected_files[index - 1], \
                self.selected_files[index]
        elif direction == "down" and index < len(self.selected_files) - 1:
            self.selected_files[index], self.selected_files[index + 1] = self.selected_files[index + 1], \
                self.selected_files[index]
        self.update_file_list(self.selected_files)

    def delete_item(self, file_path):
        if file_path in self.selected_files:
            self.selected_files.remove(file_path)
            self.update_file_list(self.selected_files)

    def play(self):
        bpm_input = self.ids.bpm_input.text.strip()
        swing_active = self.ids.swing_checkbox.active
        self.manager.get_screen("play").start_playing(self.selected_files, bpm_input, swing_active)
        self.manager.current = "play"


class PlayScreen(Screen):
    def start_playing(self, selected_files, bpm, swing):
        """Receives file list, BPM, and Swing settings to start playback."""
        print(f"Playing files: {selected_files}")
        print(f"BPM: {bpm}, Swing: {swing}")
    pass


class MyApp(MDApp):
    def build(self):

        self.theme_cls.primary_palette = "Blue"

        Window.size = (480 * 1.1, 800 * 1.1)
        Window.title = "Demo Application"

        self.sm = ScreenManager()
        self.sm.add_widget(MainScreen(name="main"))
        self.sm.add_widget(PlayScreen(name="play"))
        return Builder.load_string(KV)


if __name__ == "__main__":
    MyApp().run()
