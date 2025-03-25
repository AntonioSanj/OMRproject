import os
import sys
from os.path import expanduser

from kivy.core.window import Window
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.card import MDSeparator
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "main"))
from main.main import readAndPlay

mainScreen = """
Screen:
    BoxLayout:
        orientation: 'vertical'
                
        MDTopAppBar:
            title: 'Demo Application'
            elevation: 5
            
        BoxLayout:
            orientation: 'vertical'
            padding: dp(30)
                
            BoxLayout:
                orientation: 'horizontal'
                spacing: dp(30)
                size_hint_y: None
                height: dp(80)
                padding: dp(30)
                
                MDRaisedButton:
                    text: "Load Images"
                    size_hint_x: 0.5
                    size_hint_y: None
                    height: dp(70)
                    on_release: app.open_file_manager()
                    
                    
                MDTextField:
                    id: bpm_input
                    hint_text: "BPM"
                    input_filter: "int"  # Allows only integer values
                    height: dp(70)
                    size_hint_x: 0.2
                    pos_hint: {"center_x": 0.5}
    
            Widget:
                size_hint_y: 0.05
    
            MDScrollView:
                size_hint_y: 0.5
                MDList:
                    id: file_list
    
            Widget:
                size_hint_y: 0.05
                
            MDRaisedButton:
                id: play_button
                text: "Play"
                pos_hint: {"center_x": 0.5}
                size_hint_y: None
                height: dp(50)
                on_release: app.play()
                opacity: 0
                disabled: True
                
            Widget:
                size_hint_y: 0.05
"""


class MyApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = None
        self.selected_files = []

    def build(self):

        self.theme_cls.primary_palette = "Blue"
        self.file_manager = MDFileManager(
            select_path=self.select_files,
            exit_manager=self.close_file_manager,
            preview=True,
            selector="multi"
        )

        Window.size = (480*1.1, 800*1.1)
        Window.title = "Demo Application"

        ui = Builder.load_string(mainScreen)

        return ui

    def open_file_manager(self):
        self.file_manager.show(expanduser("~/Desktop/UDC/QUINTO/TFG/src_code/dataset/fullsheets"))

    def select_files(self, paths):
        if paths:
            cleaned_paths = [os.path.abspath(path) for path in paths]  # Convert to absolute paths
            self.update_file_list(cleaned_paths)
            self.file_manager.close()

    def update_file_list(self, files):
        self.selected_files = files
        file_list = self.root.ids.file_list
        file_list.clear_widgets()
        play_button = self.root.ids.play_button

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
        """Moves an item up or down in the list."""
        if direction == "up" and index > 0:
            self.selected_files[index], self.selected_files[index - 1] = self.selected_files[index - 1], \
                self.selected_files[index]
        elif direction == "down" and index < len(self.selected_files) - 1:
            self.selected_files[index], self.selected_files[index + 1] = self.selected_files[index + 1], \
                self.selected_files[index]

        self.update_file_list(self.selected_files)

    def delete_item(self, file_path):
        if file_path in self.selected_files:
            self.selected_files.remove(file_path)  # Remove from list
            self.update_file_list(self.selected_files)

    def close_file_manager(self):
        self.file_manager.close()

    def play(self):
        bpm = self.root.ids.bpm_input.text.strip()

        readAndPlay(self.selected_files, bpm)


if __name__ == "__main__":
    MyApp().run()
