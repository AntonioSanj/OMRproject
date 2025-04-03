import os
import sys

from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from app.MainScreen import MainScreen
from app.MainScreenUI import mainScreen
from app.PlayScreen import PlayScreen
from app.PlayScreenUI import playScreen

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "main"))

KV = """
ScreenManager:
    MainScreen:
    PlayScreen:
""" + mainScreen + playScreen


class MyApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Red"

        Window.size = (430 * 1.1, 800 * 1.1)
        Window.title = "Demo Application"

        self.sm = ScreenManager()
        self.sm.add_widget(MainScreen(name="main"))
        self.sm.add_widget(PlayScreen(name="play"))
        return Builder.load_string(KV)


if __name__ == "__main__":
    MyApp().run()
