from typing import Tuple, List
import gradio as gr

from scripts import optimiser_gui
from modules import script_callbacks
import importlib
importlib.reload(optimiser_gui)


def on_ui_tabs() -> List[Tuple[gr.Blocks, str, str]]:
    gui = optimiser_gui.OptimiserGui()
    return [gui.get_webui_tab()]


script_callbacks.on_ui_tabs(on_ui_tabs)
