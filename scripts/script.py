from typing import Tuple, List
import gradio as gr

from scripts import optimiser_gui
from modules import scripts, script_callbacks
import importlib
importlib.reload(optimiser_gui)

class BayesianMergerScript(scripts.Script):
    def title(self) -> str:
        return 'Bayesian Merger'

    def ui(self, is_img2img: bool) -> Tuple[gr.components.Component, ...]:
        return ()

    def show(self, is_img2img: bool) -> bool | scripts.AlwaysVisible:
        return scripts.AlwaysVisible


def on_ui_tabs() -> List[Tuple[gr.Blocks, str, str]]:
    gui = optimiser_gui.OptimiserGui()
    return [gui.get_webui_tab()]


script_callbacks.on_ui_tabs(on_ui_tabs)
