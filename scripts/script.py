from scripts import optimiser_gui
from modules import scripts, script_callbacks
import importlib
importlib.reload(optimiser_gui)

class BayesianMergerScript(scripts.Script):
    def title(self):
        return 'Bayesian Merger'

    def ui(self, is_img2img):
        return ()

    def show(self, is_img2img):
        return scripts.AlwaysVisible


def on_ui_tabs():
    gui = optimiser_gui.OptimiserGui()
    return [gui.get_webui_tab()]


script_callbacks.on_ui_tabs(on_ui_tabs)
