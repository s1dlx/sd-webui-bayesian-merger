from scripts import main_gui
from modules import scripts, script_callbacks
import importlib
importlib.reload(main_gui)

class BayesianMergerScript(scripts.Script):
    def title(self):
        return 'Bayesian Merger'

    def ui(self, is_img2img):
        return ()

    def show(self, is_img2img):
        return scripts.AlwaysVisible


def on_ui_tabs():
    return [main_gui.create_tab()]


script_callbacks.on_ui_tabs(on_ui_tabs)
