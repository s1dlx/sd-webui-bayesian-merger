from modules import script_callbacks
import gradio as gr
import bayesian_merger
from hydra import compose, initialize
from pathlib import Path


relative_conf_path = Path("..") / "conf"


def on_ui_tabs():
    with gr.Blocks() as root:
        generate_button = gr.Button(title="Optimize!")

        generate_button.click(
            fn=launch_optimizer
        )

    return [(root, "Bayesian Merger", "bbwm")]


script_callbacks.on_ui_tabs(on_ui_tabs)


def launch_optimizer():
    initialize(config_path=str(relative_conf_path), version_base=None)
    cfg = compose(config_name="config")
    bayesian_merger.start_optimize(cfg)
