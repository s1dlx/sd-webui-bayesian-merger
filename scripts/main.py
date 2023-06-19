import gradio as gr
import pathlib
import subprocess
import sys
from modules import script_callbacks


main_path = pathlib.Path(__file__).parent.parent / "bayesian_merger.py"


def on_ui_tabs():
    with gr.Blocks() as root:
        generate_button = gr.Button(value="Optimize!", variant="primary")

        generate_button.click(
            fn=launch_optimizer
        )

    return [(root, "Bayesian Merger", "bbwm")]


script_callbacks.on_ui_tabs(on_ui_tabs)


def launch_optimizer():
    args = [
        sys.executable,
        str(main_path),
    ]
    print(f"Starting bayesian merger using command: {' '.join(args)}")
    subprocess.Popen(
        args=args,
        cwd=str(main_path.parent),
    )
