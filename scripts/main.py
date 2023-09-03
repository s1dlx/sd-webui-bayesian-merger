import os

import gradio as gr
import pathlib
import subprocess
import sys
from modules import script_callbacks
from typing import Optional


MAIN_PATH = pathlib.Path(__file__).parent.parent / "bayesian_merger.py"
TOTAL_MANUAL_SCORE_BUTTONS = 11
pipe: Optional[subprocess.Popen] = None


def on_ui_tabs():
    with gr.Blocks() as root:
        generate_button = gr.Button(
            value="Optimize!",
            variant="primary",
        )

        score_buttons = []
        with gr.Accordion(label="Manual scoring"):
            with gr.Row():
                for i in range(TOTAL_MANUAL_SCORE_BUTTONS):
                    score_button = gr.Button(
                        value=str(i),
                        interactive=False,
                    )
                    score_button.click(
                        fn=send_manual_score,
                        inputs=[gr.State(i)],
                    )
                    score_buttons.append(score_button)

        generate_button.click(
            fn=launch_optimizer,
            outputs=score_buttons
        )

    return [(root, "Bayesian Merger", "bbwm")]


script_callbacks.on_ui_tabs(on_ui_tabs)


def launch_optimizer():
    global pipe
    args = [
        sys.executable,
        str(MAIN_PATH),
        #"--webui-protocol",
    ]
    print(f"Starting bayesian merger using command: {' '.join(args)}")
    pipe = subprocess.Popen(
        args=args,
        cwd=str(MAIN_PATH.parent),
        stdin=subprocess.PIPE,
    )

    return [gr.Button.update(interactive=True)] * TOTAL_MANUAL_SCORE_BUTTONS


def send_manual_score(score):
    global pipe
    print(score)
    pipe.stdin.write(f"{score}{os.linesep}".encode("utf-8"))
    pipe.stdin.flush()
