from sd_webui_bayesian_merger.optimiser import Optimiser
from sd_webui_bayesian_merger.bayes_optimiser import BayesOptimiser
from sd_webui_bayesian_merger.tpe_optimiser import TPEOptimiser
from modules import shared
from dataclasses import dataclass
import gradio as gr
import torch


# use webui's clip skip value
# wildcards. use extensions instead?
# split optimiser model name vs optimiser models path. detect optimiser?
# draw unet merge chart


def create_tab():
    gui = OptimizerGui()
    with gr.Blocks() as root:
        rearrange(gui)
        connect_events(gui)

    return root, 'Bayesian Merger', 'bayesian_merger_root'


@dataclass
class OptimizerGui:
    api_url = gr.Textbox(visible=False, elem_id='bayesian_merger_api_url')
    model_a = gr.Dropdown(
        label='Model A',
        choices=shared.list_checkpoint_tiles(),
    )
    model_b = gr.Dropdown(
        label='Model B',
        choices=shared.list_checkpoint_tiles(),
    )
    device = gr.Dropdown(
        label='Merge on device',
        choices=['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())],
        value='cpu',
    )
    payloads_dir = gr.Textbox(
        label='Payloads directory',
    )
    scorer_model = gr.Textbox(
        label='Path to scorer model',
    )
    optimiser = gr.Dropdown(
        label='Optimiser',
        choices=['bayes', 'tpe'],
        value='bayes',
    )
    batch_count = gr.Number(
        label='Batch count',
        value=10,
    )
    init_points = gr.Number(
        label='Initialization points',
        value=10,
    )
    iterations = gr.Number(
        label='Iterations',
        value=10,
    )
    save_model = gr.Checkbox(
        label='Save best model',
    )
    save_model_format = gr.Dropdown(
        label='Model format',
        choices=['safetensors', 'ckpt'],
        value='safetensors',
    )
    save_model_precision = gr.Dropdown(
        label='Model precision',
        choices=['16', '32'],
        value='16',
    )
    start_optimiser_button = gr.Button(
        value='Start Optimizer',
        variant='primary',
    )


def connect_events(gui: OptimizerGui):
    optimiser_object = gr.State()
    gui.optimiser.change(
        fn=on_create_optimiser,
        inputs=[gui.optimiser, gui.api_url],
        outputs=[optimiser_object]
    )
    gui.start_optimiser_button.click(
        fn=on_start_optimise,
        inputs=[optimiser_object]
    )


def rearrange(gui: OptimizerGui):
    gui.api_url.render()


def on_start_optimise(optimiser: Optimiser):
    optimiser.optimise()
    optimiser.postprocess()


def on_create_optimiser(optimiser_tag: str, *args):
    clip_skip = shared.opts.CLIP_stop_at_last_layers - 1
    return optimisers[optimiser_tag](*args, skip_position_ids=clip_skip)


optimisers = {
    'bayes': BayesOptimiser,
    'tpe': TPEOptimiser,
}
