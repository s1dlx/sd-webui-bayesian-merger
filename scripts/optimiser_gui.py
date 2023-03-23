import subprocess
import sys
from pathlib import Path

from modules import shared
from dataclasses import dataclass, fields, field
import gradio as gr
import torch


# personal notes
### wildcards. use extensions instead?
### detect optimiser?
### draw unet merge chart


def factory_field(cls, **kwargs):
    return field(default_factory=lambda: cls(**kwargs))


@dataclass
class OptimiserGui:
    api_url: ... = factory_field(
        gr.Textbox,
        visible=False,
        elem_id='bayesian_merger_api_url',
    )
    model_a: ... = factory_field(
        gr.Dropdown,
        label='Model A',
        choices=shared.list_checkpoint_tiles(),
    )
    model_b: ... = factory_field(
        gr.Dropdown,
        label='Model B',
        choices=shared.list_checkpoint_tiles(),
    )
    device: ... = factory_field(
        gr.Dropdown,
        label='Merge on device',
        choices=['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())],
        value='cpu',
    )
    payloads_dir: ... = factory_field(
        gr.Textbox,
        label='Payloads directory',
    )
    scorer_model: ... = factory_field(
        gr.Textbox,
        label='Path to scorer model',
    )
    optimiser: ... = factory_field(
        gr.Dropdown,
        label='Optimiser',
        choices=['bayes', 'tpe'],
        value='bayes',
        elem_id='bayesian_merger_optimiser_dropdown',
    )
    batch_count: ... = factory_field(
        gr.Number,
        label='Batch count',
        value=10,
    )
    init_points: ... = factory_field(
        gr.Number,
        label='Initialization points',
        value=10,
    )
    iterations: ... = factory_field(
        gr.Number,
        label='Iterations',
        value=10,
    )
    scorer_method: ... = factory_field(
        gr.Dropdown,
        label='Scorer method',
        choices=[
            "chad",
            "laion",
            "aes",
            "cafe_aesthetic",
            "cafe_style",
            "cafe_waifu",
        ],
        value='chad',
    )
    save_model: ... = factory_field(
        gr.Checkbox,
        label='Save best model',
        value=False,
    )
    save_model_format: ... = factory_field(
        gr.Dropdown,
        label='Model format',
        choices=['safetensors', 'ckpt'],
        value='safetensors',
    )
    save_model_precision: ... = factory_field(
        gr.Dropdown,
        label='Model precision',
        choices=['16', '32'],
        value='16',
    )

    def __post_init__(self):
        self.start_optimiser_button = gr.Button(
            value='Start Optimizer',
            variant='primary',
        )
        self.message = gr.Textbox(
            label='Message',
            interactive=False,
        )

        with gr.Blocks() as self.root:
            self.rearrange_components()
            self.connect_events()

    def get_webui_tab(self):
        return self.root, 'Bayesian Merger', 'bayesian_merger'

    def connect_events(self):
        self.start_optimiser_button.click(
            fn=on_start_optimise,
            inputs=[getattr(self, self_field.name) for self_field in fields(self)],
            outputs=[self.message],
        )

    def rearrange_components(self):
        self.api_url.render()
        self.model_a.render()
        self.model_b.render()
        self.device.render()
        self.payloads_dir.render()
        self.scorer_model.render()
        self.scorer_method.render()
        self.optimiser.render()
        self.batch_count.render()
        self.init_points.render()
        self.iterations.render()
        self.save_model.render()
        self.save_model_format.render()
        self.save_model_precision.render()
        self.start_optimiser_button.render()
        self.message.render()


def on_start_optimise(
    api_url: str,
    model_a: str | list,
    model_b: str | list,
    device: str,
    payloads_dir: str,
    scorer_model: str,
    optimiser: str,
    batch_count: int,
    init_points: int,
    iterations: int,
    scorer_method: str,
    save_model: bool,
    save_model_format: str,
    save_model_precision: str,
):
    if not model_a or not model_b:
        return 'Error: models A and B need to be selected'

    clip_skip = shared.opts.CLIP_stop_at_last_layers - 1
    cli_args = [
        sys.executable, 'bayesian_merger.py',
        '--url', api_url,
        '--model_a', model_a,
        '--model_b', model_b,
        '--skip_position_ids', str(clip_skip),
        '--device', device,
        '--payloads_dir', payloads_dir,
        '--scorer_model_dir', str(Path(scorer_model).parent.resolve()),
        '--optimiser', optimiser,
        '--batch_size', str(batch_count),
        "--init_points", str(init_points),
        "--n_iters", str(iterations),
        "--scorer_method", scorer_method,
        "--scorer_model_name", Path(scorer_model).name,
    ]

    if save_model:
        cli_args.extend([
            "--save_best",
            "--best_format", save_model_format,
            "--best_precision", save_model_precision,
        ])
    else:
        cli_args.append("--no_save_best")

    script_root = Path(__file__).parent.parent.resolve()
    process = subprocess.Popen(cli_args, cwd=script_root, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
