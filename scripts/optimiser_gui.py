import subprocess
import sys
from pathlib import Path
from typing import Tuple

from modules import shared
from dataclasses import dataclass, fields, field
import gradio as gr
import torch
from sd_webui_bayesian_merger import DefaultCliArgs

# personal notes
### wildcards. use extensions instead?
### draw unet merge chart


def factory_field(cls, **kwargs):
    return field(default_factory=lambda: cls(**kwargs))


@dataclass
class OptimiserGui:
    api_url: ... = factory_field(
        gr.Textbox,
        visible=False,
        elem_id="bayesian_merger_api_url",
    )
    model_a: ... = factory_field(
        gr.Dropdown,
        label="Model A",
        choices=shared.list_checkpoint_tiles(),
    )
    model_b: ... = factory_field(
        gr.Dropdown,
        label="Model B",
        choices=shared.list_checkpoint_tiles(),
    )
    device: ... = factory_field(
        gr.Dropdown,
        label="Merge on device",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        value=DefaultCliArgs.device,
    )
    payloads_dir: ... = factory_field(
        gr.Textbox,
        label="Payloads directory",
        placeholder=str(DefaultCliArgs.payloads_dir),
    )
    wildcards_dir: ... = factory_field(
        gr.Textbox,
        label="Wildcards directory",
        placeholder=str(DefaultCliArgs.wildcards_dir),
    )
    scorer_model: ... = factory_field(
        gr.Textbox,
        label="Path to scorer model",
        placeholder=str(DefaultCliArgs.scorer_model_dir / DefaultCliArgs.scorer_model_name)
    )
    optimiser: ... = factory_field(
        gr.Dropdown,
        label="Optimiser",
        choices=["bayes", "tpe"],
        value=DefaultCliArgs.device,
    )
    batch_size: ... = factory_field(
        gr.Number,
        label="Batch count",
        value=DefaultCliArgs.batch_size,
    )
    init_points: ... = factory_field(
        gr.Number,
        label="Initialization points",
        value=DefaultCliArgs.init_points,
    )
    n_iters: ... = factory_field(
        gr.Number,
        label="Iterations",
        value=DefaultCliArgs.n_iters,
    )
    scorer_method: ... = factory_field(
        gr.Dropdown,
        label="Scorer method",
        choices=[
            "chad",
            "laion",
            "aes",
            "cafe_aesthetic",
            "cafe_style",
            "cafe_waifu",
        ],
        value=DefaultCliArgs.scorer_method,
    )
    save_best: ... = factory_field(
        gr.Checkbox,
        label="Save best model",
        value=DefaultCliArgs.save_best,
    )
    best_format: ... = factory_field(
        gr.Dropdown,
        label="Model format",
        choices=["safetensors", "ckpt"],
        value=DefaultCliArgs.best_format,
    )
    best_precision: ... = factory_field(
        gr.Dropdown,
        label="Model precision",
        choices=["16", "32"],
        value=DefaultCliArgs.best_precision,
    )

    def __post_init__(self):
        self.start_optimiser_button = gr.Button(
            value="Start Optimizer",
            variant="primary",
        )
        self.message = gr.Textbox(
            label="Message",
            interactive=False,
        )

        with gr.Blocks() as self.root:
            self.rearrange_components()
            self.connect_events()

    def get_webui_tab(self) -> Tuple[gr.Blocks, str, str]:
        return self.root, "Bayesian Merger", "bayesian_merger"

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
        self.wildcards_dir.render()
        self.scorer_model.render()
        self.scorer_method.render()
        self.optimiser.render()
        self.batch_size.render()
        self.init_points.render()
        self.n_iters.render()
        self.save_best.render()
        self.best_format.render()
        self.best_precision.render()
        self.start_optimiser_button.render()
        self.message.render()


def on_start_optimise(
    api_url: str,
    model_a: str | list,
    model_b: str | list,
    device: str,
    payloads_dir: str,
    wildcards_dir: str,
    scorer_model: str,
    optimiser: str,
    batch_size: int,
    init_points: int,
    n_iters: int,
    scorer_method: str,
    save_best: bool,
    best_format: str,
    best_precision: str,
) -> str:
    if not model_a or not model_b:
        return "Error: models A and B need to be selected"

    clip_skip = shared.opts.CLIP_stop_at_last_layers - 1
    cli_args = [
        sys.executable, "bayesian_merger.py",
        "--url", api_url,
        "--model_a", model_a,
        "--model_b", model_b,
        "--skip_position_ids", str(clip_skip),
        "--device", device,
        "--batch_size", str(batch_size),
        "--init_points", str(init_points),
        "--n_iters", str(n_iters),
        "--scorer_method", scorer_method,
        "--optimiser", optimiser,
    ]

    if payloads_dir:
        cli_args += ["--payloads_dir", payloads_dir]

    if payloads_dir:
        cli_args += ["--wildcards_dir", wildcards_dir]

    if scorer_model:
        cli_args += [
            "--scorer_model_dir", str(Path(scorer_model).parent.resolve()),
            "--scorer_model_name", Path(scorer_model).name,
        ]

    if save_best:
        cli_args += [
            "--save_best",
            "--best_format", best_format,
            "--best_precision", best_precision,
        ]
    else:
        cli_args += ["--no_save_best"]

    script_root = Path(__file__).parent.parent.resolve()
    process = subprocess.Popen(cli_args, cwd=script_root, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
