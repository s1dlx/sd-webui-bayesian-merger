import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional

from modules import shared, sd_models
from dataclasses import dataclass, fields, field
import gradio as gr
from sd_webui_bayesian_merger import cli_args


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
        choices=cli_args.Choices.device,
        value=cli_args.Defaults.device,
    )
    payloads_dir: ... = factory_field(
        gr.Textbox,
        label="Payloads directory",
        placeholder=str(cli_args.Defaults.payloads_dir),
    )
    wildcards_dir: ... = factory_field(
        gr.Textbox,
        label="Wildcards directory",
        placeholder=str(cli_args.Defaults.wildcards_dir),
    )
    scorer_model: ... = factory_field(
        gr.Textbox,
        label="Path to scorer model",
        placeholder=str(cli_args.Defaults.scorer_model_dir / cli_args.Defaults.scorer_model_name)
    )
    optimiser: ... = factory_field(
        gr.Dropdown,
        label="Optimiser",
        choices=cli_args.Choices.optimiser,
        value=cli_args.Defaults.optimiser,
    )
    batch_size: ... = factory_field(
        gr.Number,
        label="Batch size",
        value=cli_args.Defaults.batch_size,
    )
    init_points: ... = factory_field(
        gr.Number,
        label="Initialization points",
        value=cli_args.Defaults.init_points,
    )
    n_iters: ... = factory_field(
        gr.Number,
        label="Iterations",
        value=cli_args.Defaults.n_iters,
    )
    scorer_method: ... = factory_field(
        gr.Dropdown,
        label="Scorer method",
        choices=cli_args.Choices.scorer_method,
        value=cli_args.Defaults.scorer_method,
    )
    save_best: ... = factory_field(
        gr.Checkbox,
        label="Save best model",
        value=cli_args.Defaults.save_best,
    )
    best_format: ... = factory_field(
        gr.Dropdown,
        label="Model format",
        choices=cli_args.Choices.best_format,
        value=cli_args.Defaults.best_format,
    )
    best_precision: ... = factory_field(
        gr.Dropdown,
        label="Model precision",
        choices=cli_args.Choices.best_precision,
        value=cli_args.Defaults.best_precision,
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
    batch_size: float,
    init_points: float,
    n_iters: float,
    scorer_method: str,
    save_best: bool,
    best_format: str,
    best_precision: str,
) -> str:
    if not model_a or not model_b:
        return "Error: models A and B need to be selected"

    clip_skip = shared.opts.CLIP_stop_at_last_layers - 1
    script_args = [
        sys.executable, "bayesian_merger.py",
        "--url", api_url,
        "--model_a", get_model_absolute_path(model_a),
        "--model_b", get_model_absolute_path(model_b),
        "--skip_position_ids", str(clip_skip),
        "--device", device,
        "--batch_size", str(int(batch_size)),
        "--init_points", str(int(init_points)),
        "--n_iters", str(int(n_iters)),
        "--scorer_method", scorer_method,
        "--optimiser", optimiser,
    ]

    if payloads_dir:
        script_args += ["--payloads_dir", payloads_dir]

    if payloads_dir:
        script_args += ["--wildcards_dir", wildcards_dir]

    if scorer_model:
        script_args += [
            "--scorer_model_dir", str(Path(scorer_model).parent.resolve()),
            "--scorer_model_name", Path(scorer_model).name,
        ]

    if save_best:
        script_args += [
            "--save_best",
            "--best_format", best_format,
            "--best_precision", best_precision,
        ]
    else:
        script_args += ["--no_save_best"]

    script_root = cli_args.extension_dir

    print(script_args)
    process = subprocess.Popen(script_args, cwd=script_root, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()


def get_model_absolute_path(model_name: str) -> Optional[Path]:
    model_name = sd_models.checkpoint_alisases[model_name].name
    model_path = Path(sd_models.model_path) / model_name
    if model_path.exists():
        return model_path

    if shared.cmd_opts.ckpt is not None and shared.cmd_opts.ckpt.endswith(model_name):
        return Path(shared.cmd_opts.ckpt)

    if shared.cmd_opts.ckpt_dir is not None:
        model_path = Path(shared.cmd_opts.ckpt_dir) / model_name
        if model_path.exists():
            return model_path

    return None
