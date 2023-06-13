import torch
from fastapi import HTTPException
from sd_meh.merge import merge_models, NUM_TOTAL_BLOCKS
from modules import script_callbacks, sd_models, shared
from typing import Optional, List, Dict
import fastapi
import safetensors.torch
import gradio as gr
from pathlib import Path


def on_app_started(_gui: Optional[gr.Blocks], api: fastapi.FastAPI):
    @api.post("/bbwm/merge-models")
    async def merge_models_api(
        destination: str = fastapi.Body(title="Destination to save the merge result. Pass 'load' to load it in memory instead"),
        merge_method: str = fastapi.Body(title="Merge method"),
        model_a: str = fastapi.Body(title="Path to model A"),
        model_b: str = fastapi.Body(title="Path to model B"),
        model_c: str = fastapi.Body(None, title="Path to model C"),
        base_alpha: float = fastapi.Body(0.5, title="Base alpha"),
        base_beta: float = fastapi.Body(0.5, title="Base beta"),
        alpha: List[float] = fastapi.Body(None, title="Alpha"),
        beta: List[float] = fastapi.Body(None, title="Beta"),
        precision: int = fastapi.Body(16, title="Precision"),
        weights_clip: bool = fastapi.Body(False, title="Weights clip"),
        re_basin: bool = fastapi.Body(False, title="Git re-basin"),
        re_basin_iterations: int = fastapi.Body(1, title="Git re-basin iterations"),
        device: str = fastapi.Body("cpu", title="Device used to load models"),
        prune: bool = fastapi.Body(False, title="Prune model during merge"),
    ):
        if not alpha:
            alpha = [base_alpha] * NUM_TOTAL_BLOCKS
        if not beta:
            beta = [base_beta] * NUM_TOTAL_BLOCKS

        if destination != "load":
            destination = Path(destination)
            if not destination.is_absolute() or not destination.exists():
                raise HTTPException(422, "Destination path must be an absolute path")

        sd_models.unload_model_weights()
        merged = merge_models(
            models={
                "model_a": model_a,
                "model_b": model_b,
                **({"model_c": model_c} if model_c else {}),
            },
            weights={
                "alpha": alpha,
                "beta": beta,
            },
            bases={
                "alpha": base_alpha,
                "beta": base_beta,
            },
            merge_mode=merge_method,
            precision=precision,
            weights_clip=weights_clip,
            re_basin=re_basin,
            iterations=re_basin_iterations,
            device=device,
            prune=prune,
        )
        if not isinstance(merged, dict):
            merged = merged.to_dict()

        if destination == "load":
            checkpoint_info = sd_models.checkpoint_alisases[Path(model_a).name]
            sd_models.load_model(checkpoint_info, merged)
            return

        save_model(merged, destination)
        shared.refresh_checkpoints()


script_callbacks.on_app_started(on_app_started)


def save_model(merged: Dict, path: Path):
    print(f"Saving {path}")
    if path.suffix == ".safetensors":
        safetensors.torch.save_file(
            merged,
            path.with_suffix(""),
            metadata={"format": "pt"},
        )
    else:
        torch.save({"state_dict": merged}, path.with_suffix(""))
