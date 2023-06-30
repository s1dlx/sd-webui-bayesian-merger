import fastapi
import gradio as gr
import inspect
import re
import safetensors.torch
import torch
from modules import script_callbacks, sd_models, shared
from pathlib import Path
from sd_meh.merge import merge_methods, merge_models, NUM_TOTAL_BLOCKS
from typing import Dict, List, Optional


MEMORY_DESTINATION = "memory"


def on_app_started(_gui: Optional[gr.Blocks], api: fastapi.FastAPI):
    @api.post("/bbwm/merge-models")
    async def merge_models_api(
        destination: str = fastapi.Body(
            title="Destination",
            description=format_multiline_description(f"""
                Path to save the merge result.
                If relative, the merge result will be saved in the directory of model A.
                Pass "{MEMORY_DESTINATION}" to load it in memory instead
            """),
        ),
        unload_before: bool = fastapi.Body(
            False,
            title="Unload before merging",
            description="Unload current model before merging to save memory",
        ),
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
        work_device: str = fastapi.Body(None, title="Device used to merge models"),
        prune: bool = fastapi.Body(False, title="Prune model during merge"),
        threads: int = fastapi.Body(
            1,
            title="Number of threads",
            description="Number of keys to merge simultaneously. Only useful with device='cpu'",
        ),
    ):
        validate_merge_method(merge_method)
        alpha, beta, input_models, weights, bases = normalize_merge_args(
            base_alpha, base_beta,
            alpha, beta,
            model_a, model_b, model_c,
        )

        model_a_info = get_checkpoint_info(Path(model_a))
        load_in_memory = destination == MEMORY_DESTINATION
        if not load_in_memory:
            destination = normalize_destination(destination, model_a_info)

        unload_before = (unload_before or load_in_memory) and shared.sd_model is not None
        if unload_before:
            sd_models.unload_model_weights()

        try:
            merged = merge_models(
                models=input_models,
                weights=weights,
                bases=bases,
                merge_mode=merge_method,
                precision=precision,
                weights_clip=weights_clip,
                re_basin=re_basin,
                iterations=re_basin_iterations,
                device=device,
                work_device=work_device,
                prune=prune,
                threads=threads,
            )
            if not isinstance(merged, dict):
                merged = merged.to_dict()

            if load_in_memory:
                sd_models.load_model(model_a_info, merged)
            else:
                save_model(merged, destination)
                shared.refresh_checkpoints()

        finally:
            if unload_before and (not load_in_memory or shared.sd_model is None):
                sd_models.reload_model_weights()


script_callbacks.on_app_started(on_app_started)


def validate_merge_method(merge_method: str) -> None:
    if merge_method not in dict(inspect.getmembers(merge_methods, inspect.isfunction)).keys():
        raise fastapi.HTTPException(422, "Merge method is not defined")


def normalize_merge_args(base_alpha, base_beta, alpha, beta, model_a, model_b, model_c):
    if not alpha:
        alpha = [base_alpha] * NUM_TOTAL_BLOCKS
    if not beta:
        beta = [base_beta] * NUM_TOTAL_BLOCKS

    input_models = {
        "model_a": model_a,
        "model_b": model_b,
        **({"model_c": model_c} if model_c else {}),
    }
    weights = {
        "alpha": alpha,
        "beta": beta,
    }
    bases = {
        "alpha": base_alpha,
        "beta": base_beta,
    }

    return alpha, beta, input_models, weights, bases


def get_checkpoint_info(path: Path) -> sd_models.CheckpointInfo:
    checkpoint_aliases = getattr(sd_models, "checkpoint_alisases", None)
    if checkpoint_aliases is None: # we are on vlad webui
        checkpoint_aliases = getattr(sd_models, "checkpoint_aliases")

    checkpoint_info = None
    path_parts_len = len(path.parts)
    for i in range(path_parts_len):
        sub_path = Path(*path.parts[path_parts_len-1-i:])
        checkpoint_info = checkpoint_aliases.get(str(sub_path), None)
        if checkpoint_info is not None:
            break

    if checkpoint_info is None:
        raise fastapi.HTTPException(422, "Could not find checkpoint alias for model A")

    return checkpoint_info


def normalize_destination(
    destination: str,
    checkpoint_info: sd_models.CheckpointInfo,
) -> Path:
    destination = Path(destination)
    if not destination.is_absolute():
        destination = destination.relative_to(checkpoint_info.filename)
    if not destination.parent.exists():
        raise fastapi.HTTPException(422, "Destination parent directory does not exist")

    return destination


def save_model(merged: Dict, path: Path):
    print(f"Saving merge to {path}")
    path.unlink(missing_ok=True)
    if path.suffix == ".safetensors":
        safetensors.torch.save_file(
            merged,
            path,
            metadata={"format": "pt"},
        )
    else:
        torch.save({"state_dict": merged}, path)


def format_multiline_description(description: str) -> str:
    return re.sub(r"\s{2,}", " ", description).strip()
