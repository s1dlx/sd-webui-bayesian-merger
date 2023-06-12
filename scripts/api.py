from pathlib import Path

from modules import script_callbacks, sd_models, paths, shared
from typing import Optional, List
import fastapi
import gradio as gr
from sd_meh.merge import merge_models


def on_app_started(_gui: Optional[gr.Blocks], api: fastapi.FastAPI):
    @api.post("/bbwm/merge-models")
    async def detect(
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
        work_device: str = fastapi.Body("cpu", title="Device used to merge models"),
        prune: bool = fastapi.Body(False, title="Prune model during merge")
    ):
        if not alpha:
            alpha = [base_alpha] * 25
        if not beta:
            beta = [base_beta] * 25

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
            work_device=work_device,
            prune=prune,
        ).to_dict()

        checkpoint_info = sd_models.checkpoint_alisases[Path(model_a).stem]
        sd_models.load_model(checkpoint_info, merged)


script_callbacks.on_app_started(on_app_started)
