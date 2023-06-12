from sd_webui_bayesian_merger.webui import load_model_weights
from sd_webui_bayesian_merger.sharer import ModelSharer
from modules import script_callbacks, shared
from typing import Optional, Dict, Tuple, List
import fastapi
import gradio as gr
import torch


def on_app_started(_gui: Optional[gr.Blocks], api: fastapi.FastAPI):
    @api.post("/bbwm/load-shared-model")
    async def detect(
        model_shapes: Dict[str, Tuple[List[int], str]] = fastapi.Body(title="theta shapes by key"),
        model_a: str = fastapi.Body(None, title="path to reference checkpoint. used for VAE fallback and the like"),
    ):
        if model_a is None:
            model_a = shared.opts.data['sd_model_checkpoint']

        theta = {}
        with ModelSharer(theta, owner=False) as sharer:
            for k, (shape, dtype) in model_shapes.items():
                dtype = getattr(torch, dtype)
                theta[k] = sharer.deserialize(shape, dtype)

            load_model_weights(theta, model_a)


script_callbacks.on_app_started(on_app_started)


def align_offset(offset: int) -> int:
    return offset + (8 - offset % 8) % 8
