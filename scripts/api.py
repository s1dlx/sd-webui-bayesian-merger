from sd_webui_bayesian_merger.webui import load_model_weights
from modules import script_callbacks, shared
from multiprocessing import shared_memory
from functools import reduce
from typing import Optional
import operator
import fastapi
import gradio as gr
import torch


def on_app_started(_gui: Optional[gr.Blocks], api: fastapi.FastAPI):
    @api.post("/bbwm/upload-model")
    async def detect(
        model: dict = fastapi.Body(title="serialized theta"),
        model_a: str = fastapi.Body(None, title="path to reference checkpoint. used for VAE fallback and the like"),
    ):
        if model_a is None:
            model_a = shared.opts.data['sd_model_checkpoint']

        key_memories = {}
        try:
            for k, shape in list(model.items()):
                key_memories[k] = shared_memory.SharedMemory(create=False, name=f"bbwm-{k}")
                model[k] = torch.frombuffer(
                    buffer=key_memories[k].buf,
                    count=reduce(operator.mul, shape, 1),
                    dtype=torch.float,
                ).reshape(shape)

            load_model_weights(model, model_a)

        finally:
            for mem in key_memories.values():
                mem.close()


script_callbacks.on_app_started(on_app_started)
