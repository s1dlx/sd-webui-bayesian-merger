from sd_webui_bayesian_merger.webui import load_model_weights
from modules import script_callbacks, shared
from multiprocessing import shared_memory
from functools import reduce
from typing import Optional, Dict, Tuple, List
import operator
import fastapi
import gradio as gr
import torch


def on_app_started(_gui: Optional[gr.Blocks], api: fastapi.FastAPI):
    @api.post("/bbwm/load-shm-model")
    async def detect(
        model_shapes: Dict[str, Tuple[List[int], str]] = fastapi.Body(title="theta shapes by key"),
        model_a: str = fastapi.Body(None, title="path to reference checkpoint. used for VAE fallback and the like"),
    ):
        if model_a is None:
            model_a = shared.opts.data['sd_model_checkpoint']

        theta = {}
        try:
            memory = shared_memory.SharedMemory(create=False, name=f"bbwm-model-bytes")
            offset = 0
            for k, (shape, dtype) in model_shapes.items():
                dtype = getattr(torch, dtype)
                count = reduce(operator.mul, shape, 1)
                theta[k] = torch.frombuffer(
                    offset=offset,
                    buffer=memory.buf,
                    count=count,
                    dtype=dtype,
                ).reshape(shape)
                offset = align_offset(offset + count * theta[k].element_size())

            load_model_weights(theta, model_a)

        finally:
            memory = locals().get('memory', None)
            if memory is not None:
                memory.close()


script_callbacks.on_app_started(on_app_started)


def align_offset(offset: int) -> int:
    return offset + (8 - offset % 8) % 8
