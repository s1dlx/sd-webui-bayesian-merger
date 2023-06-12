import base64
import io
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Tuple

import requests
from PIL import Image
from tqdm import tqdm


@dataclass
class Generator:
    url: str
    batch_size: int

    def generate(self, payload: Dict) -> List[Image.Image]:
        r = requests.post(
            url=f"{self.url}/sdapi/v1/txt2img",
            json=payload,
        )
        r.raise_for_status()

        r_json = r.json()
        images = r_json["images"]

        return [
            Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
            for img in images
        ]

    def unload_model(self) -> None:
        print("Unloading webui checkpoint")
        r = requests.post(
            url=f"{self.url}/sdapi/v1/unload-checkpoint",
        )
        r.raise_for_status()

    def upload_model(self, theta: Dict, model_a: str) -> None:
        shapes = {}
        total_size = 0
        for v in theta.values():
            total_size = align_offset(total_size + v.untyped_storage().nbytes())

        try:
            memory = shared_memory.SharedMemory(create=True, name=f"bbwm-model-bytes", size=total_size)
            offset = 0
            for k, v in tqdm(list(theta.items()), desc="move model to shared memory"):
                next_offset = offset + v.numpy().nbytes
                memory.buf[offset:next_offset] = v.numpy(force=True).tobytes()
                shapes[k] = v.shape, str(v.dtype)[str(v.dtype).find('.') + 1:]
                del theta[k]
                offset = align_offset(next_offset)

            option_payload = {
                "model_shapes": shapes,
                "model_a": str(model_a),
            }

            print("Loading merged model")
            r = requests.post(
                url=f"{self.url}/bbwm/load-shm-model",
                json=option_payload,
            )
            r.raise_for_status()

        finally:
            memory = locals().get('memory', None)
            if memory is not None:
                memory.close()
                memory.unlink()

    def refresh_models(self) -> None:
        r = requests.post(url=f"{self.url}/sdapi/v1/refresh-checkpoints")
        r.raise_for_status()

    def list_models(self) -> List[Tuple[str, str]]:
        r = requests.get(url=f"{self.url}/sdapi/v1/sd-models")
        r.raise_for_status()

        return [(m["title"], m["model_name"]) for m in r.json()]


def align_offset(offset: int) -> int:
    return offset + (8 - offset % 8) % 8
