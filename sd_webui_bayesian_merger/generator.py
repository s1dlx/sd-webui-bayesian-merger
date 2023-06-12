from sd_webui_bayesian_merger.sharer import ModelSharer
import base64
import io
from dataclasses import dataclass
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
        with ModelSharer(theta, owner=True) as sharer:
            for k, v in tqdm(list(theta.items()), desc="move model to shared memory"):
                shapes[k] = v.shape, str(v.dtype)[str(v.dtype).find('.') + 1:]
                sharer.serialize(k)
                del theta[k], v

            option_payload = {
                "model_shapes": shapes,
                "model_a": str(model_a),
            }

            print("Loading merged model")
            r = requests.post(
                url=f"{self.url}/bbwm/load-shared-model",
                json=option_payload,
            )
            r.raise_for_status()

    def refresh_models(self) -> None:
        r = requests.post(url=f"{self.url}/sdapi/v1/refresh-checkpoints")
        r.raise_for_status()

    def list_models(self) -> List[Tuple[str, str]]:
        r = requests.get(url=f"{self.url}/sdapi/v1/sd-models")
        r.raise_for_status()

        return [(m["title"], m["model_name"]) for m in r.json()]
