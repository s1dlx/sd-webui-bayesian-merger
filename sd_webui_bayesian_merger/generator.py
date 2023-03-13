from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import requests
import io
import base64

from PIL import Image


@dataclass
class Generator:
    url: str
    batch_size: int

    def generate(self, payload: Dict) -> Image.Image:
        response = requests.post(
            url=f"{self.url}/sdapi/v1/txt2img",
            json=payload,
        )

        r_json = response.json()
        img = r_json["images"][0]

        return Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))

    def batch_generate(self, payload: Dict) -> List[Image.Image]:
        return [self.generate(payload) for _ in range(self.batch_size)]

    def switch_model(self, ckpt: str) -> None:
        self.refresh_models()
        title = self.find_title(Path(ckpt).stem)

        option_payload = {
            "sd_model_checkpoint": title,
        }

        requests.post(
            url=f"{self.url}/sdapi/v1/options",
            json=option_payload,
        )

    def refresh_models(self) -> None:
        requests.post(url=f"{self.url}/sdapi/v1/refresh-checkpoints")

    def list_models(self) -> List[Tuple[str, str]]:
        response = requests.get(url=f"{self.url}/sdapi/v1/sd-models")
        return [(m["title"], m["model_name"]) for m in response.json()]

    def find_title(self, model_name) -> str:
        models = self.list_models()
        for p in models:
            title, name = p
            if name == model_name:
                return title

        raise ValueError(f"model {model_name} not found")
