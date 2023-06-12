import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple
import requests
from PIL import Image


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

    def list_models(self) -> List[Tuple[str, str]]:
        r = requests.get(url=f"{self.url}/sdapi/v1/sd-models")
        r.raise_for_status()

        return [(m["title"], m["model_name"]) for m in r.json()]
