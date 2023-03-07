import requests
import io
import base64

from PIL import Image


class Generator:
    # TODO: defaultclass?
    def __init__(self, url: str, batch_size: int):
        self.url = url
        self.batch_size = batch_size

    def generate(self, payload: dict) -> Image:
        response = requests.post(
            url=f"{self.url}/sdapi/v1/txt2img",
            json=payload,
        )

        # TODO: handle bad response
        r_json = response.json()
        img = r_json["images"][0]

        return Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))

    def batch_generate(self, payload: dict) -> [Image]:
        # TODO: tqdm?
        return [self.generate(payload) for _ in range(self.batch_size)]

    def switch_model(self, ckpt: str) -> None:
        option_payload = {
            "sd_model_checkpoint": ckpt.name,
        }

        # TODO: do something with the response?
        response = requests.post(
            url=f"{self.url}/sdapi/v1/options",
            json=option_payload,
        )
        print(response)
