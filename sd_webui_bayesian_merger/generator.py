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
        title = self.find_title(Path(ckpt).stem)
        
        option_payload = {
            "sd_model_checkpoint": title,
        }

        # TODO: do something with the response?
        response = requests.post(
            url=f"{self.url}/sdapi/v1/options",
            json=option_payload,
        )

    def list_models(self)->[(str, str)]:
        response = requests.get(
            url=f"{self.url}/sdapi/v1/sd-models")
        return [(m['title'], m['model_name']) for m in response.json()]

    def find_title(self, model_name) -> str:
        models = self.list_models()
        for p in models:
            title, name = p
            print(title, '-', name, '-', model_name)
            if name == model_name:
                return title

        raise ValueError(f'model {model_name} not found')
        
