import requests
import base64
import io

from PIL import Image

payload = {
    "prompt": "car",
    "negative_prompt": "",
    "seed": -1,
    "batch_size": 1, # DO NOT CHANGE!!!
    "steps": 2,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "sampler_index": "Euler",
}

def gen(url: str, payload: dict) -> [Image]:
    response = requests.post(
        url=f"{url}/sdapi/v1/txt2img",
        json=payload,
    )

    r_json = response.json()
    images = []
    for img in r_json["images"]:
        image = Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
        images.append(image)

    return images

