import requests
import base64
import io

from PIL import Image

from functools import partial

from bayes_opt import BayesianOptimization


def gen(url: str, payload: dict) -> Image:
    response = requests.post(
        url=f"{url}/sdapi/v1/txt2img",
        json=payload,
    )

    # TODO: handle bad response
    r_json = response.json()
    img = r_json["images"][0]

    return Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))


def gen_batch(url: str, payload: dict, batch_size: int) -> [Image]:
    # TODO: tqdm?
    return [gen(url, payload) for _ in range(batch_size)]


def make_payload(
    prompt: str = "",
    neg_prompt: str = "",
    seed: int = -1,
    steps: int = 1,
    cfg: int = 7,
    width: int = 512,
    height: int = 512,
    sampler: str = "Euler",
) -> dict:
    return {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg,
        "width": width,
        "height": height,
        "sampler_index": sampler,
    }


def score_image(img: Image) -> float:
    # TODO: do something
    return 0.0


def average_score(scores: [float]) -> float:
    # TODO: do something else?
    return sum(scores) / len(scores)


def switch_model(url: str, ckpt: str) -> None:
    option_payload = {
        "sd_model_checkpoint": ckpt,
    }

    # TODO: do something with the response?
    response = requests.post(
        url=f"{url}/sdapi/v1/options",
        json=option_payload,
    )
    print(response)


def sd_target_function(
    url: str,
    batch_size: int,
    payloads: [dict],
    model_a,
    model_b,
    **params,
):
    # TODO: use params to merge models
    new_mix = ""
    # hijack `merge` from
    # https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/blob/8a62a753e791a75273863dd04958753f0df7532f/scripts/mbw/merge_block_weighted.py#L30

    switch_model(url, new_mix)

    images = []
    for payload in payloads:
        images.extend(gen_batch(url, payload, batch_size))

    scores = []
    for image in images:
        scores.append(score_image(image))

    return average_score(scores)


def opt_run(
    url: str,
    batch_size: int,
    payloads: [dict],
    model_a,
    model_b,
    init_points: int,
    n_iter: int,
) -> BayesianOptimization:

    partial_sd_target_function = partial(
        sd_target_function, url, batch_size, payloads, model_a, model_b
    )

    # TODO: init UNET blocks
    pbounds = {}

    # TODO: what if we want to optimise only certain blocks?

    optimizer = BayesianOptimization(
        f=partial_sd_target_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    return optimizer


def postprocess(optmizer:BayesianOptimization) -> None:
    # TODO: analyse the results
    return
