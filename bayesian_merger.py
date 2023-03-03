from sd_webui_bayesian_merger.merger import BayesianOptimisationMerger


def build_payload(
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


# TODO: clip args
if __name__ == "__main__":
    url = ""
    batch_size = ""
    model_a = ""
    model_b = ""
    device = ""
    output_file = ""
    bom = BayesianOptimisationMerger(
        url, batch_size, model_a, model_b, device, output_file
    )

    # TODO: where are the prompts coming from? add a Prompter class
    payloads = []

    # TODO: compute total number of generations as batch_size * len(payloads)
    init_points = 30
    n_iters = 20
    bom.optimise(payloads, init_points, n_iters)
