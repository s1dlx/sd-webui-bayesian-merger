from pathlib import Path

import hydra
from omegaconf import DictConfig

from sd_webui_bayesian_merger import BayesOptimiser, TPEOptimiser, ATPEOptimiser
from sd_webui_bayesian_merger.artist import draw_unet


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg["scorer_method"] == "laion":
        cfg["scorer_model_name"] = "laion-sac-logos-ava-v2.safetensors"
    elif cfg["scorer_method"] == "aes":
        cfg["scorer_model_name"] = "aes-B32-v0.safetensors"
    elif cfg["scorer_method"].startswith("cafe"):
        cfg["scorer_model_name"] = ""

    if cfg["draw_unet_weights"] and cfg["draw_unet_base_alpha"]:
        weights = list(map(float, cfg["draw_unet_weights"].split(",")))
        draw_unet(
            cfg["draw_unet_base_alpha"],
            weights,
            Path(cfg["model_a"]).stem,
            Path(cfg["model_b"]).stem,
            "./unet.png",
        )
        return

    if cfg["optimiser"] == "bayes":
        cls = BayesOptimiser
    elif cfg["optimiser"] == "tpe":
        cls = TPEOptimiser
    elif cfg["optimiser"] == "atpe":
        cls = ATPEOptimiser
    else:
        exit(f"Invalid optimiser:{cfg['optimiser']}")

    bo = cls(cfg)
    bo.optimise()
    bo.postprocess()


if __name__ == "__main__":
    main()
