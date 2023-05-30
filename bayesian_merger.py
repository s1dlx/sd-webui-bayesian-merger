from pathlib import Path

import hydra
from omegaconf import DictConfig

from sd_webui_bayesian_merger import ATPEOptimiser, BayesOptimiser, TPEOptimiser
from sd_webui_bayesian_merger.artist import draw_unet


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
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
