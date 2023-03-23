from pathlib import Path
from configparser import ConfigParser

import click

from sd_webui_bayesian_merger import BayesOptimiser, TPEOptimiser, DefaultCliArgs
from sd_webui_bayesian_merger.artist import draw_unet


def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    try:
        options = dict(cfg["options"])
    except KeyError:
        options = {}
    ctx.default_map = options


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    default=DefaultCliArgs.config,
    callback=configure,
    is_eager=True,
    expose_value=False,
    help="Read option defaults from the specified INI file",
    show_default=True,
)
@click.option(
    "--url",
    type=str,
    help="where webui api is running, by default http://127.0.0.1:7860",
    default=DefaultCliArgs.url,
)
@click.option(
    "--model_a",
    type=click.Path(exists=True),
    required=True,
    help="absolute path to first model",
)
@click.option(
    "--model_b",
    type=click.Path(exists=True),
    required=True,
    help="absolute path to second model",
)
@click.option(
    "--skip_position_ids",
    type=int,
    default=DefaultCliArgs.skip_position_ids,
    help="clip skip, default 0",
)
@click.option(
    "--device",
    type=str,
    default=DefaultCliArgs.device,
    help='where to merge models and score images, default and recommended "cpu"',
)
@click.option(
    "--payloads_dir",
    type=click.Path(exists=True),
    default=DefaultCliArgs.payloads_dir,
    help="absolute path to payloads directory",
)
@click.option(
    "--wildcards_dir",
    type=click.Path(exists=True),
    default=DefaultCliArgs.wildcards_dir,
    help="absolute path to wildcards directory",
)
@click.option(
    "--scorer_model_dir",
    type=click.Path(exists=True),
    default=DefaultCliArgs.scorer_model_dir,
    help="absolute path to scorer models directory",
)
@click.option(
    "--optimiser",
    type=click.Choice(["bayes", "tpe"]),
    default=DefaultCliArgs.optimiser,
    help="optimiser, bayes (default) or tpe",
)
@click.option(
    "--batch_size",
    type=int,
    default=DefaultCliArgs.batch_size,
    help="number of images to generate for each payload",
)
@click.option(
    "--init_points",
    type=int,
    default=DefaultCliArgs.init_points,
    help="exploratory/warmup phase sample size",
)
@click.option(
    "--n_iters",
    type=int,
    default=DefaultCliArgs.n_iters,
    help="exploitation/optimisation phase sample size",
)
@click.option(
    "--save_imgs/--no_save_imgs",
    default=DefaultCliArgs.save_imgs,
    help="save all the generated images",
)
@click.option(
    "--scorer_method",
    type=click.Choice(
        [
            "chad",
            "laion",
            "aes",
            "cafe_aesthetic",
            "cafe_style",
            "cafe_waifu",
        ]
    ),
    default=DefaultCliArgs.scorer_method,
    help="scoring methods, chad (default)",
)
@click.option(
    "--scorer_model_name",
    type=click.Choice(
        [
            "sac+logos+ava1-l14-linearMSE.pth",  # chad
            "ava+logos-l14-linearMSE.pth",
            "ava+logos-l14-reluMSE.pth",
        ]
    ),
    default=DefaultCliArgs.scorer_model_name,
    help="scoring model options for chad method",
)
@click.option(
    "--save_best/--no_save_best",
    default=DefaultCliArgs.save_best,
    help="save best model across the whole run",
)
@click.option(
    "--best_format",
    type=click.Choice(["safetensors", "ckpt"]),
    default=DefaultCliArgs.best_format,
    help="best model saving format, either safetensors (default) or ckpt",
)
@click.option(
    "--best_precision",
    type=click.Choice(["16", "32"]),
    default=DefaultCliArgs.best_precision,
    help="best model saving precision, either 16 (default) or 32 bit",
)
@click.option(
    "--draw_unet_weights",
    type=str,
    help="list of weights for drawing mode",
    default=DefaultCliArgs.draw_unet_weights,
)
@click.option(
    "--draw_unet_base_alpha",
    type=float,
    default=DefaultCliArgs.draw_unet_base_alpha,
    help="base alpha value for drawing mode",
)
def main(*args, **kwargs) -> None:
    if kwargs["scorer_method"] == "laion":
        kwargs["scorer_model_name"] = "laion-sac-logos-ava-v2.safetensors"
    elif kwargs["scorer_method"] == "aes":
        kwargs["scorer_model_name"] = "aes-B32-v0.safetensors"
    elif kwargs["scorer_method"].startswith("cafe"):
        kwargs["scorer_model_name"] = ""

    if kwargs["draw_unet_weights"] and kwargs["draw_unet_base_alpha"]:
        weights = list(map(float, kwargs["draw_unet_weights"].split(",")))
        draw_unet(
            kwargs["draw_unet_base_alpha"],
            weights,
            Path(kwargs["model_a"]).stem,
            Path(kwargs["model_b"]).stem,
            "./unet.png",
        )
    else:
        kwargs.pop("draw_unet_weights")
        kwargs.pop("draw_unet_base_alpha")
        optimiser = kwargs.pop("optimiser")
        if optimiser == "bayes":
            cls = BayesOptimiser
        elif optimiser == "tpe":
            cls = TPEOptimiser
        else:
            exit(f"Invalid optimiser:{optimiser}")
        bo = cls(*args, method=optimiser, **kwargs)
        bo.optimise()
        bo.postprocess()


if __name__ == "__main__":
    main()
