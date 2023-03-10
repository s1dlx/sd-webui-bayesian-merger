from pathlib import Path

import click

from sd_webui_bayesian_merger.optimiser import BayesianOptimiser


@click.command()
@click.option(
    "--url",
    type=str,
    help="where webui api is running, by default http://127.0.0.1:7860",
    default="http://127.0.0.1:7860",
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="number of images to generate for each payload",
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
    "--device",
    type=str,
    default="cpu",
    help='where to merge models and score images, default and recommended "cpu"',
)
@click.option(
    "--payloads_dir",
    type=click.Path(exists=True),
    default=Path("payloads").absolute(),
    help="absolute path to payloads directory",
)
@click.option(
    "--wildcards_dir",
    type=click.Path(exists=True),
    default=Path("wildcards").absolute(),
    help="absolute path to wildcards directory",
)
@click.option(
    "--scorer_model_dir",
    type=click.Path(exists=True),
    default=Path("models").absolute(),
    help="absolute path to scorer models directory",
)
@click.option(
    "--init_points", type=int, default=1, help="exploratory phase sample size",
)
@click.option("--n_iters", type=int, default=1, help="exploitation phase sample size",)
def main(*args, **kwargs) -> None:
    bo = BayesianOptimiser(*args, **kwargs)
    bo.optimise()
    bo.postprocess()


if __name__ == "__main__":
    main()
