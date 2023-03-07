from pathlib import Path

import click

from sd_webui_bayesian_merger.optimiser import BayesianOptimiser


@click.command()
@click.option("--url", type=str, default="http://127.0.0.1:7860")
@click.option("--batch_size", type=int, default=1)
@click.option("--model_a", type=click.Path(exists=True), required=True)
@click.option("--model_b", type=click.Path(exists=True), required=True)
@click.option("--model_out", type=click.Path())
@click.option("--device", type=str, default="cpu")
@click.option(
    "--payloads_dir", type=click.Path(exists=True), default=Path("payloads").absolute()
)
@click.option(
    "--wildcards_dir",
    type=click.Path(exists=True),
    default=Path("wildcards").absolute(),
)
@click.option(
    "--scorer_model_path",
    type=click.Path(exists=True),
    default=Path("models").absolute(),
)
@click.option("--init_points", type=int, default=1)
@click.option("--n_iters", type=int, default=1)
def main(*args, **kwargs) -> None:
    print(kwargs)
    bo = BayesianOptimiser(*args, **kwargs)
    bo.optimise()


if __name__ == "__main__":
    main()
