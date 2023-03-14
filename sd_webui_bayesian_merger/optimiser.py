import os

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import json
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.merger import Merger
from sd_webui_bayesian_merger.scorer import Scorer
from sd_webui_bayesian_merger.artist import draw_unet

PathT = os.PathLike | str


@dataclass
class BayesianOptimiser:
    url: str
    batch_size: int
    model_a: PathT
    model_b: PathT
    device: str
    payloads_dir: PathT
    wildcards_dir: PathT
    scorer_model_dir: PathT
    init_points: int
    n_iters: int
    skip_position_ids: int

    def __post_init__(self):
        self.generator = Generator(self.url, self.batch_size)
        self.merger = Merger(
            self.model_a,
            self.model_b,
            self.device,
            self.skip_position_ids,
        )
        self.scorer = Scorer(self.scorer_model_dir, self.device)
        self.prompter = Prompter(self.payloads_dir, self.wildcards_dir)
        self.start_logging()
        self.iteration = 0

    def start_logging(self):
        log_path = Path("logs", f"{self.merger.output_file.stem}.json")
        self.logger = JSONLogger(path=str(log_path))

    def sd_target_function(self, **params):
        self.iteration += 1
        print(f'Iteration: {self.iteration}')

        weights = [params[f"block_{i}"] for i in range(25)]
        base_alpha = params["base_alpha"]

        self.merger.create_model_out_name(self.iteration)
        self.merger.merge(
            weights,
            base_alpha,
        )
        self.merger.remove_previous_ckpt(self.iteration)

        # TODO: is this forcing the model load despite the same name?
        self.generator.switch_model(self.merger.model_out_name)

        # generate images
        images = []
        for payload in tqdm(self.prompter.render_payloads()):
            images.extend(self.generator.batch_generate(payload))

        # score images
        scores = self.scorer.batch_score(images)

        # spit out a single value for optimisation
        return self.scorer.average_score(scores)

    def optimise(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        pbounds = {f"block_{i}": (0.0, 1.0) for i in range(25)}
        pbounds["base_alpha"] = (0.0, 1.0)

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=1,
        )

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)

        self.optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iters,
        )

    def postprocess(self) -> None:
        for i, res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \n\t{res}")

        print(self.optimizer.max)

        img_path = Path("logs", f"{self.merger.output_file.stem}.png")
        scores = parse_scores(self.optimizer.res)
        convergence_plot(scores, figname=img_path)

        unet_path = Path("logs", f"{self.merger.output_file.stem}-unet.png")
        best_weights = self.optimizer.max
        best_base_alpha, best_weights = parse_params(self.optimizer.max["params"])
        draw_unet(
            best_base_alpha,
            best_weights,
            model_a=Path(self.model_a).stem,
            model_b=Path(self.model_b).stem,
            figname=unet_path,
        )


def load_log(log: PathT) -> List[Dict]:
    iterations = []
    with open(log, "r") as j:
        while True:
            try:
                iteration = next(j)
            except StopIteration:
                break

            iterations.append(json.loads(iteration))

    return iterations


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]


def parse_params(params: Dict) -> Tuple[float, List[float]]:
    weights = [params[f"block_{i}"] for i in range(25)]
    base_alpha = params["base_alpha"]
    return base_alpha, weights


def maxwhere(l: List[float]) -> Tuple[int, float]:
    m = 0
    mi = -1
    for i, v in enumerate(l):
        if v > m:
            m = v
            mi = i
    return mi, m


def convergence_plot(scores: List[float], figname: PathT = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(scores)

    max_i, max_score = maxwhere(scores)
    plt.plot(max_i, max_score, "or")

    plt.xlabel("iterations")
    plt.ylabel("score")

    sns.despine()

    if figname:
        plt.title(figname.name)
        plt.savefig(figname)
