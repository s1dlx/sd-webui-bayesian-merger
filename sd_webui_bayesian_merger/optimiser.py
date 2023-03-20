import os
from abc import abstractmethod

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import json
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from bayes_opt.logger import JSONLogger

from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.merger import Merger, NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.scorer import AestheticScorer

PathT = os.PathLike


@dataclass
class Optimiser:
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
    best_format: str
    best_precision: int
    save_best: bool
    method: str
    scorer_method: str
    scorer_model_name: str

    def __post_init__(self):
        self.generator = Generator(self.url, self.batch_size)
        self.init_merger()
        self.init_scorer()
        self.prompter = Prompter(self.payloads_dir, self.wildcards_dir)
        self.start_logging()
        self.iteration = 0

    def init_merger(self):
        self.merger = Merger(
            self.model_a,
            self.model_b,
            self.device,
            self.skip_position_ids,
            self.best_format,
            self.best_precision,
        )

    def init_scorer(self):
        if self.scorer_method == "chad":
            self.scorer = AestheticScorer(
                self.scorer_method,
                self.scorer_model_dir,
                self.scorer_model_name,
                self.device,
            )
        else:
            raise NotImplementedError(
                f"{self.scorer_name} scorer not implemented",
            )

    def _cleanup(self):
        # clean up and remove the last merge
        self.merger.remove_previous_ckpt(self.iteration + 1)

    def start_logging(self):
        log_path = Path("logs", f"{self.merger.output_file.stem}-{self.method}.json")
        self.logger = JSONLogger(path=str(log_path))

    def sd_target_function(self, **params):
        self.iteration += 1

        if self.iteration == 1:
            print("\n" + "-" * 10 + " warmup " + "-" * 10 + ">")
        elif self.iteration == self.init_points + 1:
            print("\n" + "-" * 10 + " optimisation " + "-" * 10 + ">")

        it_type = "warmup" if self.iteration <= self.init_points else "optimisation"
        print(f"\n{it_type} - Iteration: {self.iteration}")

        weights = [params[f"block_{i}"] for i in range(NUM_TOTAL_BLOCKS)]
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
        for payload in tqdm(
            self.prompter.render_payloads(),
            desc="Batches generation",
        ):
            images.extend(self.generator.batch_generate(payload))

        # score images
        scores = self.scorer.batch_score(images)

        # spit out a single value for optimisation
        avg_score = self.scorer.average_score(scores)
        print(f"Score: {avg_score}")

        return avg_score

    @abstractmethod
    def optimise(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")


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


def maxwhere(l: List[float]) -> Tuple[int, float]:
    m = 0
    mi = -1
    for i, v in enumerate(l):
        if v > m:
            m = v
            mi = i
    return mi, m


def convergence_plot(scores: List[float], figname: Path = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(scores)

    max_i, max_score = maxwhere(scores)
    plt.plot(max_i, max_score, "or")

    plt.xlabel("iterations")
    plt.ylabel("score")

    sns.despine()

    if figname:
        figname.parent.mkdir(exist_ok=True)
        plt.title(figname.stem)
        print("Saving fig to:", figname)
        plt.savefig(figname)
