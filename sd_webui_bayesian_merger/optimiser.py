import os
from abc import abstractmethod
from datetime import datetime

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
from sd_webui_bayesian_merger.artist import draw_unet

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
    save_imgs: bool

    def __post_init__(self):
        self.generator = Generator(self.url, self.batch_size)
        self.init_merger()
        self.start_logging()
        self.init_scorer()
        self.prompter = Prompter(self.payloads_dir, self.wildcards_dir)
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
        if self.scorer_method in [
            "chad",
            "laion",
            "aes",
            "cafe_aesthetic",
            "cafe_style",
            "cafe_waifu",
        ]:
            self.scorer = AestheticScorer(
                self.scorer_method,
                self.scorer_model_dir,
                self.scorer_model_name,
                self.device,
                self.save_imgs,
                self.log_dir,
            )
        else:
            raise NotImplementedError(
                f"{self.scorer_method} scorer not implemented",
            )

    def _cleanup(self):
        # clean up and remove the last merge
        self.merger.remove_previous_ckpt(self.iteration + 1)

    def start_logging(self):
        now = datetime.now()
        str_now = datetime.strftime(now, "%Y-%m-%d-%H-%M-%S")
        h, e, l, _ = self.merger.output_file.stem.split("-")
        dir_name = "-".join([h, e, l])
        self.log_name = f"{dir_name}-{self.method}"
        self.log_dir = Path(
            "logs",
            f"{self.log_name}-{str_now}",
        )
        if not self.log_dir.exists():
            self.log_dir.mkdir()
        log_path = Path(self.log_dir, "log.json")
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
        payloads, paths = self.prompter.render_payloads()
        gen_paths = []
        for i, payload in tqdm(
            enumerate(payloads),
            desc="Batches generation",
        ):
            images.extend(self.generator.batch_generate(payload))
            gen_paths.extend([paths[i]] * self.batch_size)

        # score images
        print("\nScoring")
        scores = self.scorer.batch_score(
            images,
            gen_paths,
            self.iteration,
        )

        # spit out a single value for optimisation
        avg_score = self.scorer.average_score(scores)
        print(f"{'-'*10}\nRun score: {avg_score}")

        print(f"\nrun base_alpha: {base_alpha}")
        print("run weights:")
        print(",".join(list(map(str, weights))))

        return avg_score

    @abstractmethod
    def optimise(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    def plot_and_save(
        self,
        scores: List[float],
        best_base_alpha: float,
        best_weights: List[float],
        minimise: bool,
    ) -> None:
        img_path = Path(
            self.log_dir,
            f"{self.log_name}.png",
        )
        convergence_plot(scores, figname=img_path, minimise=minimise)

        unet_path = Path(
            self.log_dir,
            f"{self.log_name}-unet.png",
        )
        print("\nBest run:")
        print("best base_alpha:")
        print(best_base_alpha)
        print("\nbest weights:")
        print(",".join(list(map(str, best_weights))))
        draw_unet(
            best_base_alpha,
            best_weights,
            model_a=Path(self.model_a).stem,
            model_b=Path(self.model_b).stem,
            figname=unet_path,
        )

        if self.save_best:
            print(f"Saving best merge: {self.merger.best_output_file}")
            self.merger.merge(best_weights, best_base_alpha, best=True)


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


def maxwhere(li: List[float]) -> Tuple[int, float]:
    m = 0
    mi = -1
    for i, v in enumerate(li):
        if v > m:
            m = v
            mi = i
    return mi, m


def minwhere(li: List[float]) -> Tuple[int, float]:
    m = 10
    mi = -1
    for i, v in enumerate(li):
        if v < m:
            m = v
            mi = i
    return mi, m


def convergence_plot(
    scores: List[float],
    figname: Path = None,
    minimise=False,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(scores)

    if minimise:
        star_i, star_score = minwhere(scores)
    else:
        star_i, star_score = maxwhere(scores)
    plt.plot(star_i, star_score, "or")

    plt.xlabel("iterations")

    if minimise:
        plt.ylabel("loss")
    else:
        plt.ylabel("score")

    sns.despine()

    if figname:
        figname.parent.mkdir(exist_ok=True)
        plt.title(figname.stem)
        print("Saving fig to:", figname)
        plt.savefig(figname)
