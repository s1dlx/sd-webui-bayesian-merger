import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from bayes_opt.logger import JSONLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from sd_webui_bayesian_merger.artist import convergence_plot, draw_unet
from sd_webui_bayesian_merger.bounds import Bounds
from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.merger import Merger
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.scorer import AestheticScorer

PathT = os.PathLike


@dataclass
class Optimiser:
    cfg: DictConfig
    best_rolling_score: float = 0.0

    def __post_init__(self) -> None:
        self.bounds_initialiser = Bounds()
        self.generator = Generator(self.cfg.url, self.cfg.batch_size)
        self.merger = Merger(self.cfg)
        self.start_logging()
        self.scorer = AestheticScorer(self.cfg)
        self.prompter = Prompter(self.cfg)
        self.iteration = 0

    def start_logging(self) -> None:
        run_name = "-".join(self.merger.output_file.stem.split("-")[:-1])
        self.log_name = f"{run_name}-{self.cfg.optimiser}"
        self.logger = JSONLogger(
            path=str(
                Path(
                    HydraConfig.get().runtime.output_dir,
                    f"{self.log_name}.json",
                )
            )
        )

    def init_params(self) -> Dict:
        for guide in ["frozen_params", "custom_ranges", "groups"]:
            if guide not in self.cfg.optimisation_guide.keys():
                with open_dict(self.cfg):
                    self.cfg["optimisation_guide"][guide] = None
        return self.bounds_initialiser.get_bounds(
            self.merger.greek_letters,
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.custom_ranges
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else None,
        )

    def sd_target_function(self, **params) -> float:
        def print_iteration_info(iteration_type: str):
            print(f"\n{iteration_type} - Iteration: {self.iteration}")

        self.iteration += 1
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.init_points else "optimisation"
        )

        if self.iteration in {1, self.cfg.init_points + 1}:
            print("\n" + "-" * 10 + f" {iteration_type} " + "-" * 10 + ">")
        print_iteration_info(iteration_type)

        weights, bases = self.bounds_initialiser.assemble_params(
            params,
            self.merger.greek_letters,
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else None,
        )
        self.merger.merge(weights, bases)

        images, gen_paths, payloads = self.generate_images()
        scores, norm = self.score_images(images, gen_paths, payloads)
        avg_score = self.scorer.average_score(scores, norm)
        self.update_best_score(bases, weights, avg_score)

        return avg_score

    def generate_images(self) -> Tuple[List, List, List]:
        images = []
        gen_paths = []
        payloads, paths = self.prompter.render_payloads(self.cfg.batch_size)
        for i, payload in tqdm(enumerate(list(payloads)), desc="Batches generation"):
            generated_images = self.generator.generate(payload)
            images.extend(generated_images)
            gen_paths.extend([paths[i]] * len(generated_images))
            payloads[i : i + 1] = [payloads[i]] * len(generated_images)
        return images, gen_paths, payloads

    def score_images(self, images, gen_paths, payloads) -> List[float]:
        print("\nScoring")
        return self.scorer.batch_score(images, gen_paths, payloads, self.iteration)

    def update_best_score(self, bases, weights, avg_score):
        print(f"{'-'*10}\nRun score: {avg_score}")
        weights_strings = {
            gl: ",".join(map(str, weights[gl])) for gl in self.merger.greek_letters
        }

        for gl in self.merger.greek_letters:
            print(f"\nrun base_{gl}: {bases[gl]}")
            print(f"run weights_{gl}: {weights_strings[gl]}")

        if avg_score > self.best_rolling_score:
            print("\n NEW BEST!")
            self.best_rolling_score = avg_score
            Optimiser.save_best_log(bases, weights_strings)

    @abstractmethod
    def optimise(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    def plot_and_save(
        self,
        scores: List[float],
        best_bases: Dict,
        best_weights: Dict,
        minimise: bool,
    ) -> None:
        img_path = Path(
            HydraConfig.get().runtime.output_dir,
            f"{self.log_name}.png",
        )
        convergence_plot(scores, figname=img_path, minimise=minimise)

        unet_path = Path(
            HydraConfig.get().runtime.output_dir,
            f"{self.log_name}-unet.png",
        )
        print("\n" + "-" * 10 + "> Done!")
        print("\nBest run:")

        best_weights_strings = {}
        for gl in self.merger.greek_letters:
            print(f"\nbest base_{gl}: {best_bases[gl]}")
            print(f"best weights_{gl}:")
            w_str = ",".join(list(map(str, best_weights[gl])))
            print(w_str)
            best_weights_strings[gl] = w_str

        Optimiser.save_best_log(best_bases, best_weights_strings)
        draw_unet(
            best_bases["alpha"],
            best_weights["alpha"],
            model_a=Path(self.cfg.model_a).stem,
            model_b=Path(self.cfg.model_b).stem,
            figname=unet_path,
        )

        if self.cfg.save_best:
            print("Merging best model")
            self.merger.merge(best_weights, best_bases, save_best=True)

    @staticmethod
    def save_best_log(bases: Dict, weights_strings: Dict) -> None:
        print("Saving best.log")
        with open(
            Path(HydraConfig.get().runtime.output_dir, "best.log"),
            "w",
            encoding="utf-8",
        ) as f:
            for m, b in bases.items():
                f.write(f"{bases[m]}\n\n{weights_strings[m]}\n\n")

    @staticmethod
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
