import json
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

from bayes_opt.logger import JSONLogger
from hydra.core.hydra_config import HydraConfig
from hyperopt import hp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from sd_webui_bayesian_merger.artist import convergence_plot, draw_unet
from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS, Merger
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.scorer import AestheticScorer

PathT = os.PathLike


class BoundsInitialiser:
    @staticmethod
    def get_block_bounds(
        optimiser: str, block_name: str, lb: float = 0.0, ub: float = 1.0
    ) -> Union[Tuple[float, float], Any]:
        # TODO: check/clip bounds in [-1, 1]
        return hp.uniform(block_name, lb, ub) if optimiser == "tpe" else (lb, ub)

    @staticmethod
    def get_greek_letter_bounds(
        greek_letter: str,
        optimiser: str,
        frozen_params: Dict[str, float] = None,
        custom_ranges: Dict[str, Tuple[float, float]] = None,
    ) -> Dict:
        if frozen_params is None:
            frozen_params = {}
        if custom_ranges is None:
            custom_ranges = DictConfig({})
        block_names = [f"block_{i}_{greek_letter}" for i in range(NUM_TOTAL_BLOCKS)] + [
            f"base_{greek_letter}"
        ]
        ranges = {b: (0.0, 1.0) for b in block_names} | OmegaConf.to_object(
            custom_ranges
        )
        return {
            b: BoundsInitialiser.get_block_bounds(optimiser, b, *ranges[b])
            for b in block_names
            if b not in frozen_params
        }

    @staticmethod
    def get_grouped_bounds_and_names(groups: List[List[str]]) -> Tuple[Dict, Set]:
        grouped_bounds = {}
        blocks_to_group = set()
        for group in groups:
            blocks_to_group.update(group)
            group_name = "-".join(group)
            grouped_bounds |= {b: group_name for b in group}
        return grouped_bounds, blocks_to_group

    @staticmethod
    def extract_grouped_bounds(bounds: Dict, blocks_to_group: Set) -> Dict:
        return {block: bounds[block] for block in bounds if block in blocks_to_group}

    @staticmethod
    def consolidate_groups(bounds: Dict, groups: List[List[str]]) -> Dict:
        (
            grouped_bounds,
            blocks_to_group,
        ) = BoundsInitialiser.get_grouped_bounds_and_names(groups)
        replaced_bounds = BoundsInitialiser.extract_grouped_bounds(
            bounds, blocks_to_group
        )

        for block in replaced_bounds:
            del bounds[block]

        for block, bound in replaced_bounds.items():
            group_name = grouped_bounds[block]
            if group_name not in bounds:
                bounds[group_name] = bound
            elif bound != bounds[group_name]:
                raise KeyError(
                    f"you are tring to freeze/set range differently within the same group! {group_name}"
                )

        return bounds

    @staticmethod
    def get_bounds(
        greek_letters: List[str],
        optimiser: str,
        frozen_params: Dict[str, float] = None,
        custom_ranges: Dict[str, Tuple[float, float]] = None,
        groups: List[List[str]] = None,
    ) -> Dict:
        if frozen_params is None:
            frozen_params = {}
        if custom_ranges is None:
            custom_ranges = DictConfig({})
        if groups is None:
            groups = []

        bounds = {}
        for greek_letter in greek_letters:
            bounds |= BoundsInitialiser.get_greek_letter_bounds(
                greek_letter, optimiser, frozen_params, custom_ranges
            )

        return BoundsInitialiser.consolidate_groups(bounds, groups)


@dataclass
class Optimiser:
    cfg: DictConfig
    best_rolling_score: float = 0.0

    def __post_init__(self) -> None:
        self.bounds_initialiser = BoundsInitialiser()
        self.generator = Generator(self.cfg.url, self.cfg.batch_size)
        self.merger = Merger(self.cfg)
        self.start_logging()
        self.scorer = AestheticScorer(self.cfg)
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self._clean = True

    def cleanup(self) -> None:
        if self._clean:
            # clean up and remove the last merge
            self.merger.remove_previous_ckpt(self.iteration)
        else:
            self._clean = True

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
        return self.bounds_initialiser.get_bounds(
            self.merger.greek_letters,
            self.cfg.optimiser,
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else {},
            self.cfg.optimisation_guide.custom_ranges
            if self.cfg.guided_optimisation
            else {},
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else [[]],
        )

    def assemble_params(self, params: Dict) -> Tuple[Dict, Dict]:
        weights = {}
        bases = {}
        for gl in self.merger.greek_letters:
            w = []
            for i in range(NUM_TOTAL_BLOCKS):
                block_name = f"block_{i}_{gl}"
                if block_name in params:
                    w.append(params[block_name])
                elif (
                    self.cfg.optimisation_guide.frozen_params
                    and block_name in self.cfg.optimisation_guide.frozen_params
                ):
                    w.append(self.cfg.optimisation_guide.frozen_params[block_name])
                elif self.cfg.optimisation_guide.groups:
                    for group in self.cfg.optimisation_guide.groups:
                        if block_name in group:
                            group_name = "-".join(group)
                            w.append(params[group_name])
                            break
            assert len(w) == NUM_TOTAL_BLOCKS
            weights[gl] = w

            base_name = f"base_{gl}"
            if base_name in params:
                bases[gl] = params[base_name]
            elif (
                self.cfg.optimisation_guide.frozen_params
                and base_name in self.cfg.optimisation_guide.frozen_params
            ):
                bases[gl] = self.cfg.optimisation_guide.frozen_params[base_name]
            elif self.cfg.optimisation_guide.groups:
                for group in self.cfg.optimisation_guide.groups:
                    if base_name in group:
                        group_name = "-".join(group)
                        bases[gl] = params[group_name]
                        break

        assert len(weights) == len(self.merger.greek_letters)
        assert len(bases) == len(self.merger.greek_letters)

        return weights, bases

    def sd_target_function(self, **params) -> float:
        self.iteration += 1

        if self.iteration == 1:
            print("\n" + "-" * 10 + " warmup " + "-" * 10 + ">")
        elif self.iteration == self.cfg.init_points + 1:
            print("\n" + "-" * 10 + " optimisation " + "-" * 10 + ">")

        it_type = "warmup" if self.iteration <= self.cfg.init_points else "optimisation"
        print(f"\n{it_type} - Iteration: {self.iteration}")

        weights, bases = self.assemble_params(params)

        self.merger.create_model_out_name(self.iteration)
        self.merger.merge(weights, bases)
        self.cleanup()

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
            gen_paths.extend([paths[i]] * self.cfg.batch_size * payload["batch_size"])

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

        weights_strings = {}
        for gl in self.merger.greek_letters:
            print(f"\nrun base_{gl}: {bases[gl]}")
            print(f"run weights_{gl}:")
            w_str = ",".join(list(map(str, weights[gl])))
            print(w_str)
            weights_strings[gl] = w_str

        if avg_score > self.best_rolling_score:
            print("\n NEW BEST!")
            print("Saving best model merge")
            self.best_rolling_score = avg_score
            Optimiser.save_best_log(bases, weights_strings)
            self.merger.keep_best_ckpt()
            self._clean = False

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
            print(f"Saving best merge: {self.merger.best_output_file}")
            self.merger.merge(
                best_weights,
                best_bases,
                best=True,
            )

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
