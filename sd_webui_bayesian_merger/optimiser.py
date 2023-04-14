import os
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import json

from tqdm import tqdm
from omegaconf import DictConfig
from bayes_opt.logger import JSONLogger
from hydra.core.hydra_config import HydraConfig

from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.merger import Merger, NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.scorer import AestheticScorer
from sd_webui_bayesian_merger.artist import draw_unet, convergence_plot

PathT = os.PathLike


@dataclass
class Optimiser:
    cfg: DictConfig
    best_rolling_score: float = 0.0

    def __post_init__(self) -> None:
        self.generator = Generator(self.cfg.url, self.cfg.batch_size)
        self.merger = Merger(self.cfg)
        self.start_logging()
        self.scorer = AestheticScorer(self.cfg)
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self._clean = True
        self.has_beta = self.cfg.merge_mode in ["sum_twice", "triple_sum"]

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

    def pbound(self, param_name: str):
        if self.optimiser == "tpe":
            return hp.uniform(param_name, 0.0, 1.0)
        return (0.0, 1.0)

    def init_params(self) -> Dict:
        if self.cfg.enable_fix_blocks:
            pbounds = {}
            for i in range(NUM_TOTAL_BLOCKS):
                block_name = f"block_{i}"
                if block_name not in self.cfg.fix_blocks:
                    for gl in self.merger.greek_letters:
                        pbounds[f"{block_name}_{gl}"] = self.pbound(
                            f"{block_name}_{gl}"
                        )
            for gl in self.merger.greek_letters:
                if f"base_{gl}" not in self.cfg.fix_blocks:
                    pbounds[f"base_{gl}"] = self.pbound(f"base_{gl}")
        else:
            pbounds = {
                f"block_{i}_alpha": self.pbound(f"block_{i}_alpha")
                for i in range(NUM_TOTAL_BLOCKS)
            }
            pbounds["base_alpha"] = self.pbound("base_alpha")
            for gl in self.merger.greek_letters:
                pbounds |= {
                    f"block_{i}_{gl}": self.pbound(f"block_{i}_{gl}")
                    for i in range(NUM_TOTAL_BLOCKS)
                }
                pbounds[f"base_{gl}"] = self.pbound(f"base_{gl}")
        return pbounds

    def sd_target_function(self, **params) -> float:
        self.iteration += 1

        if self.iteration == 1:
            print("\n" + "-" * 10 + " warmup " + "-" * 10 + ">")
        elif self.iteration == self.cfg.init_points + 1:
            print("\n" + "-" * 10 + " optimisation " + "-" * 10 + ">")

        it_type = "warmup" if self.iteration <= self.cfg.init_points else "optimisation"
        print(f"\n{it_type} - Iteration: {self.iteration}")

        if self.cfg.enable_fix_blocks:
            weights = {}
            bases = {}
            for gl in self.merger.greek_letters:
                greek_weights = []
                for i in range(NUM_TOTAL_BLOCKS):
                    block_name = f"block_{i}"
                    greek_block_name = f"{block_name}_{gl}"
                    if greek_block_name in params:
                        greek_weights.append(params[greek_block_name])
                    else:
                        fixed_block_weight = self.cfg.fix_block_values[block_name]
                        greek_weights.append(fixed_block_weight)
                weights[gl] = greek_weights
                greek_base = f"base_{gl}"
                if greek_base in params:
                    bases[gl] = params[greek_base]
                else:
                    fixed_base = self.cfg.fix_block_values[greek_base]
                    bases[gl] = fixed_base
        else:
            weights = {
                gl: [params[f"block_{i}_{gl}"] for i in range(NUM_TOTAL_BLOCKS)]
                for gl in self.merger.greek_letters
            }
            bases = {gl: params[f"base_{gl}"] for gl in self.merger.greek_letters}

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
            save_best_log(bases, weights_strings)
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

        save_best_log(best_bases, best_weights_strings)
        draw_unet(
            best_bases["alpha"],
            best_weights["alpha"],
            model_a=Path(self.cfg.model_a).stem,
            model_b=Path(self.cfg.model_b).stem,
            figname=unet_path,
        )
        # TODO: draw more UNETs?

        # TODO: is this really necessary?
        if self.cfg.save_best:
            print(f"Saving best merge: {self.merger.best_output_file}")
            self.merger.merge(
                best_weights,
                best_bases,
                best=True,
            )


def save_best_log(bases: Dict, weights_strings: Dict) -> None:
    print("Saving best.log")
    with open(
        Path(HydraConfig.get().runtime.output_dir, "best.log"),
        "w",
        encoding="utf-8",
    ) as f:
        for m, b in bases.items():
            f.write(f"{bases[m]}\n\n{weights_strings[m]}\n\n")


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
