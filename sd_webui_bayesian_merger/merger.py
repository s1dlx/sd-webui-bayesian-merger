import os
import re
import sys

from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import torch
import safetensors.torch

from tqdm import tqdm
from omegaconf import DictConfig

from sd_webui_bayesian_merger.model import SDModel

PathT = os.PathLike

MAX_TOKENS = 77
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)

NUM_MODELS_NEEDED = {
    "add_difference": 3,
    "weighted_add_difference": 3,
    "weighted_sum": 2,
    "sum_twice": 3,
    "triple_sum": 3,
    "weighted_double_difference": 5,
}


@dataclass
class Merger:
    cfg: DictConfig

    def __post_init__(self):
        self.parse_models()
        self.create_model_out_name(0)
        self.create_best_model_out_name()

    def parse_models(self):
        self.model_a = Path(self.cfg.model_a)
        self.model_b = Path(self.cfg.model_b)
        self.models = {"model_a": self.model_a, "model_b": self.model_b}
        self.model_names = ["model_a", "model_b"]
        self.greek_letters = ["alpha", "beta"]
        seen_models = 2
        for m, l in zip(
            ["model_c", "model_d", "model_e"], ["gamma", "delta", "epsilon"]
        ):
            if seen_models == NUM_MODELS_NEEDED[self.cfg.merge_mode]:
                break
            if m in self.cfg:
                p = Path(self.cfg[m])
            else:
                break
            if p.exists():
                self.models[m] = p
                self.model_names.append(m)
                self.greek_letters.append(l)
            else:
                break
            seen_models += 1
        self.model_name_suffix = f"bbwm-{'-'.join(self.model_names)}"

        try:
            assert len(self.model_names) == NUM_MODELS_NEEDED[self.cfg.merge_mode]
        except AssertionError:
            print(
                "number of models defined not compatible with merge mode, aborting optimisation"
            )
            sys.exit()

    def create_model_out_name(self, it: int = 0) -> None:
        model_out_name = self.model_name_suffix
        model_out_name += f"-it_{it}"
        model_out_name += ".safetensors"
        self.model_out_name = model_out_name  # this is needed to switch
        self.output_file = Path(self.model_a.parent, model_out_name)

    def create_best_model_out_name(self):
        model_out_name = self.model_name_suffix
        model_out_name += "-best"
        model_out_name += f"-{self.cfg.best_precision}bit"
        model_out_name += f".{self.cfg.best_format}"
        self.best_output_file = Path(self.model_a.parent, model_out_name)

    def remove_previous_ckpt(self, current_it: int) -> None:
        if current_it > 1 and self.output_file.exists():
            self.create_model_out_name(current_it - 1)
            print(f"Removing {self.output_file}")
            self.output_file.unlink()
        self.create_model_out_name(current_it)

    def keep_best_ckpt(self) -> None:
        if self.best_output_file.exists():
            self.best_output_file.unlink()
        self.output_file.rename(self.best_output_file)

    def load_sd_model(self, model_path: PathT) -> SDModel:
        return SDModel(model_path, self.cfg.device).load_model()

    def merge_key(
        self, key: str, thetas: Dict, weights: Dict, bases: Dict, best: bool
    ) -> Tuple[str, Dict]:
        if KEY_POSITION_IDS in key or "model" not in key:
            return
        for theta in thetas.values():
            if key not in theta:
                return

        if "model.diffusion_model." in key:
            weight_index = -1

            re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
            re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
            re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

            if "time_embed" in key:
                weight_index = 0  # before input blocks
            elif ".out." in key:
                weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
            elif m := re_inp.search(key):
                weight_index = int(m.groups()[0])
            elif re_mid.search(key):
                weight_index = NUM_INPUT_BLOCKS
            elif m := re_out.search(key):
                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.groups()[0])

            if weight_index >= NUM_TOTAL_BLOCKS:
                raise ValueError(f"illegal block index {key}")

            if weight_index >= 0:
                current_bases = {k: w[weight_index] for k, w in weights.items()}

            merged = self.merge_block(current_bases, thetas, key)

            if not best or self.cfg.best_precision == "16":
                merged = merged.half()

            return (key, merged)

    def merge_block(self, current_bases: Dict, thetas: Dict, key: str) -> Dict:
        t0 = thetas["model_a"][key]
        t1 = thetas["model_b"][key]
        alpha = current_bases["alpha"]
        if self.cfg.merge_mode == "weighted_sum":
            return (1 - alpha) * t0 + alpha * t1

        t2 = thetas["model_c"][key]
        if self.cfg.merge_mode == "add_difference":
            return t0 + alpha * (t1 - t2)
        elif self.cfg.merge_mode == "weighted_add_difference":
            return (1 - alpha) * t0 + alpha * (t1 - t2)

        beta = current_bases["beta"]
        if self.cfg.merge_mode == "sum_twice":
            return (1 - beta) * ((1 - alpha) * t0 + alpha * t1) + beta * t2
        elif self.cfg.merge_mode == "triple_sum":
            return (1 - alpha - beta) * t0 + alpha * t1 + beta * t2
        elif self.cfg.merge_mode == "weighted_double_difference":
            t3 = thetas["model_d"][key]
            t4 = thetas["model_e"][key]
            return (1 - alpha) * t0 + alpha * (
                (1 - beta) * (t1 - t2) + beta * (t3 - t4)
            )

    def merge(
        self,
        weights: Dict,
        bases: Dict,
        best: bool = False,
    ) -> None:
        thetas = {k: self.load_sd_model(m) for k, m in self.models.items()}

        merged_model = {}
        for key in tqdm(thetas["model_a"].keys(), desc="stage 1"):
            if result := self.merge_key(
                key,
                thetas,
                weights,
                bases,
                best,
            ):
                merged_model[key] = result[1]

        for key in tqdm(thetas['model_b'].keys(), desc="stage 2"):
            if KEY_POSITION_IDS in key or "model" not in key:
                continue
            if key not in merged_model:
                merged_model.update({key: thetas['model_b'][key]})

        if best:
            print(f"Saving {self.best_output_file}")
            if self.cfg.best_format == "safetensors":
                safetensors.torch.save_file(
                    merged_model,
                    self.best_output_file,
                    metadata={"format": "pt"},
                )
            else:
                torch.save({"state_dict": merged_model}, self.best_output_file)
        else:
            print(f"Saving {self.output_file}")
            safetensors.torch.save_file(
                merged_model,
                self.output_file,
                metadata={"format": "pt", "precision": "fp16"},
            )
