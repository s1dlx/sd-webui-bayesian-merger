# adapted from
# bbc-mc/sdweb-merge-block-weighted-gui/scripts/mbw/merge_block_weighted.py

import os
import re

from dataclasses import dataclass
from typing import List, Dict, Tuple
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


@dataclass
class Merger:
    cfg: DictConfig

    def __post_init__(self):
        self.model_a = Path(self.cfg.model_a)
        self.model_b = Path(self.cfg.model_b)
        self.model_name_suffix = f"bbwm-{self.model_a.stem}-{self.model_b.stem}"
        self.create_model_out_name(0)
        self.create_best_model_out_name()

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
        self,
        key: str,
        weights: List[float],
        base_alpha: float,
        best: bool,
        theta_0: Dict,
        theta_1: Dict,
    ) -> Tuple[str, Dict]:
        if "model" not in key or key not in theta_1:
            return
        if KEY_POSITION_IDS in key and self.cfg.skip_position_ids in [1, 2]:
            if self.cfg.skip_position_ids == 2:
                theta_0[key] = torch.tensor(
                    [list(range(MAX_TOKENS))], dtype=torch.int64
                )
            return

        re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
        re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
        re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12
        c_alpha = base_alpha
        if "model.diffusion_model." in key:
            weight_index = -1

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
                c_alpha = weights[weight_index]

        merged = (1 - c_alpha) * theta_0[key] + c_alpha * theta_1[key]
        if not best or self.cfg.best_precision == "16":
            merged = merged.half()

        return (key, merged)

    def merge(
        self,
        weights: List[float],
        base_alpha: float,
        best: bool = False,
    ) -> None:
        if len(weights) != NUM_TOTAL_BLOCKS:
            raise ValueError(f"weights value must be {NUM_TOTAL_BLOCKS}")

        theta_0 = self.load_sd_model(self.model_a)
        theta_1 = self.load_sd_model(self.model_b)

        merged_model = {}
        for key in tqdm(theta_0.keys(), desc="merging 1/1"):
            if result := self.merge_key(
                key, weights, base_alpha, best, theta_0, theta_1
            ):
                merged_model[key] = result[1]

        for key in tqdm(theta_1.keys(), desc="merging 2/2"):
            if "model" in key and key not in theta_0:
                continue
            if KEY_POSITION_IDS in key and self.cfg.skip_position_ids in [1, 2]:
                if self.cfg.skip_position_ids == 2:
                    theta_1[key] = torch.tensor(
                        [list(range(MAX_TOKENS))], dtype=torch.int64
                    ).half()
                continue
            merged_model[key] = theta_1[key]
            if not best or self.cfg.best_precision == "16":
                merged_model[key] = merged_model[key].half()

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
