import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import safetensors.torch
import torch
from omegaconf import DictConfig
from sd_meh import merge_methods
from sd_meh.merge import merge_models

BETA_METHODS = [
    name
    for name, fn in dict(inspect.getmembers(merge_methods, inspect.isfunction)).items()
    if "beta" in inspect.getfullargspec(fn)[0]
]


NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
NUM_MODELS_NEEDED = {
    name: 3 if "c" in inspect.getfullargspec(fn)[0] else 2
    for name, fn in MERGE_METHODS.items()
}


@dataclass
class Merger:
    cfg: DictConfig

    def __post_init__(self):
        self.parse_models()
        self.create_model_out_name()
        self.create_best_model_out_name()

    def parse_models(self):
        self.model_a = Path(self.cfg.model_a)
        self.model_b = Path(self.cfg.model_b)
        self.models = {"model_a": self.model_a, "model_b": self.model_b}
        self.model_keys = ["model_a", "model_b"]
        self.model_real_names = [
            self.models["model_a"].stem,
            self.models["model_b"].stem,
        ]
        self.greek_letters = ["alpha"]
        seen_models = 2
        for m in ["model_c", "model_d", "model_e"]:
            if seen_models == NUM_MODELS_NEEDED[self.cfg.merge_mode]:
                break
            if m in self.cfg:
                p = Path(self.cfg[m])
            else:
                break
            if p.exists():
                self.models[m] = p
                self.model_keys.append(m)
                self.model_real_names.append(self.models[m].stem)
            else:
                break
            seen_models += 1
        if self.cfg.merge_mode in BETA_METHODS:
            self.greek_letters.append("beta")
        self.model_name_suffix = f"bbwm-{'-'.join(self.model_real_names)}"

        try:
            assert len(self.model_keys) == NUM_MODELS_NEEDED[self.cfg.merge_mode]
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

    def merge(
        self,
        weights: Dict,
        bases: Dict,
        best: bool = False,
    ) -> None:
        thetas = dict(self.models.items())

        thetas["model_a"] = merge_models(
            thetas,
            weights,
            bases,
            self.cfg.merge_mode,
            self.cfg.best_precision,
            device=self.cfg.device,
            work_device=self.cfg.work_device,
            prune=self.cfg.prune,
            threads=self.cfg.threads,
            weights_clip=self.cfg.weights_clip,
            re_basin=self.cfg.rebasin,
            iterations=self.cfg.rebasin_iterations,
        )

        if best:
            print(f"Saving {self.best_output_file}")
            if self.cfg.best_format == "safetensors":
                safetensors.torch.save_file(
                    thetas["model_a"]
                    if type(thetas["model_a"]) == dict
                    else thetas["model_a"].to_dict(),
                    self.best_output_file,
                    metadata={"format": "pt"},
                )
            else:
                torch.save(
                    {
                        "state_dict": thetas["model_a"]
                        if type(thetas["model_a"]) == dict
                        else thetas["model_a"].to_dict()
                    },
                    self.best_output_file,
                )
        else:
            print(f"Saving {self.output_file}")
            safetensors.torch.save_file(
                thetas["model_a"]
                if type(thetas["model_a"]) == dict
                else thetas["model_a"].to_dict(),
                self.output_file,
                metadata={"format": "pt", "precision": "fp16"},
            )
