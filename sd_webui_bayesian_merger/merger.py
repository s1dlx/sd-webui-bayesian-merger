import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import safetensors.torch
import torch
from omegaconf import DictConfig
from sd_meh import merge_methods
import requests


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
    url: str
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

    def keep_best_ckpt(self) -> None:
        if self.best_output_file.exists():
            self.best_output_file.unlink()
        # self.output_file.rename(self.best_output_file)

    def merge(
        self,
        weights: Dict,
        bases: Dict,
    ) -> None:
        bases = {f"base_{k}": v for k, v in bases.items()}
        option_payload = {
            "merge_method": self.cfg.merge_mode,
            **{k: str(v) for k, v in self.models.items()},
            **bases,
            **weights,
            "precision": self.cfg.best_precision,
            "device": self.cfg.device,
        }

        print("Merging models")
        r = requests.post(
            url=f"{self.url}/bbwm/merge-models",
            json=option_payload,
        )
        r.raise_for_status()

    def save_best(self, theta):
        print(f"Saving {self.best_output_file}")
        if self.cfg.best_format == "safetensors":
            safetensors.torch.save_file(
                theta,
                self.best_output_file,
                metadata={"format": "pt"},
            )
        else:
            torch.save({"state_dict": theta}, self.best_output_file)
