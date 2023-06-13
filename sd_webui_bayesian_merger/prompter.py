import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

PathT = os.PathLike


class CardDealer:
    def __init__(self, wildcards_dir: str):
        self.find_wildcards(wildcards_dir)

    def find_wildcards(self, wildcards_dir: str) -> None:
        wdir = Path(wildcards_dir)
        if wdir.exists():
            self.wildcards = {w.stem: w for w in wdir.glob("*.txt")}
        else:
            self.wlldcards: Dict[str, PathT] = {}

    def sample_wildcard(self, wildcard_name: str) -> str:
        if wildcard_name in self.wildcards:
            with open(
                self.wildcards[wildcard_name],
                "r",
                encoding="utf-8",
            ) as f:
                lines = f.readlines()
                return random.choice(lines).strip()

        # TODO raise warning?
        return wildcard_name

    def replace_wildcards(self, prompt: str) -> str:
        chunks = prompt.split("__")
        replacements = [self.sample_wildcard(w) for w in chunks[1::2]]
        chunks[1::2] = replacements
        return "".join(chunks)


def assemble_payload(defaults: Dict, payload: Dict) -> Dict:
    for k, v in defaults.items():
        if k not in payload.keys():
            payload[k] = v
    return payload


def unpack_cargo(cargo: DictConfig) -> Tuple[Dict, Dict]:
    defaults = {}
    payloads = {}
    for k, v in cargo.items():
        if k == "cargo":
            for p_name, p in v.items():
                payloads[p_name] = OmegaConf.to_container(p)
        elif isinstance(v, (DictConfig, ListConfig)):
            defaults[k] = OmegaConf.to_container(v)
        else:
            defaults[k] = v
    return defaults, payloads


@dataclass
class Prompter:
    cfg: DictConfig

    def __post_init__(self):
        self.load_payloads()
        self.dealer = CardDealer(self.cfg.wildcards_dir)

    def load_payloads(self) -> None:
        self.raw_payloads = {}
        defaults, payloads = unpack_cargo(self.cfg.payloads)
        for payload_name, payload in payloads.items():
            self.raw_payloads[payload_name] = assemble_payload(
                defaults,
                payload,
            )

    def render_payloads(self, batch_size: int = 0) -> Tuple[List[Dict], List[str]]:
        payloads = []
        paths = []
        for p_name, p in self.raw_payloads.items():
            for _ in range(batch_size):
                rendered_payload = p.copy()
                rendered_payload["prompt"] = self.dealer.replace_wildcards(p["prompt"])
                paths.append(p_name)
                payloads.append(rendered_payload)
        return payloads, paths
