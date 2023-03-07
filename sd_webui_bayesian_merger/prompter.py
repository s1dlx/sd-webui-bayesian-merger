import random
import yaml

from pathlib import Path


class CardDealer:
    def __init__(self, wildcards_dir: str):
        self.find_wildcards(wildcards_dir)

    def find_wildcards(self, wildcards_dir: str) -> None:
        wdir = Path(wildcards_dir)
        if wdir.exists():
            self.wildcards = {w.stem: w for w in wdir.glob("*.txt")}
        else:
            self.widcards = {}

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


def load_yaml(yaml_file: Path) -> dict:
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_payload(payload: dict) -> dict:
    if "prompt" not in payload:
        raise ValueError(f"{payload['path']} doesn't have a prompt")

    # set defaults
    # TODO: get default values from args
    for k, v in zip(
        [
            "neg_prompt",
            "seed",
            "steps",
            "cfg",
            "width",
            "height",
            "sampler",
        ],
        [
            "",
            -1,
            20,
            7,
            512,
            512,
            "Euler",
        ],
    ):
        if k not in payload:
            payload[k] = v

    return payload


class Prompter:
    def __init__(self, payloads_dir: str, wildcards_dir: str):
        self.find_payloads(payloads_dir)
        self.load_payloads()
        self.dealer = CardDealer(wildcards_dir)

    def find_payloads(self, payloads_dir: str) -> None:
        # TODO: allow for listing payloads instead of taking all of them
        pdir = Path(payloads_dir)
        if pdir.exists():
            self.raw_payloads = {
                p.stem: {"path": p}
                for p in pdir.glob("*.yaml")
                if ".tmpl.yaml" not in p.name
            }

        else:
            # TODO: pick a better error
            raise ValueError("payloads directory not found!")

    def load_payloads(self) -> None:
        for payload_name, payload in self.raw_payloads.items():
            raw_payload = load_yaml(payload["path"])
            checked_payload = check_payload(raw_payload)
            self.raw_payloads[payload_name].update(checked_payload)

    def render_payloads(self) -> [dict]:
        payloads = []
        for _, p in self.raw_payloads.items():
            rendered_payload = p.copy()
            rendered_payload["prompt"] = self.dealer.replace_wildcards(p["prompt"])
            payloads.append(rendered_payload)
        return payloads
