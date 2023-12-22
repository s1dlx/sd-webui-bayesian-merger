import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import clip
import requests
import safetensors.torch
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from PIL import Image, PngImagePlugin
from sd_webui_bayesian_merger.models.AestheticScore import AestheticScore as AES
from sd_webui_bayesian_merger.models.ImageReward import ImageReward as IMGR
from sd_webui_bayesian_merger.models.CLIPScore import CLIPScore as CLP
from sd_webui_bayesian_merger.models.BLIPScore import BLIPScore as BLP

AES_URL = (
    "https://github.com/grexzen/SD-Chad/blob/main/"
)
IR_URL = (
    "https://huggingface.co/THUDM/ImageReward/resolve/main/"
)
CLIP_URL = (
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/"
)
BLIP_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/"
)

printWSLFlag = 0


@dataclass
class AestheticScorer:
    cfg: DictConfig
    scorer_model_name: Dict
    model_path: Dict
    model: Dict

    def __post_init__(self):
        if "manual" in self.cfg.scorer_method:
            self.cfg.save_imgs = True

        if self.cfg.save_imgs:
            self.imgs_dir = Path(HydraConfig.get().runtime.output_dir, "imgs")
            if not self.imgs_dir.exists():
                self.imgs_dir.mkdir()

        self.scorer_model_name['aes'] = {}
        self.model_path['aes'] = {}
        self.model['aes'] = {}
        if "aes" in self.cfg.scorer_method:
            self.scorer_model_name['aes']['laion'] = "sac+logos+ava1-l14-linearMSE.pth"
            self.model_path['aes']['laion'] = Path(
                self.cfg.scorer_model_dir,
                self.scorer_model_name['aes']['laion'],
            )
            self.scorer_model_name['aes']['chad'] = "chadscorer.pth"
            self.model_path['aes']['chad'] = Path(
                self.cfg.scorer_model_dir,
                self.scorer_model_name['aes']['chad'],
            )
        if "ir" in self.cfg.scorer_method:
            self.scorer_model_name['ir'] = "ImageReward.pt"
            self.model_path['ir'] = Path(
                self.cfg.scorer_model_dir,
                self.scorer_model_name['ir'],
            )
        if "clip" in self.cfg.scorer_method:
            self.scorer_model_name['clip'] = "ViT-L-14.pt"
            self.model_path['clip'] = Path(
                self.cfg.scorer_model_dir,
                self.scorer_model_name['clip'],
            )
        if "blip" in self.cfg.scorer_method:
            self.scorer_model_name['blip'] = "model_large.pth"
            self.model_path['blip'] = Path(
                self.cfg.scorer_model_dir,
                self.scorer_model_name['blip'],
            )
        self.get_models()
        self.load_models()

    def get_models(self) -> None:
        if "aes" in self.cfg.scorer_method and not self.model_path['aes']['laion'].is_file():
            print("You do not have the laion aesthetic model, let me download that for you")
            url = AES_URL

            url += f"{self.scorer_model_name['aes']['laion']}?raw=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path['aes']['laion'].absolute(), "wb") as f:
                print(f"saved into {self.model_path['aes']['laion']}")
                f.write(r.content)

        if "aes" in self.cfg.scorer_method and not self.model_path['aes']['chad'].is_file():
            print("You do not have the chad aesthetic model, let me download that for you")
            url = AES_URL

            url += f"{self.scorer_model_name['aes']['chad']}?raw=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path['aes']['chad'].absolute(), "wb") as f:
                print(f"saved into {self.model_path['aes']['chad']}")
                f.write(r.content)

        if "ir" in self.cfg.scorer_method and not self.model_path['ir'].is_file():
            print("You do not have the image reward model, let me download that for you")
            url = IR_URL

            url += f"{self.scorer_model_name['ir']}?download=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path['ir'].absolute(), "wb") as f:
                print(f"saved into {self.model_path['ir']}")
                f.write(r.content)

        if not self.model_path['clip'].is_file():
            print("You do not have the clip model, let me download that for you")
            url = CLIP_URL

            url += f"{self.scorer_model_name['clip']}?raw=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path['clip'].absolute(), "wb") as f:
                print(f"saved into {self.model_path['clip']}")
                f.write(r.content)

        if "blip" in self.cfg.scorer_method and not self.model_path['blip'].is_file():
            print("You do not have the blip model, let me download that for you")
            url = BLIP_URL

            url += f"{self.scorer_model_name['blip']}?raw=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path['blip'].absolute(), "wb") as f:
                print(f"saved into {self.model_path['blip']}")
                f.write(r.content)

    def load_models(self) -> None:
        if "aes" in self.cfg.scorer_method:
            print(f"Loading {self.scorer_model_name['aes']['laion']}")
            self.model['aes']['laion'] = AES(self.model_path['clip'], 'cuda')
            state_dict = torch.load(self.model_path['aes']['laion'], map_location='cuda')
            self.model['aes']['laion'].mlp.load_state_dict(state_dict, strict=False)
            self.model['aes']['laion'].mlp.to('cuda')

            print(f"Loading {self.scorer_model_name['aes']['chad']}")
            self.model['aes']['chad'] = AES(self.model_path['clip'], 'cuda')
            state_dict = torch.load(self.model_path['aes']['chad'], map_location='cuda')
            self.model['aes']['chad'].mlp.load_state_dict(state_dict, strict=False)
            self.model['aes']['chad'].mlp.to('cuda')

        if "ir" in self.cfg.scorer_method:
            print(f"Loading {self.scorer_model_name['ir']}")
            med_config = Path(
                self.cfg.scorer_model_dir,
                "med_config.json"
            )
            self.model['ir'] = IMGR(med_config, 'cuda').to('cuda')
            state_dict = torch.load(self.model_path['ir'], map_location='cuda')
            self.model['ir'].load_state_dict(state_dict, strict=False)

        if "clip" in self.cfg.scorer_method:
            print(f"Loading {self.scorer_model_name['clip']}")
            self.model['clip'] = CLP(self.model_path['clip'], 'cuda')

        if "blip" in self.cfg.scorer_method:
            print(f"Loading {self.scorer_model_name['blip']}")
            med_config = Path(
                self.cfg.scorer_model_dir,
                "med_config.json"
            )
            self.model['blip'] = BLP(med_config, 'cuda').to('cuda')
            state_dict = torch.load(self.model_path['blip'], map_location='cuda')
            self.model['blip'].load_state_dict(state_dict, strict=False)

    def score(self, image: Image.Image, prompt) -> float:
        score = 0
        nr = 0
        for eval in self.model:
            if eval == 'aes':
                #print(f"{eval}: {(self.model[eval]['laion'].score(prompt, image) + self.model[eval]['laion'].score(prompt, image)) / 2}")
                score += (self.model[eval]['laion'].score(prompt, image) + self.model[eval]['laion'].score(prompt, image)) / 2
            elif eval == 'ir' or eval == 'blip':
                tmp = (self.model[eval].score(prompt, image) + 2.5) * 2
                if tmp < 0:
                    tmp = 0
                if tmp > 10:
                    tmp = 10
                #print(f"{eval}: {tmp}")
                score += tmp
            elif eval == 'clip':
                #print(f"{eval}: {(self.model[eval].score(prompt, image) + 1) * 5}")
                score += (self.model[eval].score(prompt, image) + 1) * 5
            nr += 1
        return score / nr

    def batch_score(
            self,
            images: List[Image.Image],
            payload_names: List[str],
            payloads: Dict,
            it: int,
    ) -> List[float]:
        scores = []
        norm = []
        for i, (img, name, payload) in enumerate(zip(images, payload_names, payloads)):
            # in manual mode, we save a temp image first then request user input
            if self.cfg.scorer_method == "manual":
                tmp_path = Path(Path.cwd(), "tmp.png")
                img.save(tmp_path)
                self.open_image(tmp_path)
                score = AestheticScorer.get_user_score()
                tmp_path.unlink()  # remove temporary image
            else:
                score = self.score(img, payload["prompt"])
            if self.cfg.save_imgs:
                self.save_img(img, name, score, it, i, payload)

            if "score_weight" in payload:
                score *= payload["score_weight"]
                norm.append(payload["score_weight"])
            else:
                norm.append(1.0)
            scores.append(score)

            print(f"{name}-{i} {score:4.3f}")

        return scores, norm

    def average_score(self, scores: List[float], norm: List[float]) -> float:
        num = sum(scores)
        den = sum(norm)
        return 0.0 if den == 0.0 else num / den

    def image_path(self, name: str, score: float, it: int, batch_n: int) -> Path:
        return Path(
            self.imgs_dir,
            f"{it:03}-{batch_n:02}-{name}-{score:4.3f}.png",
        )

    def save_img(
            self,
            image: Image.Image,
            name: str,
            score: float,
            it: int,
            batch_n: int,
            payload: Dict,
    ) -> Path:
        img_path = self.image_path(name, score, it, batch_n)
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in payload.items():
            pnginfo.add_text(k, str(v))

        image.save(img_path, pnginfo=pnginfo)
        return img_path

    def open_image(self, image_path: Path) -> None:
        system = platform.system()

        if system == "Windows":
            subprocess.run(["start", str(image_path)], shell=True, check=True)
        elif system == "Linux":
            global printWSLFlag
            if ("microsoft-standard" in platform.uname().release) and printWSLFlag == 0:
                print(
                    "Make sure to install xdg-open-wsl from here: https://github.com/cpbotha/xdg-open-wsl otherwise the images will NOT open."
                )
                printWSLFlag = 1
            subprocess.run(["xdg-open", str(image_path)], check=True)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(image_path)], check=True)
        else:
            print(
                f"Sorry, we do not support opening images on '{system}' operating system."
            )

    @staticmethod
    def get_user_score() -> float:
        while True:
            try:
                score = float(
                    input(
                        f"\n\tPlease enter the score for the shown image (a number between 0 and 10)\n\t> "
                    )
                )
                if 0 <= score <= 10:
                    return score
                else:
                    print("\tInvalid input. Please enter a number between 0 and 10.")
            except ValueError:
                print("\tInvalid input. Please enter a number between 0 and 10.")
