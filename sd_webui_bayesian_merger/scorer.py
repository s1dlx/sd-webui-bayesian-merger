import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import requests
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from PIL import Image, PngImagePlugin
from sd_webui_bayesian_merger.models.Laion import Laion as AES
from sd_webui_bayesian_merger.models.ImageReward import ImageReward as IMGR
from sd_webui_bayesian_merger.models.CLIPScore import CLIPScore as CLP
from sd_webui_bayesian_merger.models.BLIPScore import BLIPScore as BLP
from sd_webui_bayesian_merger.models.HPSv2 import HPSv2 as HPS
from sd_webui_bayesian_merger.models.PickScore import PickScore as PICK
from sd_webui_bayesian_merger.models.WDAes import WDAes as WDA
from sd_webui_bayesian_merger.models.ShadowScore import ShadowScore as SS
from sd_webui_bayesian_merger.models.CafeScore import CafeScore as CAFE

LAION_URL = (
    "https://github.com/grexzen/SD-Chad/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true"
)
CHAD_URL = (
    "https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth?raw=true"
)
WDAES_URL = (
    "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth?download=true"
)
IR_URL = (
    "https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt?download=true"
)
CLIP_URL = (
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt?raw=true"
)
BLIP_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth?raw=true"
)
HPSV2_URL = (
    "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt?download=true"
)
PICK_URL = (
    "https://huggingface.co/yuvalkirstain/PickScore_v1/resolve/main/model.safetensors?download=true"
)
SHADOW_URL = (
    "https://huggingface.co/shadowlilac/aesthetic-shadow/resolve/main/model.safetensors?download=true"
)
CAFE_URL = (
    "https://huggingface.co/cafeai/cafe_aesthetic/resolve/3bca27c5c0b6021056b1e84e5a18cf1db9fe5d4c/model.safetensors?download=true"
)

LAION_MODEL = (
    "Laion.pth"
)
CHAD_MODEL = (
    "Chad.pth"
)
WDAES_MODEL = (
    "WD_Aes.pth"
)
IR_MODEL = (
    "ImageReward.pt"
)
CLIP_MODEL = (
    "CLIP-ViT-L-14.pt"
)
BLIP_MODEL = (
    "BLIP_Large.pth"
)
HPSV2_MODEL = (
    "HPS_v2.1.pt"
)
PICK_MODEL = (
    "Pick-A-Pic.safetensors"
)
SHADOW_MODEL = (
    "Shadow.safetensors"
)
CAFE_MODEL = (
    "Cafe.safetensors"
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

        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                self.scorer_model_name[evaluator] = eval(f"{evaluator.upper() + '_MODEL'}")
                self.model_path[evaluator] = Path(
                    self.cfg.scorer_model_dir,
                    self.scorer_model_name[evaluator],
                )
        if 'clip' not in self.cfg.scorer_method and any(
                x in ['laion', 'chad'] for x in self.cfg.scorer_method):
            self.model_path['clip'] = Path(
                self.cfg.scorer_model_dir,
                CLIP_MODEL,
            )

        self.get_models()
        self.load_models()

    def get_models(self) -> None:
        blip_config = Path(
            self.cfg.scorer_model_dir,
            'med_config.json',
        )
        if not blip_config.is_file():
            url = "https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json?download=true"

            r = requests.get(url)
            r.raise_for_status()

            with open(blip_config.absolute(), "wb") as f:
                print(f"saved into {blip_config}")
                f.write(r.content)

        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                if not self.model_path[evaluator].is_file():
                    print(f"You do not have the {evaluator.upper()} model, let me download that for you")
                    url = eval(f"{evaluator.upper() + '_URL'}")

                    r = requests.get(url)
                    r.raise_for_status()

                    with open(self.model_path[evaluator].absolute(), "wb") as f:
                        print(f"saved into {self.model_path[evaluator]}")
                        f.write(r.content)

                if evaluator == 'wdaes':
                    clip_vit_b_32 = Path(
                        self.cfg.scorer_model_dir,
                        "CLIP-ViT-B-32.safetensors",
                    )
                    if not clip_vit_b_32.is_file():
                        print(f"You do not have the CLIP-ViT-B-32 necessary for the wdaes model, let me download that for you")
                        url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/b527df4b30e5cc18bde1cc712833a741d2d8c362/model.safetensors?download=true"

                        r = requests.get(url)
                        r.raise_for_status()

                        with open(clip_vit_b_32.absolute(), "wb") as f:
                            print(f"saved into {clip_vit_b_32}")
                            f.write(r.content)

        if ('clip' not in self.cfg.scorer_method and
                any(x in ['laion', 'chad'] for x in self.cfg.scorer_method)):
            if not self.model_path['clip'].is_file():
                print(f"You do not have the CLIP(which you need) model, let me download that for you")
                url = CLIP_URL

                r = requests.get(url)
                r.raise_for_status()

                with open(self.model_path['clip'].absolute(), "wb") as f:
                    print(f"saved into {self.model_path['clip']}")
                    f.write(r.content)

    def load_models(self) -> None:
        med_config = Path(
            self.cfg.scorer_model_dir,
            "med_config.json"
        )
        for evaluator in self.cfg.scorer_method:
            if evaluator != 'manual':
                print(f"Loading {self.scorer_model_name[evaluator]}")
            if evaluator == 'wdaes':
                clip_vit_b_32 = Path(
                    self.cfg.scorer_model_dir,
                    "CLIP-ViT-B-32.safetensors",
                )
                self.model[evaluator] = WDA(self.model_path[evaluator], clip_vit_b_32, self.cfg.scorer_device[evaluator])
            elif evaluator == 'clip':
                self.model[evaluator] = CLP(self.model_path[evaluator], self.cfg.scorer_device[evaluator])
            elif evaluator == 'blip':
                self.model[evaluator] = BLP(self.model_path[evaluator], med_config, self.cfg.scorer_device[evaluator])
            elif evaluator == 'ir':
                self.model[evaluator] = IMGR(self.model_path[evaluator], med_config, self.cfg.scorer_device[evaluator])
            elif evaluator == 'laion' or evaluator == 'chad':
                self.model[evaluator] = AES(self.model_path[evaluator], self.model_path['clip'],
                                            self.cfg.scorer_device[evaluator])
            elif evaluator == 'hpsv2':
                self.model[evaluator] = HPS(self.model_path[evaluator], self.cfg.scorer_device[evaluator])
            elif evaluator == 'pick':
                self.model[evaluator] = PICK(self.model_path[evaluator], self.cfg.scorer_device[evaluator])
            elif evaluator == 'shadow':
                self.model[evaluator] = SS(self.model_path[evaluator], self.cfg.scorer_device[evaluator])
            elif evaluator == 'cafe':
                self.model[evaluator] = CAFE(self.model_path[evaluator], self.cfg.scorer_device[evaluator])

    def score(self, image: Image.Image, prompt) -> float:
        values = []
        weights = []
        for evaluator in self.cfg.scorer_method:
            weights.append(int(self.cfg.scorer_weights[evaluator]))
            if evaluator == 'manual':
                # in manual mode, we save a temp image first then request user input
                tmp_path = Path(Path.cwd(), "tmp.png")
                image.save(tmp_path)
                self.open_image(tmp_path)
                values.append(self.get_user_score())
                tmp_path.unlink()  # remove temporary image
            else:
                values.append(self.model[evaluator].score(prompt, image))

            if self.cfg.scorer_print_individual:
                print(f"{evaluator}:{values[-1]}")

        score = self.average_calc(values, weights, self.cfg.scorer_average_type)
        return score

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
            score = self.score(img, payload["prompt"])
            if self.cfg.save_imgs:
                self.save_img(img, name, score, it, i, payload)

            if "score_weight" in payload:
                norm.append(payload["score_weight"])
            else:
                norm.append(1.0)
            scores.append(score)

            print(f"{name}-{i} {score:4.3f}")

        return scores, norm

    def average_calc(self, values: List[float], weights: List[float], average_type: str) -> float:
        norm = 0
        for weight in weights:
            norm += weight
        avg = 0
        if average_type == 'geometric':
            avg = 1
        elif average_type == 'arithmetic' or average_type == 'exponential':
            avg = 0

        for value, weight in zip(values, weights):
            if average_type == 'arithmetic':
                avg += value * weight
            elif average_type == 'geometric':
                avg *= value ** weight
            elif average_type == 'exponential':
                avg += (value ** norm) * weight

        if average_type == 'arithmetic':
            avg = avg / norm
        elif self.cfg.scorer_average_type == 'geometric':
            avg = avg ** (1 / norm)
        elif self.cfg.scorer_average_type == 'exponential':
            avg = (avg / norm) ** (1 / norm)
        return avg

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
