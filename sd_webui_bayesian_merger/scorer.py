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
from sd_webui_bayesian_merger.models.AestheticScore import AestheticScore as AES
from sd_webui_bayesian_merger.models.ImageReward import ImageReward as IMGR
from sd_webui_bayesian_merger.models.CLIPScore import CLIPScore as CLP
from sd_webui_bayesian_merger.models.BLIPScore import BLIPScore as BLP
from sd_webui_bayesian_merger.models.OPENCLIPScore import OPENCLIPScore as OPS

LAION_URL = (
    "https://github.com/grexzen/SD-Chad/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true"
)
CHAD_URL = (
    "https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth?raw=true"
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
    "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt?download=true"
)
PICK_URL = (
    "https://huggingface.co/yuvalkirstain/PickScore_v1/resolve/main/model.safetensors?download=true"
)
LAION_MODEL = (
    "sac+logos+ava1-l14-linearMSE.pth"
)
CHAD_MODEL = (
    "chadscorer.pth"
)
IR_MODEL = (
    "ImageReward.pt"
)
CLIP_MODEL = (
    "ViT-L-14.pt"
)
BLIP_MODEL = (
    "model_large.pth"
)
HPSV2_MODEL = (
    "HPS_v2_compressed.pt"
)
PICK_MODEL = (
    "model.safetensors"
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
            self.scorer_model_name[evaluator] = eval(f"{evaluator.upper() + '_MODEL'}")
            self.model_path[evaluator] = Path(
                self.cfg.scorer_model_dir,
                self.scorer_model_name[evaluator],
            )
        if bool(self.model_path['clip']):
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
            if not self.model_path[evaluator].is_file():
                print(f"You do not have the {evaluator.upper()} model, let me download that for you")
                url = eval(f"{evaluator.upper() + '_URL'}")

                r = requests.get(url)
                r.raise_for_status()

                with open(self.model_path[evaluator].absolute(), "wb") as f:
                    print(f"saved into {self.model_path[evaluator]}")
                    f.write(r.content)
        if not self.model_path['clip'].is_file():
            print(f"You do not have the CLIP(which you need) model, let me download that for you")
            url = CLIP_URL

            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path['clip'].absolute(), "wb") as f:
                print(f"saved into {self.model_path['clip']}")
                f.write(r.content)

    def load_models(self) -> None:

        for evaluator in self.cfg.scorer_method:
            print(f"Loading {self.scorer_model_name[evaluator]}")
            if evaluator == 'blip' or evaluator == 'ir':
                med_config = Path(
                    self.cfg.scorer_model_dir,
                    "med_config.json"
                )
                if evaluator == 'blip':
                    self.model[evaluator] = BLP(med_config, self.cfg.scorer_device[evaluator]).to(
                        self.cfg.scorer_device[evaluator])
                elif evaluator == 'ir':
                    self.model[evaluator] = IMGR(med_config, self.cfg.scorer_device[evaluator]).to(
                        self.cfg.scorer_device[evaluator])
                state_dict = torch.load(self.model_path[evaluator], map_location='cpu')
                self.model[evaluator].load_state_dict(state_dict, strict=False)
            elif evaluator == 'laion' or evaluator == 'chad':
                self.model[evaluator] = AES(self.model_path['clip'], self.cfg.scorer_device[evaluator])
                state_dict = torch.load(self.model_path[evaluator], map_location='cpu')
                self.model[evaluator].mlp.load_state_dict(state_dict, strict=False)
                self.model[evaluator].mlp.to(self.cfg.scorer_device[evaluator])
            elif evaluator == 'clip':
                self.model[evaluator] = CLP(self.model_path[evaluator], self.cfg.scorer_device[evaluator])
            elif evaluator == 'hpsv2' or evaluator == 'pick':
                self.model[evaluator] = OPS(self.model_path[evaluator], self.cfg.scorer_device[evaluator], evaluator)

    def score(self, image: Image.Image, prompt) -> float:
        values = []
        weights = []
        for evaluator in self.model:
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
