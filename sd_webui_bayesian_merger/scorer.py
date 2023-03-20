import os

from typing import List
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import clip
import safetensors

LAION_URL = "https://github.com/Xerxemi/sdweb-auto-MBW/blob/master/scripts/classifiers/laion/"

CHAD_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/"

# from https://github.com/grexzen/SD-Chad
class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

@dataclass
class AestheticScorer():
    scorer_method: str
    model_dir: os.PathLike
    model_name: str
    device: str

    def __post_init__(self):
        self.model_path = Path(self.model_dir, self.model_name)
        self.get_model()
        self.load_model()

    def get_model(self) -> None:
        if self.model_path.is_file():
            return
        print(
            "You do not have an aesthetic model ckpt, let me download that for you"
        )
        if self.scorer_method == "chad":
            url = "{CHAD_URL}"
        elif self.scorer_method == "laion":
            url = "{LAION_URL}"
        url += f"{self.model_name}?raw=true"

        r = requests.get(url)
        r.raise_for_status()

        with open(self.model_path.absolute(), "wb") as f:
            print(f"saved into {self.model_path}")
            f.write(r.content)

    def load_model(self) -> None:
        print(f"Loading {self.model_name}")
        self.model = AestheticPredictor(768).to(self.device).eval()
        if self.model_path.suffix == ".safetensors":
            # pt_path = Path.join(dirname, "laion-sac-logos-ava-v2.safetensors")
            self.model.load_state_dict(
                safetensors.torch.load_file(
                    self.model_path,
                )
            )
            self.model.to(self.device)
        else:
            self.model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location=self.device,
                )
            )
        self.model.eval()
        self.load_clip()

    def load_clip(self) -> None:
        if self.scorer_method in ["chad", "laion"]:
            self.clip_model = "ViT-L/14"
        print(f"Loading {self.clip_model}")
        self.clip_model, self.clip_preprocess = clip.load(
            self.clip_model,
            device=self.device,
        )

    def get_image_features(self, image: Image.Image) -> torch.Tensor:
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        return image_features

    def score(self, image: Image.Image) -> float:
        image_features = self.get_image_features(image)
        score = self.model(
            torch.from_numpy(image_features).to(self.device).float(),
        )

        return score.item()

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        return [self.score(img) for img in images]

    def average_score(self, scores: List[float]) -> float:
        return sum(scores) / len(scores)


