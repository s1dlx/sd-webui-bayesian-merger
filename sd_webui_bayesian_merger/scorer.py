import os
import requests

from typing import List
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline

import torch
import torch.nn as nn
import clip
import safetensors
import numpy as np

LAION_URL = (
    "https://github.com/Xerxemi/sdweb-auto-MBW/blob/master/scripts/classifiers/laion/"
)

CHAD_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/"
)

AES_URL = "https://raw.githubusercontent.com/Xerxemi/sdweb-auto-MBW/master/scripts/classifiers/aesthetic/aes-B32-v0.safetensors"


class AestheticClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = torch.nn.Linear(hidden_size // 2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


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
class AestheticScorer:
    scorer_method: str
    model_dir: os.PathLike
    model_name: str
    device: str

    def __post_init__(self):
        self.model_path = Path(self.model_dir, self.model_name)
        self.get_model()
        if not self.scorer_method.startswith("cafe"):
            self.load_model()

    def get_model(self) -> None:
        if self.scorer_method.startswith("cafe"):
            print("Creating scoring pipeline")
            self.judge = pipeline(
                "image-classification",
                model=f"cafeai/{self.scorer_method}",
            )
            return

        if self.model_path.is_file():
            return

        print("You do not have an aesthetic model ckpt, let me download that for you")
        if self.scorer_method == "chad":
            url = CHAD_URL
        elif self.scorer_method == "laion":
            url = LAION_URL
        elif self.scorer_method == "aes":
            url = AES_URL

        if self.scorer_method != 'aes':
            url += f"{self.model_name}?raw=true"

        r = requests.get(url)
        r.raise_for_status()

        with open(self.model_path.absolute(), "wb") as f:
            print(f"saved into {self.model_path}")
            f.write(r.content)

    def load_model(self) -> None:
        print(f"Loading {self.model_name}")

        if self.scorer_method in ["chad", "laion"]:
            self.model = AestheticPredictor(768).to(self.device).eval()
        elif self.scorer_method in ["aes"]:
            self.model = AestheticClassifier(512, 256, 1).to(self.device).eval()

        if self.model_path.suffix == ".safetensors":
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
            self.clip_model_name = "ViT-L/14"
        elif self.scorer_method in ["aes"]:
            self.clip_model_name = "openai/clip-vit-base-patch32"

        print(f"Loading {self.clip_model_name}")

        if self.scorer_method in ["chad", "laion"]:
            self.clip_model, self.clip_preprocess = clip.load(
                self.clip_model_name,
                device=self.device,
            )
        elif self.scorer_method in ["aes"]:
            self.clip_model = (
                CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
            )
            self.clip_preprocess = CLIPProcessor.from_pretrained(self.clip_model_name)

    def get_image_features(self, image: Image.Image) -> torch.Tensor:
        if self.scorer_method in ["chad", "laion"]:
            image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()
            return image_features
        elif self.scorer_method in ["aes"]:
            inputs = self.clip_preprocess(images=image, return_tensors="pt")[
                "pixel_values"
            ].to(self.device)
            result = (
                self.clip_model.get_image_features(pixel_values=inputs)
                .cpu()
                .detach()
                .numpy()
            )
            return (result / np.linalg.norm(result)).squeeze(axis=0)

    def score(self, image: Image.Image) -> float:
        if self.scorer_method.startswith("cafe"):
            # TODO: this returns also a 'label', what can we do with it?
            # TODO: does it make sense to use top_k != 1?
            data = self.judge(image, top_k=1)
            return data[0]["score"]

        image_features = self.get_image_features(image)
        score = self.model(
            torch.from_numpy(image_features).to(self.device).float(),
        )

        return score.item()

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        return [self.score(img) for img in images]

    def average_score(self, scores: List[float]) -> float:
        return sum(scores) / len(scores)
