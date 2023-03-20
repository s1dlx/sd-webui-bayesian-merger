import os

from typing import List
from abc import abstractmethod
from dataclasses import dataclass

from PIL import Image

@dataclass
class Scorer:
    device: str
    model_dir: os.PathLike

    def __post_init__(self):
        self.get_model()
        self.load_model()

    @abstractmethod
    def get_model(self) -> None:
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def score(self, image: Image.Image) -> float:
        raise NotImplementedError("Not Implemented")

    def batch_score(self, images: List[Image.Image]) -> List[float]:
        return [self.score(img) for img in images]

    def average_score(self, scores: List[float]) -> float:
        return sum(scores) / len(scores)
