from PIL import Image
import torch
import torch.nn as nn
import clip

from pathlib import Path
import requests


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


class Scorer:
    def __init__(self, model_dir, device):
        self.device = device
        self.model_dir = Path(model_dir)
        self.get_models()
        self.load_model()
        self.load_clip()

    def get_models(self):
        # TODO: let user pick model
        state_name = "sac+logos+ava1-l14-linearMSE.pth"
        if not Path(self.model_dir, state_name).is_file():
            print(
                "You do not have an aesthetic model ckpt, let me download that for you"
            )
            url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
            r = requests.get(url)
            self.model_path = Path("./models", state_name).absolute()
            with open(self.model_path, "wb") as f:
                print(f"saved into {self.model_path}")
                f.write(r.content)
        else:
            self.model_path = Path(self.model_dir, state_name).absolute()

    def load_model(self):
        print("Loading aestetic scorer model")
        pt_state = torch.load(self.model_path, map_location=self.device)
        self.model = AestheticPredictor(768)
        self.model.load_state_dict(pt_state)
        self.model.to(self.device)
        self.model.eval()

    def load_clip(self):
        print("Loading CLIP")
        # TODO: let user pick model
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-L/14", device=self.device
        )

    def get_image_features(self, image: Image) -> torch.Tensor:
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        return image_features

    def score(self, image: Image) -> float:
        image_features = self.get_image_features(image)
        score = self.model(torch.from_numpy(image_features).to(self.device).float())
        return score.item()

    def batch_score(self, images: [Image]) -> [float]:
        return [self.score(img) for img in images]

    def average_score(self, scores: [float]) -> float:
        return sum(scores) / len(scores)
