"""
@File       :   Laion.py
@Time       :   2023/02/12 14:54:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   AestheticScore.
* Based on improved-aesthetic-predictor code base
* https://github.com/christophschuhmann/improved-aesthetic-predictor
"""
import os

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class Laion(nn.Module):
    def __init__(self, pathname, clip_path, device):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_path, device=self.device, jit=False)
        self.mlp = MLP(768)
        state_dict = torch.load(pathname, map_location='cpu')
        self.mlp.load_state_dict(state_dict, strict=False)
        self.mlp = self.mlp.to(self.device)
        self.mlp.eval()

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, image):

        if (type(image).__name__ == 'list'):
            _, rewards = self.inference_rank(prompt, image)
            return rewards
            # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image)).float()

        # score
        with torch.no_grad():
            rewards = self.mlp(image_features)

        return rewards.detach().cpu().numpy().item()

    def inference_rank(self, prompt, generations_list):

        img_set = []
        for generations in generations_list:
            # image encode
            img_path = generations
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image))
            img_set.append(image_features)

        img_features = torch.cat(img_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(img_features)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()
