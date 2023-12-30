import os

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPConfig, CLIPImageProcessor
import numpy as np


class Classifier(torch.nn.Module):
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


class WDAes(nn.Module):
    def __init__(self, pathname, clip_path, device='cpu'):
        super().__init__()
        self.device = device
        self.preprocess = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
        config = CLIPConfig.from_pretrained(pretrained_model_name_or_path="openai/clip-vit-base-patch32")
        state_dict = safetensors.torch.load_file(clip_path)
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=state_dict, config=config)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.mlp = Classifier(512, 256, 1)
        state_dict = torch.load(pathname, map_location='cpu')
        self.mlp.load_state_dict(state_dict, strict=False)
        self.mlp = self.mlp.to('cpu')
        self.mlp.eval()

        if self.device == "cpu":
            self.clip_model.float()

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, image):

        if (type(image).__name__ == 'list'):
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        with torch.no_grad():
            # image encode
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, str):
                if os.path.isfile(image):
                    pil_image = Image.open(image)
            image = self.preprocess(images=pil_image, return_tensors='pt')['pixel_values']
            image = image.to(self.device)
            image_features = self.clip_model.get_image_features(pixel_values=image).cpu().detach().numpy()

            rewards = (image_features / np.linalg.norm(image_features)).squeeze(axis=0)
            reward = self.mlp(torch.from_numpy(rewards)).float().item()

            reward = reward * 10
            return reward

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
