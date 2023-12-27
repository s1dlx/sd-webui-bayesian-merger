import os
import safetensors
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.mixins import HyperparametersMixin
from PIL import Image
from transformers import pipeline, AutoConfig, AutoProcessor, BeitForImageClassification
import timm


class SyntheticModel(pl.LightningModule, HyperparametersMixin):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
                                       pretrained=False,
                                       num_classes=0)

        self.clf = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2))

    def forward(self, image):
        image_features = self.model(image)
        return self.clf(image_features)


class NoAIScore:
    def __init__(self, class_path, real_path, anime_path, device='cpu'):
        super().__init__()
        self.transform_m = None
        self.model_class = None
        self.model_real = None
        self.model_anime = None
        self.class_path = class_path
        self.real_path = real_path
        self.anime_path = anime_path
        self.device = device
        if self.device == 'cuda':
            self.device += ':0'
        self.initialize_model()

    def initialize_model(self):
        statedict = safetensors.torch.load_file(self.class_path)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path="cafeai/cafe_style")
        model = BeitForImageClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=statedict,
                                                           config=config)
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path="cafeai/cafe_style")
        self.model_class = pipeline("image-classification", model=model, image_processor=processor,
                                    device=self.device)

        statedict = safetensors.torch.load_file(self.anime_path)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path="saltacc/anime-ai-detect")
        model = BeitForImageClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=statedict,
                                                           config=config)
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path="saltacc/anime-ai-detect")
        self.model_anime = pipeline("image-classification", model=model, image_processor=processor,
                                    device=self.device)

        self.model_real = SyntheticModel()
        statedict = torch.load(self.real_path, map_location='cpu')
        self.model_real.load_state_dict(statedict)
        self.model_real = self.model_real.to(self.device)
        self.model_real.eval()

        transform_config = {'input_size': (3, 384, 384),
                            'interpolation': 'bicubic',
                            'mean': (0.48145466, 0.4578275, 0.40821073),
                            'std': (0.26862954, 0.26130258, 0.27577711),
                            'crop_pct': 1.0,
                            'crop_mode': 'squash'}

        self.transform_m = timm.data.create_transform(**transform_config, is_training=False)

    def score(self, prompt, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)

        tmp = self.model_class(images=[pil_image], top_k=5)[0]
        anime_prob = 0
        anime_prob += [p for p in tmp if p['label'] == 'anime'][0]['score']
        anime_prob += [p for p in tmp if p['label'] == '3d'][0]['score']
        anime_prob += [p for p in tmp if p['label'] == 'manga_like'][0]['score']
        anime_prob += [p for p in tmp if p['label'] == 'other'][0]['score'] / 2


        real_prob = 0
        real_prob += [p for p in tmp if p['label'] == 'real_life'][0]['score']
        real_prob += [p for p in tmp if p['label'] == 'other'][0]['score'] / 2


        tmp = self.model_anime(images=[pil_image], top_k=5)[0]
        anime_ai_score = [p for p in tmp if p['label'] == 'human'][0]['score']

        tmp = self.transform_m(pil_image)
        tmp = self.model_real.forward(tmp.unsqueeze(0).to(self.device))

        y_1 = F.softmax(tmp, dim=1)[:, 1].cpu().detach().numpy()
        y_2 = F.softmax(tmp, dim=1)[:, 0].cpu().detach().numpy()

        real_ai_score = y_2.tolist()[0]

        score = (real_prob * real_ai_score + anime_prob * anime_ai_score) * 10

        return score
