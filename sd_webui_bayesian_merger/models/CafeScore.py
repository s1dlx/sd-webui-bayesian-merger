import os
import safetensors
import torch
from PIL import Image
from transformers import pipeline, AutoConfig, AutoProcessor, BeitForImageClassification


class CafeScore:
    def __init__(self, pathname, device='cpu'):
        super().__init__()
        self.tokenizer = None
        self.pipe = None
        self.device = device
        if self.device == 'cuda':
            self.device += ':0'
        self.pathname = pathname
        self.initialize_model()

    def initialize_model(self):
        statedict = safetensors.torch.load_file(self.pathname)
        config_pick = AutoConfig.from_pretrained(pretrained_model_name_or_path="cafeai/cafe_aesthetic")
        model = BeitForImageClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=statedict, config=config_pick)
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path="cafeai/cafe_aesthetic")
        self.pipe = pipeline("image-classification", model=model, image_processor=processor, device=self.device)

    def score(self, prompt, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)

        score = self.pipe(images=[pil_image], top_k=2)[0]
        score = [p for p in score if p['label'] == 'aesthetic'][0]['score']
        score *= 10

        return score
