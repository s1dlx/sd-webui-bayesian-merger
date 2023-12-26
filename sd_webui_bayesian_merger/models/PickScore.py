import os
import open_clip
import safetensors
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig


class PickScore:
    def __init__(self, pathname, device='cpu'):
        super().__init__()
        self.tokenizer = None
        self.model_dict = {}
        self.device = device
        self.pathname = pathname
        self.initialize_model()

    def initialize_model(self):
        if not self.model_dict:
            statedict = safetensors.torch.load_file(self.pathname)
            config_pick = AutoConfig.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1")
            model = AutoModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=statedict,
                                              config=config_pick)
            preprocess_val = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

            self.model_dict['model'] = model
            self.model_dict['preprocess_val'] = preprocess_val

            self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
            self.model_dict['model'] = model.to(self.device)
            self.model_dict['model'].eval()

    def score(self, prompt, image):
        preprocess_val = self.model_dict['preprocess_val']
        model = self.model_dict['model']
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)

        image_inputs = preprocess_val(
            images=pil_image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = preprocess_val(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = torch.sum(torch.mul(text_embs, image_embs), dim=1, keepdim=True)

        score = scores.cpu().tolist()[0][0]
        score += 1
        score *= 5
        return score
