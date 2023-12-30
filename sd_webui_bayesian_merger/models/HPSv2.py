import os
import open_clip
import torch
from PIL import Image
from open_clip import image_transform


class HPSv2:
    def __init__(self, pathname, device='cpu'):
        super().__init__()
        self.tokenizer = None
        self.model_dict = {}
        self.device = device
        self.pathname = pathname
        self.initialize_model()

    def initialize_model(self):
        if not self.model_dict:
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                'ViT-H-14',
                pretrained=str(self.pathname),
                precision='amp',
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                aug_cfg={},
                output_dict=True,
            )
            preprocess_val = image_transform(
                model.visual.image_size,
                is_train=False,
                mean=None,
                std=None,
                resize_longest_max=True,
            )

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

        image = preprocess_val(pil_image).unsqueeze(0).to(device=self.device, non_blocking=True)
        with torch.no_grad():
            text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]

                scores = torch.sum(torch.mul(image_features, text_features), dim=1, keepdim=True)
        score = scores.cpu().tolist()[0][0]
        score += 1
        score *= 5
        return score
