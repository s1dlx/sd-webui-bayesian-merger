import os
import open_clip
import safetensors
import torch
from PIL import Image
from open_clip import image_transform
from transformers import AutoModel, AutoProcessor, AutoConfig


class OPENCLIPScore:
    def __init__(self, pathname, device='cpu', model_type='hpsv2'):
        super().__init__()
        self.tokenizer = None
        self.model_dict = {}
        self.device = device
        self.pathname = pathname
        self.model_type = model_type
        self.initialize_model()

    def initialize_model(self):
        if not self.model_dict:
            if self.model_type == 'hpsv2':
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
            elif self.model_type == 'pick':
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

        if self.model_type == 'hpsv2':
            image = preprocess_val(pil_image).unsqueeze(0).to(device=self.device, non_blocking=True)
            with torch.no_grad():
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    scores = torch.sum(torch.mul(image_features, text_features), dim=1, keepdim=True)
            score = scores.cpu().tolist()[0][0]
            score += 1
            score *= 5
            return score
        elif self.model_type == 'pick':
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

                # score
                #scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                scores = torch.sum(torch.mul(text_embs, image_embs), dim=1, keepdim=True)

            score = scores.cpu().tolist()[0][0]
            score += 1
            score *= 5
            return score
