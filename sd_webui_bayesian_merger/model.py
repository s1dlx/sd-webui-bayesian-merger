from pathlib import Path

import torch
import safetensors


class SDModel:
    def __init__(self, model_path: str, device: str) -> None:
        model_path = Path(model_path)
        self.model = self.load_model()

    def load_model(self, model_path, device):
        print(f"loading: {model_path}")
        if self.model_path.suffix == ".safetensors":
            ckpt = safetensors.torch.load_file(
                model_path,
                device=device,
            )
        else:
            ckpt = torch.load(model_path, map_location=device)

        return get_state_dict_from_checkpoint(ckpt)


# TODO: tidy up
# from: stable-diffusion-webui/modules/sd_models.py
def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


chckpoint_dict_replacements = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text) :]
    return k
