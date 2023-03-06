import re
import torch
import safetensors.torch
from tqdm import tqdm

from sd_webui_bayesian_merger.model import SDModel

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)


class Merger:
    def __init__(
        self,
        model_a: str,
        model_b: str,
        device: str,
        output_file: str,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.device = device
        self.output_file = output_file

        # TODO: add as parameter?
        self.skip_position_ids = 0

    def merge(
        self,
        weights: [float],
        base_alpha: int,
    ) -> None:
        if len(weights) != NUM_TOTAL_BLOCKS:
            _err_msg = f"weights value must be {NUM_TOTAL_BLOCKS}."
            print(_err_msg)
            return False, _err_msg

        theta_0 = SDModel(self.model_a, self.device).load_model()
        theta_1 = SDModel(self.model_b, self.device).load_model()
        alpha = base_alpha

        if not self.output_file:
            model_a_name = self.model_a.stem
            model_b_name = self.model_b.stem
            self.output_file = f"bbwm-{model_a_name}-{model_b_name}.safetensors"

        re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
        re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
        re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

        for key in tqdm(theta_0.keys(), desc="merging 1/2"):
            if "model" in key and key in theta_1:
                if KEY_POSITION_IDS in key and self.skip_position_ids in [1, 2]:
                    if self.skip_position_ids == 2:
                        theta_0[key] = torch.tensor(
                            [list(range(77))], dtype=torch.int64
                        )
                    continue

                current_alpha = alpha

                if "model.diffusion_model." in key:
                    weight_index = -1

                    if "time_embed" in key:
                        weight_index = 0  # before input blocks
                    elif ".out." in key:
                        weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
                    elif m := re_inp.search(key):
                        weight_index = int(m.groups()[0])
                    else:
                        if re_mid.search(key):
                            weight_index = NUM_INPUT_BLOCKS
                        elif m := re_out.search(key):
                            weight_index = (
                                NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.groups()[0])
                            )

                    if weight_index >= NUM_TOTAL_BLOCKS:
                        raise ValueError(f"illegal block index {key}")

                    if weight_index >= 0:
                        current_alpha = weights[weight_index]

                theta_0[key] = (1 - current_alpha) * theta_0[
                    key
                ] + current_alpha * theta_1[key]

                theta_0[key] = theta_0[key].half()

        for key in tqdm(theta_1.keys(), desc="merging 2/2"):
            if "model" in key and key not in theta_0:
                if KEY_POSITION_IDS in key and self.skip_position_ids in [1, 2]:
                    if self.skip_position_ids == 2:
                        theta_1[key] = torch.tensor(
                            [list(range(77))], dtype=torch.int64
                        )
                    continue
                theta_0.update({key: theta_1[key]})
                theta_0[key] = theta_0[key].half()

        print(f"Saving {self.output_file}")
        safetensors.torch.save_file(
            theta_0,
            self.output_file,
            metadata={"format": "pt"},
        )
