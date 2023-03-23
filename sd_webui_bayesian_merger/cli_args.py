from pathlib import Path
import torch


extension_dir = Path(__file__).parent.parent.resolve()


class Defaults:
    config = extension_dir / "config.ini"
    url = "http://127.0.0.1:7860"
    skip_position_ids = 0
    device = 'cpu'
    payloads_dir = extension_dir / "payloads"
    wildcards_dir = extension_dir / "wildcards"
    scorer_model_dir = extension_dir / "models"
    optimiser = "bayes"
    batch_size = 1
    init_points = 1
    n_iters = 1
    save_imgs = False
    scorer_method = "chad"
    scorer_model_name = "sac+logos+ava1-l14-linearMSE.pth"
    save_best = False
    best_format = "safetensors"
    best_precision = "16"
    draw_unet_weights = None
    draw_unet_base_alpha = None


class Choices:
    optimiser = [
        "bayes",
        "tpe",
    ]
    scorer_method = [
        "chad",
        "laion",
        "aes",
        "cafe_aesthetic",
        "cafe_style",
        "cafe_waifu",
    ]
    scorer_model_name = [
        "sac+logos+ava1-l14-linearMSE.pth",  # chad
        "ava+logos-l14-linearMSE.pth",
        "ava+logos-l14-reluMSE.pth",
    ]
    device = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    best_format = [
        "safetensors",
        "ckpt",
    ]
    best_precision = [
        "16",
        "32",
    ]
