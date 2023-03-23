from pathlib import Path


extension_path = Path(__file__).parent.parent.resolve()


class DefaultCliArgs:
    config = "config.ini"
    url = "http://127.0.0.1:7860"
    skip_position_ids = 0
    device = 'cpu'
    payloads_dir = extension_path / "payloads"
    wildcards_dir = extension_path / "wildcards"
    scorer_model_dir = extension_path / "models"
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
