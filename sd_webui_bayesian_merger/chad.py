import requests

from pathlib import Path

from sd_webui_bayesian_merger.scorer import AestheticScorer


class ChadScorer(AestheticScorer):
    def __init__(self):
        super().__init__(**kwargs)
        
    def get_model(self) -> None:
        # TODO: let user pick model
        state_name = "sac+logos+ava1-l14-linearMSE.pth"
        if not Path(self.model_dir, state_name).is_file():
            print(
                "You do not have an aesthetic model ckpt, let me download that for you"
            )
            url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
            r = requests.get(url)
            r.raise_for_status()

            self.model_path = Path("./models", state_name).absolute()
            with open(self.model_path, "wb") as f:
                print(f"saved into {self.model_path}")
                f.write(r.content)
        else:
            self.model_path = Path(self.model_dir, state_name).absolute()

