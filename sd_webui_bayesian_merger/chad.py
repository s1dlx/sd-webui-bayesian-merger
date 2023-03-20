import requests

from sd_webui_bayesian_merger.scorer import AestheticScorer


class ChadScorer(AestheticScorer):
    def get_model(self) -> None:
        # TODO: let user pick model
        # state_name = "sac+logos+ava1-l14-linearMSE.pth"
        if not self.model_path.is_file():
            print(
                "You do not have an aesthetic model ckpt, let me download that for you"
            )
            url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{self.model_name}?raw=true"
            r = requests.get(url)
            r.raise_for_status()

            with open(self.model_path.absolute(), "wb") as f:
                print(f"saved into {self.model_path}")
                f.write(r.content)

