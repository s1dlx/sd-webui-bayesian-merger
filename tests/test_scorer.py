from sd_webui_bayesian_merger.scorer import AestheticScorer


def test_aesthetic_scorer():
    cs = AestheticScorer(
        scorer_method="chad",
        model_dir="./models/",
        model_name="sac+logos+ava1-l14-linearMSE.pth",
        device="cpu",
    )
