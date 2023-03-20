from unittest.mock import patch

from sd_webui_bayesian_merger.chad import ChadScorer


def mock_target_function(self, **params):
    # just return the average dist from 0.5
    diffs = [abs(0.5 - v) for v in params.values()]
    return sum(diffs) / len(diffs)


def mock_null(*args, **kwargs):
    pass


def test_aesthetic_scorer():
    cs = ChadScorer(
        model_dir = './models/',
        model_name = 'sac+logos+ava1-l14-linearMSE.pth',
        clip_model = 'ViT-L/14',
        device = 'cpu',
    )


