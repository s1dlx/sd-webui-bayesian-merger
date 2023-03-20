from unittest.mock import patch

from sd_webui_bayesian_merger import TPEOptimiser


def mock_target_function(self, **params):
    # just return the average dist from 0.5
    diffs = [abs(0.5 - v) for v in params.values()]
    return sum(diffs) / len(diffs)


def mock_null(*args, **kwargs):
    pass


def test_tpe_optimiser():
    with patch.object(TPEOptimiser, "__post_init__", mock_null), patch.object(
        TPEOptimiser, "sd_target_function", mock_target_function
    ), patch.object(TPEOptimiser, "_cleanup", mock_null):
        optimiser = TPEOptimiser(
            url="",
            batch_size=0,
            model_a="test-a",
            model_b="test-b",
            device="CPU",
            payloads_dir="",
            wildcards_dir="",
            scorer_model_dir="",
            init_points=15,
            n_iters=16,
            skip_position_ids=0,
            best_format="",
            best_precision="",
            save_best=False,
            method="tpe",
        )
        optimiser.init_merger()
        optimiser.optimise()
        print("Best run:", optimiser.trials.best_trial)
        optimiser.postprocess()


if __name__ == "__main__":
    test_tpe_optimiser()
