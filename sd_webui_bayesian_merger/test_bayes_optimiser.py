from unittest.mock import patch

from sd_webui_bayesian_merger import BayesOptimiser

def mock_target_function(self, **params):
    # just return the average dist from 0.5
    diffs = []
    for k, v in params.items():
        diffs.append(abs(0.5 - v))
    return sum(diffs) / len(diffs)

def mock_null(*args, **kwargs):
    pass

def test_tpe_optimiser():

    with patch.object(BayesOptimiser, '__post_init__', mock_null),\
        patch.object(BayesOptimiser, 'sd_target_function', mock_target_function), \
            patch.object(BayesOptimiser, '_cleanup', mock_null):
            optimiser = BayesOptimiser(
                url='',
                batch_size=0,
                model_a='test-a',
                model_b='test-b',
                device='CPU',
                payloads_dir='',
                wildcards_dir='',
                scorer_model_dir='',
                init_points=15,
                n_iters=64,
                skip_position_ids=0,
                method='bayes'
            )
            optimiser.init_merger()
            optimiser.start_logging()
            optimiser.optimise()
            print("Best run:", optimiser.optimizer.max)
            optimiser.postprocess()

if __name__ == "__main__":
    test_tpe_optimiser()
