from functools import partial

from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser

from hyperopt import Trials, hp, fmin, tpe, STATUS_OK


class TPEOptimiser(Optimiser):
    def _target_function(self, params):
        res = self.sd_target_function(**params)
        return {
            "loss": -res,
            "status": STATUS_OK,
            "params": params,
        }

    def optimise(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        space = {
            f"block_{i}": hp.uniform(f"block_{i}", 0.0, 1.0)
            for i in range(NUM_TOTAL_BLOCKS)
        }
        space["base_alpha"] = hp.uniform("base_alpha", 0.0, 1.0)

        self.trials = Trials()
        tpe._default_n_startup_jobs = self.cfg.init_points
        algo = partial(tpe.suggest, n_startup_jobs=self.cfg.init_points)
        fmin(
            self._target_function,
            space=space,
            algo=algo,
            trials=self.trials,
            max_evals=self.cfg.init_points + self.cfg.n_iters,
        )

        # clean up and remove the last merge
        try:
            self.cleanup()
        except FileNotFoundError:
            return

    def postprocess(self) -> None:
        print("\nRecap!")
        scores = []
        for i, res in enumerate(self.trials.losses()):
            print(f"Iteration {i} loss: \n\t{res}")
            scores.append(res)
        best = self.trials.best_trial

        best_base_alpha = best["result"]["params"]["base_alpha"]
        best_weights = [
            best["result"]["params"][f"block_{i}"] for i in range(NUM_TOTAL_BLOCKS)
        ]

        self.plot_and_save(
            scores,
            best_base_alpha,
            best_weights,
            minimise=True,
        )
