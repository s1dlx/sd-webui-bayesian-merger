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
        if self.cfg.enable_fix_blocks:
            space = {}
            for i in range(NUM_TOTAL_BLOCKS):
                block_name = f"block_{i}"
                if block_name not in self.cfg.fix_blocks:
                    for gl in self.merger.greek_letters:
                        space[f"{block_name}_{gl}"] = hp.uniform(f"{block_name}_{gl}")
            for gl in self.merger.greek_letters:
                if f"base_{gl}" not in self.cfg.fix_blocks:
                    space[f"base_{gl}"] = hp.uniform(f"base_{gl}")
        else:
            space = {
                f"block_{i}_alpha": hp.uniform(f"block_{i}_alpha",0.0,1.0)
                for i in range(NUM_TOTAL_BLOCKS)
            }
            space["base_alpha"] = hp.uniform("base_alpha",0.0,1.0)
            for gl in self.merger.greek_letters:
                space |= {
                    f"block_{i}_{gl}": hp.uniform(f"block{i}_{gl}",0.0,1.0)
                    for i in range(NUM_TOTAL_BLOCKS)
                }
                space[f"base_{gl}"] = hp.uniform(f"base_{gl}",0.0,1.0)

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

        best_bases = {
            gl: best["result"]["params"][f"base_{gl}"]
            for gl in self.merger.greek_letters
        }
        best_weights = {
            gl: [
                best["result"]["params"][f"block_{i}_{gl}"]
                for i in range(NUM_TOTAL_BLOCKS)
            ]
            for gl in self.merger.greek_letters
        }

        self.plot_and_save(
            scores,
            best_bases,
            best_weights,
            minimise=True,
        )
