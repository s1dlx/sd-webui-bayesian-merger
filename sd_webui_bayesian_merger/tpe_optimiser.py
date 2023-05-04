from functools import partial

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from sd_webui_bayesian_merger.optimiser import Optimiser


class TPEOptimiser(Optimiser):
    def _target_function(self, params):
        res = self.sd_target_function(**params)
        return {
            "loss": -res,
            "status": STATUS_OK,
            "params": params,
        }

    def optimise(self) -> None:
        bounds = self.init_params()
        space = {p: hp.uniform(p, *b) for p, b in bounds.items()}

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

        best_weights, best_bases = self.bounds_initialiser.assemble_params(
            best["result"]["params"],
            self.merger.greek_letters,
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else None,
        )

        self.plot_and_save(
            scores,
            best_bases,
            best_weights,
            minimise=True,
        )
