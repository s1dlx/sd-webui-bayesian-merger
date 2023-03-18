from functools import partial
from pathlib import Path

from sd_webui_bayesian_merger.artist import draw_unet
from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser, convergence_plot

from hyperopt import Trials, hp, fmin, tpe, STATUS_OK


class TPEOptimiser(Optimiser):

    def _target_function(self, params):
        res = self.sd_target_function(**params)
        return {
            'loss': 1. / (res + 1e-10),
            'status': STATUS_OK,
            'params': params
        }

    def optimise(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        space = {f"block_{i}": hp.uniform(f'block_{i}', 0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS)}
        space["base_alpha"] = hp.uniform('base_alpha', 0.0, 1.0)

        # this will do 20 warmup runs before optimising
        self.trials = Trials()
        tpe._default_n_startup_jobs = self.init_points
        algo = partial(tpe.suggest, n_startup_jobs=self.init_points)
        fmin(
            self._target_function,
            space=space,
            algo=algo,
            trials=self.trials,
            max_evals=self.n_iters,
        )

        # clean up and remove the last merge
        self._cleanup()

    def _cleanup(self):
        self.merger.remove_previous_ckpt(self.iteration + 1)

    def postprocess(self) -> None:
        scores = []
        for i, res in enumerate(self.trials.losses()):
            print(f"Iteration {i}: \n\t{-res}")
            scores.append(-res)
        best = self.trials.best_trial
        print("Best:", best)
        img_path = Path("logs", f"{self.merger.output_file.stem}-{self.method}.png")

        convergence_plot(scores, figname=img_path)

        unet_path = Path("logs", f"{self.merger.output_file.stem}-unet-{self.method}.png")
        draw_unet(
            best['result']['params']['base_alpha'],
            [best['result']['params'][f'block_{i}'] for i in range(NUM_TOTAL_BLOCKS)],
            model_a=Path(self.model_a).stem,
            model_b=Path(self.model_b).stem,
            figname=unet_path,
        )
