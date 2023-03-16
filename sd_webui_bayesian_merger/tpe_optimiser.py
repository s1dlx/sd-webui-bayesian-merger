from pathlib import Path

from sd_webui_bayesian_merger.artist import draw_unet
from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser, convergence_plot

from hyperopt import Trials, hp, fmin, tpe, STATUS_OK


class TPEOptimiser(Optimiser):

    def _target_function(self, **params):
        res = self.sd_target_function(**params)
        return {
            'loss': -res,
            'status': STATUS_OK,
            'params': params
        }

    def optimise(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        space = {f"block_{i}": hp.uniform(f'block_{i}', 0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS)}
        space["base_alpha"] = hp.uniform('base_alpha', 0.0, 1.0)

        self.trials = Trials()
        fmin(
            self._target_function,
            space=space,
            algo=tpe.suggest,
            trials=self.trials,
            max_evals=self.n_iters,
        )

        # clean up and remove the last merge
        self.merger.remove_previous_ckpt(self.iteration + 1)

    def postprocess(self) -> None:
        scores = []
        for i, res in enumerate(self.trials.trials):
            print(f"Iteration {i}: \n\t{-res['loss']}")
            scores.append(-res['loss'])
        best = self.trials.best_trial
        print("Best:", best)

        img_path = Path("logs", f"{self.merger.output_file.stem}.png")

        convergence_plot(scores, figname=img_path)

        unet_path = Path("logs", f"{self.merger.output_file.stem}-unet.png")
        draw_unet(
            best['params']['base_alpha'],
            best['params']['weights'],
            model_a=Path(self.model_a).stem,
            model_b=Path(self.model_b).stem,
            figname=unet_path,
        )
