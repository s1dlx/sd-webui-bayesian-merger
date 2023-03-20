from pathlib import Path
from typing import Dict, Tuple, List

from bayes_opt import BayesianOptimization, Events

from sd_webui_bayesian_merger.artist import draw_unet
from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser, convergence_plot


class BayesOptimiser(Optimiser):
    def optimise(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        pbounds = {f"block_{i}": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS)}
        pbounds["base_alpha"] = (0.0, 1.0)

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=1,
        )

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)

        self.optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iters,
        )

        # clean up and remove the last merge
        self._cleanup()

    def postprocess(self) -> None:
        for i, res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \n\t{res}")

        print(self.optimizer.max)

        img_path = Path("logs", f"{self.merger.output_file.stem}-{self.method}.png")
        scores = parse_scores(self.optimizer.res)
        convergence_plot(scores, figname=img_path)

        unet_path = Path("logs", f"{self.merger.output_file.stem}-unet-{self.method}.png")
        best_base_alpha, best_weights = parse_params(self.optimizer.max["params"])
        draw_unet(
            best_base_alpha,
            best_weights,
            model_a=Path(self.model_a).stem,
            model_b=Path(self.model_b).stem,
            figname=unet_path,
        )

        if self.save_best:
            print(f'Saving best merge: {self.merger.best_output_file}')
            self.merger.merge(best_weights, best_base_alpha, best=True)


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]


def parse_params(params: Dict) -> Tuple[float, List[float]]:
    weights = [params[f"block_{i}"] for i in range(NUM_TOTAL_BLOCKS)]
    base_alpha = params["base_alpha"]
    return base_alpha, weights
