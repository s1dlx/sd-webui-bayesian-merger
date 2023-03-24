from typing import Dict, Tuple, List

from bayes_opt import BayesianOptimization, Events

from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser


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
            init_points=self.cfg.init_points,
            n_iter=self.cfg.n_iters,
        )

        # clean up and remove the last merge
        self.cleanup()

    def postprocess(self) -> None:
        for i, res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \n\t{res}")

        scores = parse_scores(self.optimizer.res)
        best_base_alpha, best_weights = parse_params(self.optimizer.max["params"])

        self.plot_and_save(
            scores,
            best_base_alpha,
            best_weights,
            minimise=False,
        )


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]


def parse_params(params: Dict) -> Tuple[float, List[float]]:
    weights = [params[f"block_{i}"] for i in range(NUM_TOTAL_BLOCKS)]
    base_alpha = params["base_alpha"]
    return base_alpha, weights
