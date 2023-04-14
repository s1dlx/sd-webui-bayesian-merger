from typing import Dict, Tuple, List

from bayes_opt import BayesianOptimization, Events

from bayes_opt.domain_reduction import SequentialDomainReductionTransformer

from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser


class BayesOptimiser(Optimiser):
    bounds_transformer = SequentialDomainReductionTransformer()

    def optimise(self) -> None:
        # TODO: what if we want to optimise only certain blocks?
        pbounds = {f"block_{i}_alpha": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS)}
        pbounds["base_alpha"] = (0.0, 1.0)
        for gl in self.merger.greek_letters:
            pbounds |= {
                f"block_{i}_{gl}": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS)
            }
            pbounds[f"base_{gl}"] = (0.0, 1.0)

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=1,
            bounds_transformer=self.bounds_transformer
            if self.cfg.bounds_transformer
            else None,
        )

        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)

        self.optimizer.maximize(
            init_points=self.cfg.init_points,
            n_iter=self.cfg.n_iters,
        )

        # clean up and remove the last merge
        try:
            self.cleanup()
        except FileNotFoundError:
            return

    def postprocess(self) -> None:
        print("\nRecap!")
        for i, res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \n\t{res}")

        scores = parse_scores(self.optimizer.res)
        best_bases, best_weights = self.parse_params(
            self.optimizer.max["params"],
        )

        self.plot_and_save(
            scores,
            best_bases,
            best_weights,
            minimise=False,
        )

    def parse_params(self, params: Dict) -> Tuple[Dict, Dict]:
        bases = {"alpha": params["base_alpha"]}
        weights = {
            "alpha": [params[f"block_{i}_alpha"] for i in range(NUM_TOTAL_BLOCKS)]
        }
        for m, gl in zip(self.merger.model_keys, self.merger.greek_letters):
            bases[gl] = params[f"base_{gl}"]
            weights[gl] = [params[f"block_{i}_{gl}"] for i in range(NUM_TOTAL_BLOCKS)]

        return bases, weights


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]
