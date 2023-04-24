from typing import Dict, List, Tuple

from bayes_opt import BayesianOptimization, Events
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer

from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS
from sd_webui_bayesian_merger.optimiser import Optimiser


class BayesOptimiser(Optimiser):
    bounds_transformer = SequentialDomainReductionTransformer()

    def optimise(self) -> None:
        pbounds = self.init_params()

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
        best_weights, best_bases = self.assemble_params(
            self.optimizer.max["params"],
        )

        self.plot_and_save(
            scores,
            best_bases,
            best_weights,
            minimise=False,
        )

def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]
