from functools import partial

from bayes_opt import BayesianOptimization

from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.scorer import Scorer, Merger


class BayesianOptimiser:
    def __init__(
        self,
        url,
        batch_size,
        model_a,
        model_b,
        device,
        output_file,
        payloads_dir,
        wildcards_dir,
        scorer_model_path,
    ):
        self.generator = Generator(url, batch_size)
        self.merger = Merger(model_a, model_b, device, output_file)
        self.scorer = Scorer(scorer_model_path, device)
        self.prompter = Prompter(payloads_dir, wildcards_dir)
        self.output_file = output_file

    def sd_target_function(self, **params):
        # TODO: in args?
        # skip_position_ids = 0

        # TODO: weights and base_alpha from params
        # is the list of floats OK?
        weights = [params[f"block_{i}"] for i in range(25)]
        base_alpha = params["base_alpha"]

        self.merger.merge(
            weights,
            base_alpha,
        )

        # TODO: is this forcing the model load despite the same name?
        self.generator.switch_model(self.output_file)

        # generate images
        images = []
        for payload in self.prompter.render_payloads():
            images.extend(self.generator.batch_generate(payload))

        # score images
        scores = self.scorer.batch_score(images)

        # spit out a single value for optimisation
        return self.scorer.average_score(scores)

    def optimize(self, init_points, n_iter)->None:
        partial_sd_target_function = partial(self.sd_target_function, self)

        pbounds = {f"block_{i}": (0.0, 1.0) for i in range(25)}
        pbounds["base_alpha":(0.0, 1.0)]

        # TODO: what if we want to optimise only certain blocks?

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=partial_sd_target_function,
            pbounds=pbounds,
            random_state=1,
        )

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

    def postprocess(self, optmizer: BayesianOptimization) -> None:
        for i, res in enumerate(self.optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))
