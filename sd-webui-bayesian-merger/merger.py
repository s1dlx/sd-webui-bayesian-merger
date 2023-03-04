from functools import partial

from bayes_opt import BayesianOptimization

from generator import Generator
from scorer import Scorer

# TODO
# from ../sd-webui-block-merge/scripts/merge import merge


def merge(
    weights,
    model_a,
    model_b,
    device,
    base_alpha,
    output_file,
    allow_overwrite,
    verbose,
    save_as_saferensors,
    save_as_half,
    skip_position_ids,
):
    pass


class Merger:
    def __init__(
        self,
        model_a: str,
        model_b: str,
        device: str,
        output_file: str,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.device = device
        self.output_file = output_file

        # TODO: add as parameter?
        self.skip_position_ids = 0

    def sd_merge(
        self,
        weights: list,
        base_alpha=0.5,
    ):
        # TODO
        merge(
            weights,
            self.model_a,
            self.model_b,
            self.device,
            base_alpha,
            self.output_file,
            allow_overwrite=False,
            verbose=False,
            save_as_safetensors=False,
            save_as_half=False,
            skip_position_ids=self.skip_position_ids,
        )


class BayesianOptimisationMerger:
    def __init__(
        self,
        url,
        batch_size,
        model_a,
        model_b,
        device,
        output_file,
    ):
        self.generator = Generator(url, batch_size)
        self.merger = Merger(model_a, model_b, device, output_file)
        self.scorer = Scorer()
        self.output_file = output_file

    def sd_target_function(self, payloads: [dict], **params):
        # TODO: in args?
        # skip_position_ids = 0

        # TODO: weights and base_alpha from params
        # is the list of floats OK?
        weights = [params[f"block_{i}"] for i in range(25)]
        base_alpha = params['base_alpha']

        self.merger.sd_merge(
            weights,
            base_alpha,
        )

        # TODO: is this forcing the model load despite the same name?
        self.generator.switch_model(self.output_file)

        # generate images
        images = []
        for payload in payloads:
            images.extend(self.generator.batch_generate(payload))

        # score images
        scores = self.scorer.batch_score(images)

        # spit out a single value for optimisation
        return self.scorer.average_score(scores)

    def optimize(self, payloads, init_points, n_iter):
        partial_sd_target_function = partial(
            self.sd_target_function,
            self,
            payloads,
        )

        pbounds = {f"block_{i}": (0.0, 1.0) for i in range(25)}
        pbounds['base_alpha': (0.0, 1.0)]

        # TODO: what if we want to optimise only certain blocks?

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=partial_sd_target_function,
            pbounds=pbounds,
            random_state=1,
        )

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

    def postprocess(optmizer: BayesianOptimization) -> None:
        # TODO: analyse the results
        return
