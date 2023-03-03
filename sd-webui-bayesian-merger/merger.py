from functools import partial

from bayes_opt import BayesianOptimization

from generator import Generator
from scorer import Scorer

# from ../sd-webui-block-merge/scripts/merge import merge


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
        weights, base_alpha = params

        self.merger.sd_merge(
            weights,
            base_alpha,
        )

        self.generator.switch_model(self.output_file)

        # generate images
        images = []
        for payload in payloads:
            images.extend(self.generator.batch_generate(payload))

        # score images
        scores = self.scorer.batch_score(images)

        return self.scorer.average_score(scores)

    def optimize(self, payloads, init_points, n_iter):
        partial_sd_target_function = partial(
            self.sd_target_function,
            self,
            payloads,
        )

        # TODO: init UNET blocks
        pbounds = {}

        # TODO: what if we want to optimise only certain blocks?

        self.optimizer = BayesianOptimization(
            f=partial_sd_target_function,
            pbounds=pbounds,
            random_state=1,
        )

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

    def postprocess(optmizer: BayesianOptimization) -> None:
        # TODO: analyse the results
        return
