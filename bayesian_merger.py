# from sd_webui_bayesian_merger.optimiser import BayesianOptimiser
from sd_webui_bayesian_merger.merger import SDModel


# TODO: clip args
# if __name__ == "__main__":
# url = ""
# batch_size = ""
# model_a = ""
# model_b = ""
# device = ""
# output_file = ""
# bo = BayesianOptimiser(
#     url, batch_size, model_a, model_b, device, output_file
# )

# # TODO: where are the prompts coming from? add a Prompter class
# payloads = []

# # TODO: compute total number of generations as batch_size * len(payloads)
# init_points = 30
# n_iters = 20
# bom.optimise(payloads, init_points, n_iters)

sd = SDModel(
    "/Users/ale/repos/stable-diffusion-webui/models/Stable-diffusion/openjourney-v2.ckpt",
    "cpu",
)
sd.load_model()
