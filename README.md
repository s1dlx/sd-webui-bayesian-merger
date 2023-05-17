# sd-webui-bayesian-merger

## NEWS

- 2023/05/17 add `scorer_device`; remove `aes` and `cafe_*` scorers
- 2023/05/16 `latin-hypercube-sampling` for `bayes` optimiser
- 2023/05/15 `adaptive-tpe` optimiser
- 2023/05/03 `tensor_sum` merging method
- 2023/04/25 `weighted_subtraction` merging method
- 2023/04/22 `manual` scoring method
- 2023/04/18 `group` parameters
- 2023/04/17 `freeze` parameters or set custom optimisation `ranges`

## What is this?

An opinionated take on stable-diffusion models-merging automatic-optimisation.

The main idea is to treat models-merging procedure as a black-box model with 26 parameters: one for each block plus `base_alpha`.
We can then try to apply black-box optimisation techniques, in particular we focus on [Bayesian optimisation](https://en.wikipedia.org/wiki/Bayesian_optimization) with a [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) emulator.
Read more [here](https://github.com/fmfn/BayesianOptimization), [here](http://gaussianprocess.org) and [here](https://optimization.cbe.cornell.edu/index.php?title=Bayesian_optimization).

The optimisation process is split in two phases:
1. __exploration__: here we sample (at random for now, with some heuristic in the future) the 26-parameter hyperspace, our block-weights. The number of samples is set by the
`--init_points` argument. We use each set of weights to merge the two models we use the merged model to generate `batch_size * number of payloads` images which are then scored.
2. __exploitation__: based on the exploratory phase, the optimiser makes an idea of where (i.e. which set of weights) the optimal merge is.
This information is used to sample more set of weights `--n_iters` number of times. This time we don't sample all of them in one go. Instead, we sample once, merge the models,
generate and score the images and update the optimiser knowledge about the merging space. This way the optimiser can adapt the strategy step-by-step.

At the end of the exploitation phase, the set of weights scoring the highest score are deemed to be the optimal ones.

## OK, How Do I Use It In Practice?

Head to the [wiki](https://github.com/s1dlx/sd-webui-bayesian-merger/wiki/Home) for all the instructions to get you started.

### FAQ

- Why opinionated? Because we use webui API and lots of config files to run the show. No GUI. 
Embrace your inner touch-typist and leave the browser for the CLI.
- Why rely on webui API? It's a very popular platform. Chances are that if you already have a working webui, you do not need to do much to run this library.
- How many iterations and payloads? What about the batch size? I'd suggest `--init_points 10 --n_iters 10 --batch_size 10` and at least 5 different payloads.
Depending on your GPU this may take 2-3hrs to run on basic config.

## With the help of

- [sdweb-merge-block-weighted-gui](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)
- [sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger)
- [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)
- [SD-Chad](https://github.com/grexzen/SD-Chad.git)
