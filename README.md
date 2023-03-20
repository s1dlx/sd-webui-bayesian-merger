# sd-webui-bayesian-merger

## What is this?

An opinionated take on stable-diffusion models-merging automatic-optimisation.

The main idea is to treat models-merging procedure as a black-box model with 26 parameters: one for each block plus `base_alpha` (note that for the moment `clip_skip` is set to `0`).
We can then try to apply black-box optimisation techniques, in particular we focus on [Bayesian optimisation](https://en.wikipedia.org/wiki/Bayesian_optimization) with a [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) emulator.
Read more [here](https://github.com/fmfn/BayesianOptimization), [here](http://gaussianprocess.org) and [here](https://optimization.cbe.cornell.edu/index.php?title=Bayesian_optimization).

The optimisation process is split in two phases:
1. __exploration__: here we sample (at random for now, with some heuristic in the future) the 26-parameter hyperspace, our block-weights. The number of samples is set by the
`--init_points` argument. We use each set of weights to merge the two models we use the merged model to generate `batch_size * number of payloads` images which are then scored.
2. __exploitation__: based on the exploratory phase, the optimiser makes an idea of where (i.e. which set of weights) the optimal merge is.
This information is used to sample more set of weights `--n_iters` number of times. This time we don't sample all of them in one go. Instead, we sample once, merge the models,
generate and score the images and update the optimiser knowledge about the merging space. This way the optimiser can adapt the strategy step-by-step.

At the end of the exploitation phase, the set of weights scoring the highest score are deemed to be the optimal ones.

## Juicy features

- wildcards support
- TPE or Bayesian Optimisers. [cf. Bergstra et al., Algorithms for Hyper-Parameter Optimization 2011](http://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) for a comparison and explanation
- UNET visualiser
- convergence plot

## OK, How Do I Use It In Practice?

### Requirements

- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). You need to have it working locally and know how to change option flags.
- Install this _extension_ from url: `https://github.com/s1dlx/sd-webui-bayesian-merger.git`. This will place this codebase into your `extensions` folder.
- I believe you already have a stable-diffusion venv, activate it
- `cd` to `stable-diffusion-webui/extensions/sd-webui-bayesian-merger` folder
- `pip install -r requirements.txt`

### Prepare payloads

A `payload` is where you type in all the generation parameters (just like you click away in the webui). I've added a `payload.tmpl.yaml` you can use as reference:

```yaml
prompt: "your prompt, even with __wildcard__"
neg_prompt: "your negative prompt"
seed: -1
cfg: 7
width: 512
height: 512
sampler_name: "Euler"
```

As you can see, this is a subset of the configs you have in webui, but it should be enough to start with.

- copy the `payload.tmpl.yaml` file and name it `mypayloadname.yaml`
- fill in the various fields. Prompts support [wildcards](https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards) but not other extensions (e.g. [sd-dynamic-prompt](https://github.com/adieyal/sd-dynamic-prompts)) yet.
- make another copy of the payload template and keep going
- try to have different resolutions, cfg values, samplers, etc...
- however, try to be consistent with the style you want to achieve from the optimisation. For example, if you are merging two photorealistic models, it makes sense to avoid prompting for `illustration`. This is with the aim of not confusing the optimisation process.


### Run!

- Start webui in `--api --nowebui`[mode](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)
- Running `python bayesian_merger.py --help` will print

```
Usage: bayesian_merger.py [OPTIONS]

Options:
  --url TEXT                      where webui api is running, by default
                                  http://127.0.0.1:7860
  --batch_size INTEGER            number of images to generate for each
                                  payload
  --model_a PATH                  absolute path to first model  [required]
  --model_b PATH                  absolute path to second model  [required]
  --skip_position_ids INTEGER     clip skip, default 0
  --device TEXT                   where to merge models and score images,
                                  default and recommended "cpu"
  --payloads_dir PATH             absolute path to payloads directory
  --wildcards_dir PATH            absolute path to wildcards directory
  --scorer_model_dir PATH         absolute path to scorer models directory
  --init_points INTEGER           exploratory/warmup phase sample size
  --n_iters INTEGER               exploitation/optimisation phase sample size
  --draw_unet_weights TEXT        list of weights for drawing mode
  --draw_unet_base_alpha FLOAT    base alpha value for drawing mode
  --best_format [safetensors|ckpt]
                                  best model saving format, either safetensors
                                  (default) or ckpt
  --best_precision [16|32]        best model saving precision, either 16
                                  (default) or 32 bit
  --save_best                     save best model across the whole run
  --optimiser [bayes|tpe]         optimiser, bayes (default) or tpe
  --help                          Show this message and exit.
```

- Prepare the arguments accordingly and finally run `python3 bayesian_merger.py --model_a=... `
- Come back later to check results

### Results

In the `logs` function you'll find: 
- `bbwm-model_a-model_b.json`: this contains the scores and the weights for all the iterations. The final set of weights is the best one.
- `bbwm-model_a-model_b.png`: a plot showing the evolution of the score across the iterations.

- `bbwm-model_a-model_b-unet.png`: an images showing the best weights on the UNET architecture
<img width="641" alt="Screenshot 2023-03-13 at 11 35 32" src="https://user-images.githubusercontent.com/125022075/224714573-7d9ab61d-b534-4723-b029-3b12568b0ac7.png">

### Extra

- UNET drawing only: this command will save `./unet.png`, use it for quick visualise the net/debugging. Note that the weights should be passed as a string, i.e., in between quotes `"..."`
```
python3 bayesian_merger.py --model_a=name_A --model_b=name_B --draw_unet_base_alpha=0.5 --draw_unet_weights="1.0, 0.9166666667, 0.8333333333, 0.75, 0.6666666667,0.5833333333, 0.5, 0.4166666667, 0.3333333333, 0.25, 0.1666666667,0.0833333333,0.0,0.0833333333,0.1666666667,0.25,0.3333333333,0.4166666667,0.5,0.5833333333,0.6666666667,0.75,0.8333333333,0.9166666667,1.0"
```

### FAQ

- Why not [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW) extension? That amazing extension is based on brute-forcing the merge. Unfortunately, Brute force == long time to wait,
especially when generating lots of images. Hopefully, with this other method you can get away with a small number of runs!
- Why opinionated? Because we use webui API and lots of config files to run the show. No GUI. 
Embrace your inner touch-typist and leave the browser for the CLI.
- Why rely on webui? It's a very popular platform. Chances are that if you already have a working webui, you do not need to do much to run this library.
- How many iterations and payloads? What about the batch size? I'd suggest `--init_points 10 --n_iters 10 --batch_size 10` and at least 5 different payloads.
Depending on your GPU this may take 2-3hrs to run on basic config.

## With the help of

- [sdweb-merge-block-weighted-gui](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)
- [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)
- [SD-Chad](https://github.com/grexzen/SD-Chad.git)
