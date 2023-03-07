# sd-webui-bayesian-merger

## What is this?

An opinionated take on stable-diffusion models-merging automatic-optimisation.

Why not auto-merge extension? Brute force == long time to wait. With this method you can get away with <X> number of runs!

Why opinionated? Because we use webui API and lots of config files to run the show. No GUI. 
Embrace your inner touch-typist and leave the browser for the CLI.

## Requirements

- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## How to use

- I believe you already have a stable-diffusion venv, activate it
- Start webui in `--api` mode
- `cd` to `sd-webui-bayesian-merger` folder
- `pip install -r requirements.txt`
- read `python3 bayesian_merger.py --help` and prepare all the arguments accordingly
- `python3 bayesian_merger.py --model_a=... `

## To be done

- [x] prompter class
- [x] generator class
- [x] merger class
- [x] optimiser class
- [x] scorer class
- [x] fill in `payload.tmpl.yaml`
- [x] results postprocessing
- [ ] actually test this thing

## Nice to have

- [x] wildcards
- [ ] comment out individual lines in wildcard files
- [ ] additional prompt manipulation extensions
- [ ] orthogonal sampler
- [ ] UNET visualisation
- [x] fill in `requirements.txt`
- [x] click cli
- [x] simple readme walkthrough
- [ ] example results
- [ ] explain what we're doing
- [ ] logging (really?!?)
- [x] native merge function
- [ ] simpler merge function (e.g. no regex)
- [ ] no need for webui OR proper webui extension, ehe
- [x] native scorer class (took from SD-Chad)
- [ ] consisten typing
- [ ] save optimisation results

## Experiments

- [ ] Are existing scoring networks tuned towards photorealism?
- [ ] Can we optimize for prompting style? Are wildcards enough?
- [ ] Can we train a scoring network on personal preferences? How many images do we need? Tooling?
- [ ] Shall we use [sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger) instead?
- [ ] Negative payloads: we want to minimise the score for these (e.g. target `cartoon`)

## With the help of

- [sdweb-merge-block-weighted-gui](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)
- [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)
- [SD-Chad](https://github.com/grexzen/SD-Chad.git)
