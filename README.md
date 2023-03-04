# sd-webui-bayesian-merger

## What is this?

An opinionated take on stable-diffusion models-merging automatic-optimisation.

... refs on bayesian optimisation ...

Why opinionated? Because we use webui API and lots of config files to run the show. No GUI. 
Embrace your inner touch-typist and leave the browser for the CLI.

## Requirements

- A working installation of [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Installed [sdweb-merge-block-weighted-gui](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui) extension

## How to use

- Start webui in `--api` mode
- ...

## Stuff to add

- [ ] wildcards
- [ ] other prompt manipulation extensions
- [ ] orthogonal sampler
- [ ] UNET visualisation

## Stuff to test

- [ ] Are existing scoring networks tuned towards photorealism?
- [ ] Can we optimize for prompting style? Are wildcards enough?
- [ ] Can we train a scoring network on personal preferences? How many images do we need? Tooling?
- [ ] Shall we use [sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger) instead?

## Inspired by

- [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)