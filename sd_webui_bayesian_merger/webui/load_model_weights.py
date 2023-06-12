# This function entirely copy-pasted from https://github.com/hako-mikan/sd-webui-supermerger/blob/237df9aa8602aa7ff97fafea714294941b10b8b0/scripts/mergers/model_util.py#L776
# Which was itself almost entirely copy-pasted from `shared.sd_models.load_model_weights`
# I don't know why we need to do this T_T
def load_model_weights(theta_0, model_a):
    from modules import devices, sd_hijack, shared, sd_vae
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)

    model = shared.sd_model
    model.load_state_dict(theta_0, strict=False)
    del theta_0
    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)

    if not shared.cmd_opts.no_half:
        vae = model.first_stage_model

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None

        model.half()
        model.first_stage_model = vae

    devices.dtype = torch.float32 if shared.cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if shared.cmd_opts.no_half or shared.cmd_opts.no_half_vae else torch.float16
    devices.dtype_unet = model.model.diffusion_model.dtype

    if hasattr(shared.cmd_opts, "upcast_sampling"):
        devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
    else:
        devices.unet_needs_upcast = devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)
    sd_hijack.model_hijack.hijack(model)

    model.logvar = shared.sd_model.logvar.to(devices.device)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        setup_for_low_vram(model, shared.cmd_opts.medvram)
    else:
        model.to(shared.device)

    model.eval()

    shared.sd_model = model
    try:
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except:
        pass

    # shared.sd_model.sd_checkpoint_info.model_name = model_name

    def _setvae():
        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()
        vae_file, vae_source = sd_vae.resolve_vae(model_a)
        sd_vae.load_vae(shared.sd_model, vae_file, vae_source)

    try:
        _setvae()
    except:
        print("ERROR:setting VAE skipped")


import torch
from modules import devices

module_in_gpu = None
cpu = torch.device("cpu")


# This function entirely copy-pasted from https://github.com/hako-mikan/sd-webui-supermerger/blob/237df9aa8602aa7ff97fafea714294941b10b8b0/scripts/mergers/model_util.py#L851
# Which was itself almost entirely copy-pasted from `shared.lowvram.setup_for_low_vram`
# I am not sure why, but the only difference with the original function is that they omitted interactions with `sd_model.cond_stage_model.transformer` and `sd_model.embedder`
# related issue: https://github.com/hako-mikan/sd-webui-supermerger/issues/3
def setup_for_low_vram(sd_model, use_medvram):
    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(devices.device)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods

    first_stage_model = sd_model.first_stage_model
    first_stage_model_encode = sd_model.first_stage_model.encode
    first_stage_model_decode = sd_model.first_stage_model.decode

    def first_stage_model_encode_wrap(x):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_encode(x)

    def first_stage_model_decode_wrap(z):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_decode(z)

    # for SD1, cond_stage_model is CLIP and its NN is in the tranformer frield, but for SD2, it's open clip, and it's in model field
    if hasattr(sd_model.cond_stage_model, 'model'):
        sd_model.cond_stage_model.transformer = sd_model.cond_stage_model.model

    # remove four big modules, cond, first_stage, depth (if applicable), and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    stored = sd_model.first_stage_model, getattr(sd_model, 'depth_model', None), sd_model.model
    sd_model.first_stage_model, sd_model.depth_model, sd_model.model = None, None, None
    sd_model.to(devices.device)
    sd_model.first_stage_model, sd_model.depth_model, sd_model.model = stored

    # register hooks for those the first three models
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = first_stage_model_encode_wrap
    sd_model.first_stage_model.decode = first_stage_model_decode_wrap
    if sd_model.depth_model:
        sd_model.depth_model.register_forward_pre_hook(send_me_to_gpu)

    if hasattr(sd_model.cond_stage_model, 'model'):
        sd_model.cond_stage_model.model = sd_model.cond_stage_model.transformer
        del sd_model.cond_stage_model.transformer

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(devices.device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
