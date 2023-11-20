# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


# !important
def create_diffusion(
    timestep_respacing="",
    noise_schedule="linear", # 'linear' for training
    use_kl=False,
    rescale_learned_sigmas=False,
    prediction_type='v_prediction',
    variance_type='fixed_small',
    beta_start=0.0001,
    beta_end=0.02,
    # beta_start=0.00085,
    # beta_end=0.012,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps, beta_start=beta_start, beta_end=beta_end)
    if prediction_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON # EPSILON type for stable-diffusion-2-1 512
    elif prediction_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif prediction_type == 'v_prediction':
        model_mean_type = gd.ModelMeanType.PREVIOUS_V # gd.ModelMeanType.PREVIOUS_V for stable-diffusion-2-1 768/x4-upscaler
        
    if variance_type == 'fixed_small':
        model_var_type = gd.ModelVarType.FIXED_SMALL
    elif variance_type == 'fixed_large':
        model_var_type = gd.ModelVarType.FIXED_LARGE
    elif variance_type == 'learned_range':
        model_var_type = gd.ModelVarType.LEARNED_RANGE    
        
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(model_mean_type),
        model_var_type=(model_var_type),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
