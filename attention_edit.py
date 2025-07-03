import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import attention_utils

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def latent2image(vae, latents):
    latents = latents.to(vae.device)
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)

    return latents


@torch.no_grad()
def text2image_ldm_stable(model,\
                        prompts,\
                        controller,\
                        latent=None,\
                        num_inference_steps=50,\
                        guidance_scale=7.5,\
                        generator=None,\
                        low_resource=False,\
                        uncond_embeddings=None,\
                        text_embeddings=None,\
                        start_time=50,\
                        task_type=None):

    attention_utils.register_attention_control(model, controller)

    height = width = 512
    batch_size     = len(prompts)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)

    if task_type == 'editing':
        timesteps_lst = model.scheduler.timesteps
    elif task_type =='inversion':
        timesteps_lst = model.scheduler.timesteps[-start_time:]
    elif task_type == 'generation':
        timesteps_lst = model.scheduler.timesteps



    for i, t in enumerate(tqdm(timesteps_lst)):
        if type(uncond_embeddings) is list:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)

    return image, latent, latents