# References: https://github.com/google/prompt-to-prompt/tree/main

import os
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusion_inversion import NullInversion

import attention_edit
import attention_utils
from attention_utils import AttentionControl, EmptyControl, AttentionStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS  = 77
LOW_RESOURCE = False

np.random.seed(8888)
GENERATOR = torch.manual_seed(8888)

def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, text_embeddings=None, uncond_embeddings=None, verbose=True, file_name='tmp', task_type=None):

    if run_baseline:
        print('without prompt2prompt')
        images, latent, image_latents = run_and_display(prompts=prompts,\
                                         controller=EmptyControl(),\
                                         latent=latent,\
                                         run_baseline=False,\
                                         generator=generator,\
                                         file_name=file_name,\
                                         task_type=task_type)
        print('with prompt2prompt')
    
    images, x_t, image_latents = attention_edit.text2image_ldm_stable(ldm_stable,\
                                                    prompts,\
                                                    controller,\
                                                    latent=latent,\
                                                    num_inference_steps=NUM_DIFFUSION_STEPS,\
                                                    guidance_scale=GUIDANCE_SCALE,\
                                                    generator=generator,\
                                                    uncond_embeddings=uncond_embeddings,\
                                                    text_embeddings=text_embeddings,\
                                                    task_type=task_type)
    
    attention_utils.view_images(images, file_name=file_name)

    return images, x_t, image_latents


def get_text_embedding(ldm_stable, prompts):
    # Generating Text Prompt Embedding
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]

    return text_input, text_embeddings

def get_uncond_embedding(ldm_stable, batch_size, max_length):
    # Generating Null Prompt Embedding
    uncond_input    = ldm_stable.tokenizer([""] * batch_size,\
                                        padding="max_length",\
                                        max_length=max_length,\
                                        return_tensors="pt")    
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(ldm_stable.device))[0]
    return uncond_embeddings

# References: https://github.com/google/prompt-to-prompt/tree/main

import os
import math
import numpy as np
from glob import glob
from natsort import natsorted
from typing import List
from tqdm import tqdm
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusion_inversion import NullInversion

import attention_edit
import attention_utils
from attention_utils import AttentionControl, EmptyControl, AttentionStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS  = 77
LOW_RESOURCE = False

np.random.seed(8888)
GENERATOR = torch.manual_seed(8888)

# -----------------------------------------------------------------------------
# ─── INTERPOLATION FUNCTIONS ───────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def interp_linear(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Simple convex combination:  t·z0 + (1−t)·z1
    """
    return t * z0 + (1 - t) * z1


def interp_fix(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Fixed‐Norm normalization:
      1. bar = t·z0 + (1−t)·z1
      2. α = sqrt(L) / ||bar||
      3. return α·bar
    Ensures the final latent’s norm ≈ sqrt(L), where L = #elements per latent.
    """
    bar = t * z0 + (1 - t) * z1
    L = float(bar.numel())  # total number of elements in the latent
    norm_bar = bar.norm().clamp_min(1e-8)
    return bar * (math.sqrt(L) / norm_bar)


def interp_nin(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Norm‐Interpolated‐Norm (NIN):
      1. compute weighted_norm = t·||z0|| + (1−t)·||z1||
      2. bar = t·z0 + (1−t)·z1
      3. return (weighted_norm / ||bar||) · bar
    Ensures interpolation reproduces z_k if t=1 or t=0, and keeps the “average” norm along the path.
    """
    w0, w1 = t, 1 - t
    n0 = z0.norm()
    n1 = z1.norm()
    weighted_norm = w0 * n0 + w1 * n1

    bar = w0 * z0 + w1 * z1
    norm_bar = bar.norm().clamp_min(1e-8)

    return bar * (weighted_norm / norm_bar)


def interp_slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) in R^L:
      1. Compute ω = arccos( <z0,z1> / (||z0||·||z1||) )
      2. return [sin((1−t)·ω)/sin(ω)]·z0  +  [sin(t·ω)/sin(ω)]·z1
    If ω is very small, fall back to linear interpolation to avoid division by zero.
    """
    # flatten to 1D
    z0_flat = z0.view(-1)
    z1_flat = z1.view(-1)

    # cosine of angle between z0 and z1
    denom = (z0_flat.norm() * z1_flat.norm()).clamp_min(1e-8)
    cos_ω = torch.dot(z0_flat, z1_flat) / denom
    cos_ω = cos_ω.clamp(-1 + 1e-7, 1 - 1e-7)  # make sure we’re in [-1,1]
    ω = torch.acos(cos_ω)
    sin_ω = torch.sin(ω).clamp_min(1e-8)

    if sin_ω < 1e-7:
        # vectors are almost colinear → fallback to linear
        print("Falling back to linear interp")
        return interp_linear(z0, z1, t)

    factor0 = torch.sin((1 - t) * ω) / sin_ω
    factor1 = torch.sin(t * ω) / sin_ω
    z_flat = factor0 * z0_flat + factor1 * z1_flat
    return z_flat.view_as(z0)


def interp_mean_adjusted(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Mean‐Adjusted interpolation (channel‐wise mean removal + NIN on noise part):
      1. d0 = channel‐wise mean of z0, broadcast to [C,H,W]
         d1 = channel‐wise mean of z1, broadcast to [C,H,W]
         e0 = z0 - d0,  e1 = z1 - d1
      2. weighted_norm = (1−t)·||e0|| + t·||e1||
         bar_e = (1−t)·e0 + t·e1
         e' = (weighted_norm / ||bar_e||) · bar_e
      3. d' = (1−t)·d0 + t·d1
      4. return z' = d' + e'
    """
    # Corrected weight assignment
    w0, w1 = 1 - t, t

    # 1) channel-wise means
    mu0 = z0.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
    mu1 = z1.mean(dim=(1, 2), keepdim=True)
    d0 = mu0.expand_as(z0)
    d1 = mu1.expand_as(z1)

    e0 = z0 - d0
    e1 = z1 - d1

    # 2) NIN normalization on noise part
    n0 = e0.norm(p=2)
    n1 = e1.norm(p=2)
    weighted_norm = w0 * n0 + w1 * n1

    bar_e = w0 * e0 + w1 * e1
    norm_bar_e = bar_e.norm(p=2).clamp_min(1e-8)
    e_prime = bar_e * (weighted_norm / norm_bar_e)

    # 3) linear interpolation on deterministic component
    d_prime = w0 * d0 + w1 * d1

    # 4) combine
    return d_prime + e_prime


import os
import math
import numpy as np
from glob import glob
from natsort import natsorted
from typing import List
from tqdm import tqdm
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusion_inversion import NullInversion

import attention_edit
import attention_utils
from attention_utils import AttentionControl, EmptyControl, AttentionStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS  = 77
LOW_RESOURCE = False

np.random.seed(8888)
GENERATOR = torch.manual_seed(8888)

# -------------------------------------------------------------------------
# ─── DIFFUSION‐LIKE (STOCHASTIC) PATH FUNCTION ───────────────────────────
# -------------------------------------------------------------------------
def diffuse_path(z_start: torch.Tensor,
                 z_end:   torch.Tensor,
                 num_steps: int = 30,
                 alpha: float = 0.1,
                 sigma: float = 0.05,
                 generator: Optional[torch.Generator] = None
                ) -> List[torch.Tensor]:
    """
    Produce a list of latents {z₀, z₁, …, zₜ} that stochastically 
    move from z_start to z_end over `num_steps` steps.

    At each step:
        zₜ₊₁ = zₜ + α·(z_end - zₜ) + σ·εₜ ,  εₜ ~ N(0, I)

    Arguments:
        z_start:   Tensor of shape [C, H, W], initial latent.
        z_end:     Tensor of shape [C, H, W], target latent.
        num_steps: Number of discrete steps in the path (including endpoints).
        alpha:     Drift factor towards z_end (0 < α ≤ 1).  
                   Larger α → more deterministic “pull” to z_end.
        sigma:     Std‐dev of Gaussian noise added at each step.  
                   Larger σ → more randomness in the walk.
        generator: Optional torch.Generator to control reproducibility.

    Returns:
        A Python list of length `num_steps` of tensors [C, H, W], 
        starting with z_start and ending (approximately) near z_end.
    """
    # Ensure z_start, z_end are float32 tensors on the same device
    z_t = z_start.clone().to(DEVICE)
    z_end = z_end.to(DEVICE)

    latents = [z_t.clone()]  # include z₀

    for step in range(1, num_steps):
        # draw standard normal noise tensor εₜ
        eps = torch.randn_like(z_t)

        # drift term toward z_end
        drift = alpha * (z_end - z_t)

        # diffusion step
        z_t = z_t + drift + sigma * eps

        latents.append(z_t.clone())

    return latents

# -------------------------------------------------------------------------
# ─── MAIN SCRIPT (modified from your original) ───────────────────────────
# -------------------------------------------------------------------------
if __name__ == '__main__':
    task_type = 'inversion'
    # task_type = 'editing'

    # 1) Load Stable Diffusion + DDIMScheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )
    ldm_stable = (
        StableDiffusionPipeline
        .from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_safetensors=True)
        .to(DEVICE)
    )
    tokenizer = ldm_stable.tokenizer

    # 2) Prepare output directories
    folder_loc = 'newBenign/exp_diffusion/'  # set a new folder location
    os.makedirs(os.path.join(folder_loc, 'diffused_images'), exist_ok=True)
    os.makedirs(os.path.join(folder_loc, 'inverted_latent'), exist_ok=True)

    # 3) Load ALL saved latents (same as before)
    prompt_lst  = []
    latents_lst = []
    for latent_path in natsorted(glob('./results/inversion/inverted_latent/*_final_latent_*.pt')):
        basename = os.path.basename(latent_path)
        tokens = os.path.splitext(basename)[0].split('_')
        prompt_str = ' '.join(tokens[3:])
        prompt_lst.append(prompt_str)

        loaded = torch.load(latent_path)
        if loaded.ndim == 4 and loaded.shape[0] == 1:
            loaded = loaded.squeeze(0)
        latents_lst.append(loaded)

    latents_lst = torch.stack(latents_lst, dim=0)  # shape [N, C, H, W]

    # 4) Pick TWO latents for “diffusion”
    idx_a = 0    # index of first latent
    idx_b = -1   # index of second latent
    zA = latents_lst[idx_a]  # [C, H, W]
    zB = latents_lst[idx_b]  # [C, H, W]
    chosen_prompt = prompt_lst[idx_a]
    file_id = chosen_prompt.replace(' ', '_')

    # 5) Generate a “diffusion‐like” path instead of interpolation
    num_steps = 30
    alpha = 0.5   # strength of drift toward zB
    sigma = 0.05  # noise magnitude
    # use the same generator for reproducibility
    latents_path = diffuse_path(
        z_start=zA,
        z_end=zB,
        num_steps=num_steps,
        alpha=alpha,
        sigma=sigma,
        generator=GENERATOR
    )

    # 6) For each latent in the path, decode back into an image and save
    for step_idx, z_t in enumerate(tqdm(latents_path, desc="Diffusion steps")):
        with torch.no_grad():
            # decode z_t → PIL image (attention_edit.latent2image returns a list)
            decoded = attention_edit.latent2image(ldm_stable.vae, z_t.unsqueeze(0).to(DEVICE))
            img = decoded[0]  # take first (only) image

        out_path = os.path.join(
            folder_loc,
            f'diffused_images/{step_idx:02d}_diffusion_{file_id}'
        )
        attention_utils.view_images(img, file_name=out_path)

    # 7) (Optional) Pick one intermediate step (e.g. t = 15) and show cross‐attention maps
    t_idx = num_steps // 2
    z_mid = latents_path[t_idx]
    controller = AttentionStore()
    # get text and null embeddings for chosen prompt
    _, text_embeddings = get_text_embedding(ldm_stable, [chosen_prompt])
    uncond_embeddings = get_uncond_embedding(ldm_stable, batch_size=1, max_length=MAX_NUM_WORDS)

    images_attn, x_t, image_latents = attention_edit.text2image_ldm_stable(
        ldm_stable,
        [chosen_prompt],
        controller,
        latent=z_mid.unsqueeze(0).to(DEVICE),
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=GENERATOR,
        uncond_embeddings=uncond_embeddings,
        text_embeddings=text_embeddings,
        task_type=task_type
    )
    attn_out = os.path.join(folder_loc, f'attn_map_step{t_idx:02d}_{file_id}')
    attention_utils.show_cross_attention(
        attention_store=controller,
        res=16,
        from_where=["up", "down"],
        tokenizer=tokenizer,
        prompts=[chosen_prompt],
        prompt_idx=0,
        file_name=attn_out
    )
