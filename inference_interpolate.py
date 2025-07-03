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



# -----------------------------------------------------------------------------
# ─── MAIN SCRIPT ─────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    task_type = 'inversion'
    # task_type = 'editing'

    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Load Stable Diffusion + DDIMScheduler
    # ──────────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Prepare output directories
    # ──────────────────────────────────────────────────────────────────────────────
    folder_loc = 'newBeniging/exp3/res'
    os.makedirs(folder_loc + 'inv_image/linear', exist_ok=True)
    os.makedirs(folder_loc + 'inv_image/fix', exist_ok=True)
    os.makedirs(folder_loc + 'inv_image/nin', exist_ok=True)
    os.makedirs(folder_loc + 'inv_image/slerp', exist_ok=True)
    os.makedirs(folder_loc + 'inv_image/mean', exist_ok=True)

    os.makedirs(folder_loc + 'col_image', exist_ok=True)
    os.makedirs(folder_loc + 'attn_image', exist_ok=True)
    os.makedirs(folder_loc + 'inverted_latent', exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Load ALL saved latents from "inverted_latent"
    #    Each file is named "..._final_latent_<prompt>.pt"
    #    We build two parallel lists:
    #      prompt_lst  = list of prompt strings
    #      latents_lst = tensor of shape [N, C, H, W]
    # ──────────────────────────────────────────────────────────────────────────────
    prompt_lst  = []
    latents_lst = []

    for latent_path in natsorted(glob('./results/inversion/inverted_latent/*_final_latent_*.pt')):
        # Extract prompt from filename:
        # e.g. "0_final_latent_an_image_of_a_fern.pt" → ["0","final","latent","an","image","of","a","fern"]
        basename = os.path.basename(latent_path)
        tokens = os.path.splitext(basename)[0].split('_')
        # The actual prompt words start at index 3 in the underscore-split name
        prompt_str = ' '.join(tokens[3:])
        prompt_lst.append(prompt_str)

        # Load latent (shape likely [C,H,W]) and make sure it’s [C,H,W], then append
        loaded = torch.load(latent_path)
        # If loaded has a batch‐dim already (e.g. [1,C,H,W]), squeeze it
        if loaded.ndim == 4 and loaded.shape[0] == 1:
            loaded = loaded.squeeze(0)
        latents_lst.append(loaded)

    # Stack into a single Tensor [N, C, H, W]
    latents_lst = torch.stack(latents_lst, dim=0)  # shape [N, C, H, W]

    # ──────────────────────────────────────────────────────────────────────────────
    # 4) Pick TWO latents for interpolation (e.g., index 0 and index 5).
    #    Feel free to change these indices.
    # ──────────────────────────────────────────────────────────────────────────────
    idx_a = 0    # index of first latent
    idx_b = 1    # index of second latent
    zA = latents_lst[idx_a]  # tensor of shape [C, H, W]
    zB = latents_lst[idx_b]  # tensor of shape [C, H, W]
    chosen_prompt = prompt_lst[idx_a]
    file_id = chosen_prompt.replace(' ', '_')

    # ──────────────────────────────────────────────────────────────────────────────
    # 5) Define a dict of all interpolation methods so we can loop over them
    # ──────────────────────────────────────────────────────────────────────────────
    methods = {
        'linear': interp_linear,
        'fix':    interp_fix,
        'nin':    interp_nin,
        'slerp':  interp_slerp,
        'mean':   interp_mean_adjusted
    }

    # ──────────────────────────────────────────────────────────────────────────────
    # 6) For each t ∈ [0, 1], compute EVERY interpolation variant, decode, and save
    # ──────────────────────────────────────────────────────────────────────────────
    num_steps = 30
    for step_idx, t in enumerate(tqdm(np.linspace(0, 1, num_steps), desc="Interpolation steps")):
        for method_name, func in methods.items():
            # 6a) Compute the interpolated latent via the chosen function
            z_interp = func(zA, zB, float(t))  # [C, H, W]

            # 6b) Decode the latent back into an image
            #     We assume attention_edit.latent2image returns a list of PIL images or tensors
            with torch.no_grad():
                decoded_images = attention_edit.latent2image(ldm_stable.vae, z_interp.unsqueeze(0).to(DEVICE))
                # decoded_images is a list; we take the first (and only) one
                img = decoded_images[0]

            # 6c) Save/view it using attention_utils.view_images
            out_path = (
                folder_loc +
                f'inv_image/{method_name}/{step_idx:02d}_{method_name}_{file_id}'
            )
            attention_utils.view_images(img, file_name=out_path)

    # ──────────────────────────────────────────────────────────────────────────────
    # 7) (Optional) Show them in a single “compare” grid for one t to inspect:
    #    For example, compare all methods at t=0.5 side-by-side.
    # ──────────────────────────────────────────────────────────────────────────────
    t = 0.5
    t_idx = int(t * (num_steps - 1))
    images_for_compare = []
    for method_name, func in methods.items():
        z_ = func(zA, zB, float(t))
        with torch.no_grad():
            imgs = attention_edit.latent2image(ldm_stable.vae, z_.unsqueeze(0).to(DEVICE))
            images_for_compare.append(imgs[0])
    # Save a single grid of all five methods at t=0.5
    compare_out = folder_loc + f'col_image/compare_t{t_idx:02d}_{file_id}'
    attention_utils.view_images(images_for_compare, file_name=compare_out)

    # ──────────────────────────────────────────────────────────────────────────────
    # 8) (Optional) Show cross‐attention maps for one of the methods & one t:
    # ──────────────────────────────────────────────────────────────────────────────
    # E.g., pick method="mean", t=0.5 → show cross‐attention for that latent.
    selected_method = "mean"
    z_adv = methods[selected_method](zA, zB, float(t))
    controller = AttentionStore()
    # We need text_embeddings and uncond_embeddings—use dummy prompt as before:
    _, text_embeddings = get_text_embedding(ldm_stable, [chosen_prompt])
    uncond_embeddings = get_uncond_embedding(ldm_stable, batch_size=1, max_length=MAX_NUM_WORDS)

    # Now re‐run with attention control
    images_attn, x_t, image_latents = attention_edit.text2image_ldm_stable(
        ldm_stable,
        [chosen_prompt],
        controller,
        latent=z_adv.unsqueeze(0).to(DEVICE),
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=GENERATOR,
        uncond_embeddings=uncond_embeddings,
        text_embeddings=text_embeddings,
        task_type=task_type
    )
    # Save the cross‐attention visualization at "attn_image/"
    attn_out = folder_loc + f'attn_image/{selected_method}_t{t_idx:02d}_{file_id}'
    attention_utils.show_cross_attention(
        attention_store=controller,
        res=16,
        from_where=["up", "down"],
        tokenizer=tokenizer,
        prompts=[chosen_prompt],
        prompt_idx=0,
        file_name=attn_out
    )
