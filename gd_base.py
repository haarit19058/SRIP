import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from glob import glob
from natsort import natsorted
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import matplotlib.pyplot as plt

# Assume these helpers are in the same directory, as in the reference code.
import attention_edit
import attention_utils

# -----------------------------------------------------------------------------
# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "CompVis/stable-diffusion-v1-4"

# Optimization parameters
NUM_OPTIMIZATION_STEPS = 300    # Number of steps for gradient descent
LEARNING_RATE = 0.01             # Learning rate for the optimizer
SIMILARITY_LAMBDA = 1         # Weight for the similarity loss term (pulls towards target)
NORM_LAMBDA = 0.0001           # Weight for the latent‐norm loss term
GUIDANCE_SCALE = 7.5             # For the diffusion loss calculation
NUM_DIFFUSION_STEPS = 100         # Number of timesteps for the scheduler

# Paths
scene_name = 'fern'
LATENTS_DIR = f'./results/{scene_name}/inversion/inverted_latent/'  # Directory with pre‐computed latents
OUTPUT_DIR = f'./gd_base_{scene_name}/'

# Seeds for reproducibility
np.random.seed(8888)
torch.manual_seed(8888)
torch.cuda.manual_seed_all(8888)

# -----------------------------------------------------------------------------
# ─── HELPER FUNCTIONS (from reference code) ──────────────────────────────────
# -----------------------------------------------------------------------------
def get_text_embedding(ldm_stable, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates text embeddings for a list of prompts."""
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(DEVICE))[0]
    return text_input, text_embeddings

def get_uncond_embedding(ldm_stable, batch_size: int) -> torch.Tensor:
    """Generates null-prompt embeddings."""
    max_length = ldm_stable.tokenizer.model_max_length
    uncond_input = ldm_stable.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    return uncond_embeddings

def decode_and_save_image(ldm_stable, latent: torch.Tensor, file_path: str):
    """Decodes a latent vector to a PIL image and saves it."""
    with torch.no_grad():
        # The latent2image helper from the reference code handles VAE scaling and conversion
        image_list = attention_edit.latent2image(ldm_stable.vae, latent)
        if len(image_list) > 0:
            plt.figure()
            plt.imshow(image_list[0])
            plt.axis('off')
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# -----------------------------------------------------------------------------
# ─── MAIN EXECUTION ──────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Load Stable Diffusion model and scheduler
    print("Loading Stable Diffusion model...")
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
        .from_pretrained(MODEL_ID, scheduler=scheduler, use_safetensors=True)
        .to(DEVICE)
    )

    # 2. Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Images will be saved in: {OUTPUT_DIR}")

    # 3. Load start and target latent vectors
    print(f"Loading latents from: {LATENTS_DIR}")
    prompt_lst = []
    latents_lst = []
    latent_files = natsorted(glob(os.path.join(LATENTS_DIR, '*_final_latent_*.pt')))

    if not latent_files:
        raise FileNotFoundError(f"No latent files found in '{LATENTS_DIR}'. Please generate them first.")

    for latent_path in latent_files:
        basename = os.path.basename(latent_path)
        tokens = os.path.splitext(basename)[0].split('_')
        prompt_str = ' '.join(tokens[3:])
        prompt_lst.append(prompt_str)

        loaded = torch.load(latent_path, map_location=DEVICE)
        if loaded.ndim == 4 and loaded.shape[0] == 1:
            loaded = loaded.squeeze(0)
        latents_lst.append(loaded)

    start_idx = 0
    target_idx = 3

    # Ensure latents have a batch dimension of 1: [1, C, H, W]
    z_start = latents_lst[start_idx].unsqueeze(0).to(DEVICE)   # Shape: [1, C, H, W]
    z_target = latents_lst[target_idx].unsqueeze(0).to(DEVICE) # Shape: [1, C, H, W]
    # prompt = prompt_lst[start_idx]
    prompt  = 'a photo of a fern from different angle'

    print(f"Moving from latent {start_idx} to {target_idx}.")
    print(f"Using prompt: '{prompt}'")

    # 4. Prepare for optimization
    # Get conditional and unconditional text embeddings for the prompt
    _, text_embeddings = get_text_embedding(ldm_stable, [prompt])
    uncond_embeddings = get_uncond_embedding(ldm_stable, batch_size=1)
    full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # The latent vector we will optimize, initialized with the starting latent
    z_current = z_start.clone().detach().requires_grad_(True)

    # Precompute L = number of elements in the latent (excluding batch dim)
    L = z_current.numel()  # e.g., C*H*W
    target_norm_value = math.sqrt(L)
    target_norm = torch.tensor(target_norm_value, device=DEVICE)

    # Setup optimizer
    optimizer = torch.optim.Adam([z_current], lr=LEARNING_RATE)

    # Configure the scheduler with the number of timesteps
    ldm_stable.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)

    # 5. Run Gradient Descent Loop
    for step in tqdm(range(NUM_OPTIMIZATION_STEPS), desc="Optimizing latent path"):
        optimizer.zero_grad()

        # --- A. Calculate Diffusion Loss ---
        # This loss ensures the latent stays on the learned distribution (i.e., looks plausible to the model)
        with torch.no_grad():
            # Sample a random timestep
            t = torch.randint(0, ldm_stable.scheduler.config.num_train_timesteps, (1,)).long().to(DEVICE)

        # Add noise to the current latent to get z_t
        noise = torch.randn_like(z_current)*3
        z_t = ldm_stable.scheduler.add_noise(z_current, noise, t)

        # Predict the noise using the U-Net with classifier-free guidance
        latent_model_input = torch.cat([z_t] * 2)  # Duplicate for CFG
        with torch.no_grad():
            noise_pred_uncond, noise_pred_text = ldm_stable.unet(
                latent_model_input, t, encoder_hidden_states=full_text_embeddings
            ).sample.chunk(2)

        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        # The diffusion loss is the MSE between the model's noise prediction and the actual noise added.
        loss_diffusion = F.mse_loss(noise_pred, noise)

        # --- B. Calculate Similarity Loss ---
        # This loss pulls the current latent towards the target latent.
        loss_similarity = F.mse_loss(z_current, z_target)

        # --- C. Calculate Norm‐Regularization Loss ---
        # We want ||z_current|| ≈ sqrt(L), where L is latent dimensionality.
        latent_norm = torch.norm(z_current)
        loss_norm = (latent_norm - target_norm).pow(2)

        # print("Similarity Loss",loss_similarity)

        # --- D. Combine losses and update ---
        total_loss = (
            loss_diffusion
            + SIMILARITY_LAMBDA * loss_similarity
            + NORM_LAMBDA * loss_norm
        )

        total_loss.backward()
        optimizer.step()

        # --- E. Store image at the current step ---
        # We detach the latent from the computation graph before decoding
        if(step%50 == 0):
            file_path = os.path.join(OUTPUT_DIR, f"step_{step:04d}.png")
            decode_and_save_image(ldm_stable, z_current.detach(), file_path)

    # Save the very last frame
    file_path = os.path.join(OUTPUT_DIR, f"step_{NUM_OPTIMIZATION_STEPS:04d}_final.png")
    decode_and_save_image(ldm_stable, z_current.detach(), file_path)

    print(f"\nOptimization finished. {NUM_OPTIMIZATION_STEPS + 1} images saved in '{OUTPUT_DIR}'")
