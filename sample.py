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
NUM_OPTIMIZATION_STEPS = 1000    # Number of steps for gradient descent
LEARNING_RATE = 0.003             # Learning rate for the optimizer
SIMILARITY_LAMBDA = 0         # Weight for the similarity loss term (pulls towards target)
NORM_LAMBDA = 1           # Weight for the latent‐norm loss term
GUIDANCE_SCALE = 7.5             # For the diffusion loss calculation
NUM_DIFFUSION_STEPS = 100         # Number of timesteps for the scheduler

# Paths
scene_name = 'f1ern_shifted'
LATENTS_DIR = f'./results/{scene_name}/inversion/inverted_latent/'  # Directory with pre‐computed latents
OUTPUT_DIR = f'./samples_{scene_name}/'

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


latent_file = natsorted(glob(os.path.join(LATENTS_DIR, 'shifted.pt')))[0]
# print(latent_file.shape)


basename = os.path.basename(latent_file)
tokens = os.path.splitext(basename)[0].split('_')
prompt_str = ' '.join(tokens[3:])

loaded = torch.load(latent_file, map_location=DEVICE)
print(loaded.shape)
if loaded.ndim == 4 and loaded.shape[0] == 1:
    loaded = loaded.squeeze(0)
    
    
    

# -----------------------------------------------------------------------------
# ─── SAMPLE AROUND LATENT AND SAVE IMAGES ────────────────────────────────────
# -----------------------------------------------------------------------------

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling parameters
NUM_SAMPLES = 100     # how many perturbed versions to generate
RADIUS = 10        # scale of perturbation around the original latent

print(f"Sampling {NUM_SAMPLES} latents around the original...")

for i in range(NUM_SAMPLES):
    # 1) generate a random direction of the same shape as the latent
    direction = torch.randn_like(loaded, device=DEVICE)
    # 2) normalize it and scale by RADIUS
    direction = direction / direction.norm() * RADIUS
    # 3) apply the perturbation
    sampled_latent = loaded + direction
    # 4) decode and save
    out_path = os.path.join(OUTPUT_DIR, f"sample_{i:03d}.png")
    # decode_and_save_image expects a batch dimension
    decode_and_save_image(ldm_stable, sampled_latent.unsqueeze(0), out_path)

print("Done sampling – images saved to:", OUTPUT_DIR)
