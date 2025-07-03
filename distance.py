import os
import math
import numpy as np
from glob import glob
from natsort import natsorted
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# ──────────────────────────────────────────────────────────────────────────────
# 1) Print all the details about the diffusion model used
# ──────────────────────────────────────────────────────────────────────────────

print("Setting up the diffusion model...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5

# Initialize the DDIM scheduler exactly as in your reference code
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)

# Load Stable Diffusion v1.4 with this scheduler (using safetensors)
ldm_stable = (
    StableDiffusionPipeline
    .from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_safetensors=True)
    .to(DEVICE)
)

print("\n" + "="*30)
print("=== Diffusion Model Details ===")
print("="*30 + "\n")

print("-- Scheduler Configuration --")
print(scheduler, "\n")
print("-- Stable Diffusion Pipeline --")
print(ldm_stable, "\n")
print("-- Tokenizer Config (CLIP) --")
print(ldm_stable.tokenizer, "\n")
print("-- VAE Configuration --")
print(ldm_stable.vae, "\n")
print("-- UNet Configuration --")
print(ldm_stable.unet, "\n")
print("-- Text Encoder (CLIP) Configuration --")
print(ldm_stable.text_encoder, "\n")
print(f"GUIDANCE_SCALE: {GUIDANCE_SCALE}")
print(f"NUM_DIFFUSION_STEPS: {NUM_DIFFUSION_STEPS}")
print("\n" + "="*30 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 2) Create dummy latent data (20 random latents per prompt)
# ──────────────────────────────────────────────────────────────────────────────

print("Creating dummy latent data for demonstration purposes...")
prompt_list = [
    "a cat",
    "a fern",
    "a close-up of a fern",
    "a picture of a fern",
    "a green plant with fronds",
    "a botanical fern portrait",
    "a macro shot of a fern",
    "a lone fern leaf",
    "a cluster of ferns in the forest",
    "a potted fern",
    "a small wild fern",
    "a fern under sunlight",
    "a fresh green fern",
    "a detailed fern plant",
    "a fern against a white background",
    "a fern in a forest",
    "a young fern shoot",
    "a fern sprouting",
    "a delicate fern frond",
    "a tropical fern plant",
    "a single fern leaf on wood",
]

# For each prompt, create a folder and save 20 random latent tensors
for prompt in prompt_list:
    prompt_slug = prompt.replace(" ", "_")
    latent_folder = f"results/fern_{prompt_slug}/inverted_latent"
    os.makedirs(latent_folder, exist_ok=True)

    for i in range(20):
        dummy_latent = torch.randn(1, 4, 64, 64)
        filename = f"{i}_final_latent.pt"
        torch.save(dummy_latent, os.path.join(latent_folder, filename))

print("Dummy data created.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 3) Compute pairwise Euclidean norms between the 0th latent of each prompt
# ──────────────────────────────────────────────────────────────────────────────

# First, load the 0th latent (index 0) for each prompt into a list
latent0_vectors = []
for prompt in prompt_list:
    prompt_slug = prompt.replace(" ", "_")
    latent0_path = os.path.join(
        f"results/fern_{prompt_slug}/inverted_latent", "0_final_latent.pt"
    )

    if not os.path.isfile(latent0_path):
        raise FileNotFoundError(f"Expected latent file not found: {latent0_path}")

    tensor = torch.load(latent0_path, map_location="cpu")
    # Squeeze out the batch dimension if needed
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    # Flatten to a 1D NumPy array
    vec = tensor.flatten().numpy()
    latent0_vectors.append(vec)

# Compute and print all pairwise distances
print("Euclidean norms between the 0th latent vector of each prompt:\n")
num_prompts = len(prompt_list)
for i in range(num_prompts):
    for j in range(i + 1, num_prompts):
        dist = np.linalg.norm(latent0_vectors[i] - latent0_vectors[j])
        print(
            f"Distance between prompt [{i}] '{prompt_list[i]}' "
            f"and prompt [{j}] '{prompt_list[j]}': {dist:.4f}"
        )
print("\n")