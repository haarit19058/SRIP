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


# This part of the code requires pre-generated latent vectors.
# We will create a dummy directory structure and some fake latent vectors
# to make the script runnable without prior data generation.

print("Creating dummy latent data for demonstration purposes...")
prompt_list = [
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

# Create fake latents for each prompt
for prompt in prompt_list:
    latent_folder = f"results/fern_{prompt}/inverted_latent"
    os.makedirs(latent_folder, exist_ok=True)
    # Generate 20 random tensors representing inverted latents
    for i in range(20):
        # A real latent would be shape [1, 4, 64, 64]
        dummy_latent = torch.randn(1, 4, 64, 64)
        # UPDATED: Save with the new filename format
        prompt_slug = prompt.replace(" ", "_")
        torch.save(dummy_latent, os.path.join(latent_folder, f"{i}_final_latent_*.pt"))

print("Dummy data created. Starting main processing loop.\n")
# ──────────────────────────────────────────────────────────────────────────────
# START OF THE MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

for prompt in prompt_list:
    print(f"\n===== Processing prompt: '{prompt}' =====\n")

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) For the stored latent vectors, do PCA for “base” latents
    # ──────────────────────────────────────────────────────────────────────────────

    # (a) Load all inverted latents from your directory.
    latent_folder = f"results/fern_{prompt}/inverted_latent"
    savedir = f"results/fern_{prompt}"
    # UPDATED: Changed glob pattern to match new filenames
    latent_paths = natsorted(glob(os.path.join(latent_folder, "*_final_latent_*.pt")))

    if not latent_paths:
        print(f"Warning: No latent files found in '{latent_folder}'. Skipping this prompt.")
        continue

    base_latents_list = []
    for p in latent_paths:
        t = torch.load(p, map_location="cpu")
        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        base_latents_list.append(t.flatten().numpy())

    base_latents = np.stack(base_latents_list, axis=0)
    N, D = base_latents.shape
    print(f"Loaded {N} base latent vectors (each of dimension {D}).")

    # (b) Fit PCA on the base_latents (to 2D and 3D)
    pca_base_2d = PCA(n_components=2)
    base_pca_2d = pca_base_2d.fit_transform(base_latents)

    pca_base_3d = PCA(n_components=3)
    base_pca_3d = pca_base_3d.fit_transform(base_latents)

    print("\nExplained variance ratios (base PCA, 2 components):")
    print(pca_base_2d.explained_variance_ratio_)
    print("Explained variance ratios (base PCA, 3 components):")
    print(pca_base_3d.explained_variance_ratio_)
    print("\n")

    # (c) Pick two latents A and B for interpolation
    idx_a = 0
    idx_b = min(10, N - 1)

    flat_A = base_latents[idx_a]
    flat_B = base_latents[idx_b]

    A_2d = pca_base_2d.transform(flat_A.reshape(1, -1))[0]
    B_2d = pca_base_2d.transform(flat_B.reshape(1, -1))[0]
    A_3d = pca_base_3d.transform(flat_A.reshape(1, -1))[0]
    B_3d = pca_base_3d.transform(flat_B.reshape(1, -1))[0]

    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Define interpolation functions (NumPy)
    # ──────────────────────────────────────────────────────────────────────────────

    def interp_linear_np(z0, z1, t):
        return t * z0 + (1 - t) * z1

    def interp_fix_np(z0, z1, t):
        bar = t * z0 + (1 - t) * z1
        L = float(bar.size)
        norm_bar = np.linalg.norm(bar)
        if norm_bar < 1e-8:
            return bar.copy()
        return bar * (math.sqrt(L) / norm_bar)

    def interp_nin_np(z0, z1, t):
        w0, w1 = t, 1 - t
        n0 = np.linalg.norm(z0)
        n1 = np.linalg.norm(z1)
        weighted_norm = w0 * n0 + w1 * n1
        bar = w0 * z0 + w1 * z1
        norm_bar = np.linalg.norm(bar)
        if norm_bar < 1e-8:
            return bar.copy()
        return bar * (weighted_norm / norm_bar)

    def interp_slerp_np(z0, z1, t):
        dot = np.dot(z0, z1)
        norm0 = np.linalg.norm(z0)
        norm1 = np.linalg.norm(z1)
        if norm0 < 1e-8 or norm1 < 1e-8:
            return interp_linear_np(z0, z1, t)
        cos_ω = np.clip(dot / (norm0 * norm1), -1 + 1e-7, 1 - 1e-7)
        ω = math.acos(cos_ω)
        sin_ω = math.sin(ω)
        if sin_ω < 1e-8:
            return interp_linear_np(z0, z1, t)
        factor0 = math.sin((1 - t) * ω) / sin_ω
        factor1 = math.sin(t * ω) / sin_ω
        return factor0 * z0 + factor1 * z1

    def interp_mean_np(z0, z1, t):
        w0, w1 = 1 - t, t
        mu0 = z0.mean()
        mu1 = z1.mean()
        d0 = np.full_like(z0, mu0)
        d1 = np.full_like(z1, mu1)
        e0 = z0 - d0
        e1 = z1 - d1
        n0 = np.linalg.norm(e0)
        n1 = np.linalg.norm(e1)
        weighted_norm = w0 * n0 + w1 * n1
        bar_e = w0 * e0 + w1 * e1
        norm_bar_e = np.linalg.norm(bar_e)
        if norm_bar_e < 1e-8:
            e_prime = bar_e.copy()
        else:
            e_prime = bar_e * (weighted_norm / norm_bar_e)
        d_prime = w0 * d0 + w1 * d1
        return d_prime + e_prime

    methods = {
        "linear": interp_linear_np, "fix": interp_fix_np,
        "nin": interp_nin_np, "slerp": interp_slerp_np,
        "mean": interp_mean_np
    }
    print(f"Using latent indices for interpolation: A={idx_a}, B={idx_b}.\n")

    # ──────────────────────────────────────────────────────────────────────────────
    # 4) For each method, build a 30-step trajectory and project into base PCA space
    # ──────────────────────────────────────────────────────────────────────────────

    num_steps = 30
    t_values = np.linspace(0, 1, num_steps)
    trajectory_coords_2d = {}
    trajectory_coords_3d = {}

    print("=== Projecting interpolation trajectories into base PCA space ===\n")
    for name, func in methods.items():
        traj_flat = np.stack([func(flat_A, flat_B, float(t)) for t in t_values], axis=0)
        trajectory_coords_2d[name] = pca_base_2d.transform(traj_flat)
        trajectory_coords_3d[name] = pca_base_3d.transform(traj_flat)
        print(f" -> {name.upper()} trajectory projected into base PCA basis.")
    print("\n")

    # ──────────────────────────────────────────────────────────────────────────────
    # 5) Plot “base PCA latents” + “all trajectories” together in 2D & 3D
    # ──────────────────────────────────────────────────────────────────────────────

    os.makedirs(savedir, exist_ok=True)
    print(f"Saving plots to directory: {savedir}\n")

    # 5a) 2D plot with annotations
    plt.figure(figsize=(10, 8))
    # Plot base latents
    plt.scatter(base_pca_2d[:, 0], base_pca_2d[:, 1], alpha=0.25, s=25, color="gray", label="Base Latents")
    
    # NEW: Annotate each base latent with its index
    for i, p in enumerate(base_pca_2d):
        plt.text(p[0], p[1], str(i), color="dimgray", fontsize=7, ha='center', va='center')

    # Highlight A and B
    plt.scatter([A_2d[0]], [A_2d[1]], color="red", s=100, marker="X", zorder=5, label=f"Latent A (idx {idx_a})")
    plt.scatter([B_2d[0]], [B_2d[1]], color="blue", s=100, marker="X", zorder=5, label=f"Latent B (idx {idx_b})")

    # Overlay trajectories
    for name, coords in trajectory_coords_2d.items():
        plt.plot(coords[:, 0], coords[:, 1], marker='.', markersize=5, linestyle='-', label=name, alpha=0.9, zorder=3)
        plt.scatter(coords[0, 0], coords[0, 1], color="black", s=50, marker="D", zorder=4)
        plt.scatter(coords[-1, 0], coords[-1, 1], color="black", s=50, marker="s", zorder=4)

    plt.title(f"Base Latents + Interpolation Trajectories (2D PCA)\nPrompt: '{prompt}'")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{savedir}/pca_combined_all_2d_annotated.jpg", dpi=150)
    plt.close()

    # 5b) 3D plot with annotations
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot base latents
    ax.scatter(base_pca_3d[:, 0], base_pca_3d[:, 1], base_pca_3d[:, 2],
            alpha=0.15, s=25, color="gray", label="Base Latents")
            
    # NEW: Annotate each base latent with its index
    for i, p in enumerate(base_pca_3d):
        ax.text(p[0], p[1], p[2], str(i), color="dimgray", fontsize=7)

    # Highlight A and B
    ax.scatter([A_3d[0]], [A_3d[1]], [A_3d[2]], color="red", s=120, marker="X", label=f"Latent A (idx {idx_a})", depthshade=False)
    ax.scatter([B_3d[0]], [B_3d[1]], [B_3d[2]], color="blue", s=120, marker="X", label=f"Latent B (idx {idx_b})", depthshade=False)

    # Overlay trajectories
    for name, coords in trajectory_coords_3d.items():
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker='.', markersize=4, linestyle='-', label=name, alpha=0.9)
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="black", s=60, marker="D")
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="black", s=60, marker="s")

    ax.set_title(f"Base Latents + Interpolation Trajectories (3D PCA)\nPrompt: '{prompt}'", pad=20)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{savedir}/pca_combined_all_3d_annotated.jpg", dpi=150)
    plt.close()

print("\nProcessing complete for all prompts.")