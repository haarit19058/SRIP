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

print("=== Diffusion Model Details ===\n")

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
print("\n") 


# ──────────────────────────────────────────────────────────────────────────────
# 2) For the stored latent vectors, do PCA for “base” latents
# ──────────────────────────────────────────────────────────────────────────────

# (a) Load all inverted latents from your directory. 
latent_folder = "results/inversion/inverted_latent"
latent_paths = natsorted(glob(os.path.join(latent_folder, "*_final_latent_*.pt")))

base_latents_list = []
for p in latent_paths:
    t = torch.load(p, map_location="cpu")
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    base_latents_list.append(t.flatten().numpy())

base_latents = np.stack(base_latents_list, axis=0)  # shape [N, D]
N, D = base_latents.shape
print(f"Loaded {N} base latent vectors (each of dimension {D}).")

# (b) Fit PCA on the base_latents (to 2D and 3D)
pca_base_2d = PCA(n_components=2)
base_pca_2d = pca_base_2d.fit_transform(base_latents)

pca_base_3d = PCA(n_components=3)
base_pca_3d = pca_base_3d.fit_transform(base_latents)

print("Explained variance ratios (base PCA, 2 components):")
print(pca_base_2d.explained_variance_ratio_)
print("Explained variance ratios (base PCA, 3 components):")
print(pca_base_3d.explained_variance_ratio_)
print("\n")

# (c) Pick two latents A and B for interpolation
idx_a = 0
idx_b = min(10, N - 1)

flat_A = base_latents[idx_a]
flat_B = base_latents[idx_b]

# Project A and B into the same base PCA space
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
    "linear": interp_linear_np,
    "fix":    interp_fix_np,
    "nin":    interp_nin_np,
    "slerp":  interp_slerp_np,
    "mean":   interp_mean_np
}

print(f"Using latent indices: A={idx_a}, B={idx_b}.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4) For each method, build a 30-step trajectory and project into base PCA space
# ──────────────────────────────────────────────────────────────────────────────

num_steps = 30
t_values = np.linspace(0, 1, num_steps)

trajectory_coords_2d = {}
trajectory_coords_3d = {}

print("=== Projection of interpolation trajectories into base PCA space ===\n")
for name, func in methods.items():
    traj_flat = []
    for t in t_values:
        z_interp = func(flat_A, flat_B, float(t))
        traj_flat.append(z_interp)
    traj_flat = np.stack(traj_flat, axis=0)  # shape [30, D]

    # Project onto the base PCA we already fitted
    traj2d = pca_base_2d.transform(traj_flat)
    traj3d = pca_base_3d.transform(traj_flat)

    trajectory_coords_2d[name] = traj2d
    trajectory_coords_3d[name] = traj3d

    print(f"{name.upper()} trajectory projected into base PCA basis.")


# ──────────────────────────────────────────────────────────────────────────────
# 5) Plot “base PCA latents” + “all trajectories” together in 2D & 3D,
#    marking each trajectory point by its sequence index
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

# 5a) 2D combined plot
plt.figure(figsize=(10, 8))
# Plot base latents (gray, low-alpha)
plt.scatter(base_pca_2d[:, 0], base_pca_2d[:, 1], alpha=0.3, s=15, color="gray", label="Base Latents")

# Highlight A and B
plt.scatter([A_2d[0]], [A_2d[1]], color="red", s=60, marker="X", label="Latent A")
plt.scatter([B_2d[0]], [B_2d[1]], color="green", s=60, marker="X", label="Latent B")

# Overlay each method’s 30-step trajectory; annotate each point with its index
for name, coords in trajectory_coords_2d.items():
    plt.plot(coords[:, 0], coords[:, 1], marker="o", label=name, alpha=0.8)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=7, alpha=0.7)

plt.title("Base Latents + Interpolation Trajectories (2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/pca_combined_all_2d_numbered.jpg", dpi=150)
plt.show()


# 5b) 3D combined plot
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

# Plot base latents in gray
ax.scatter(base_pca_3d[:, 0], base_pca_3d[:, 1], base_pca_3d[:, 2],
           alpha=0.3, s=15, color="gray", label="Base Latents")

# Highlight A and B
ax.scatter([A_3d[0]], [A_3d[1]], [A_3d[2]], color="red", s=70, marker="X", label="Latent A")
ax.scatter([B_3d[0]], [B_3d[1]], [B_3d[2]], color="green", s=70, marker="X", label="Latent B")

# Overlay each method’s 30-step trajectory; annotate each point with its index
for name, coords in trajectory_coords_3d.items():
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker="o", label=name, alpha=0.8)
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i), fontsize=6, alpha=0.7)

ax.set_title("Base Latents + Interpolation Trajectories (3D PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend(loc="upper right")
plt.tight_layout()
fig.savefig("results/pca_combined_all_3d_numbered.jpg", dpi=150)
plt.show()
