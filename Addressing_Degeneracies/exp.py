import os
import math
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionPipeline
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
LATENT_SCALING = 0.18215  # stable-diffusion v1.x uses this constant

# Paths to your two 512×512 RGB images (replace with your actual file paths)
IMG_PATH_A = "../images/fern_1.JPG"
IMG_PATH_B = "../images/gnochi_mirror.jpeg"

# Where to save the outputs
OUTPUT_DIR = "./mean_and_fix_results"
MEAN_SUBDIR = "mean"  # channel‐wise mean–adjusted subfolder
FIX_SUBDIR  = "fix"   # fixed‐norm subfolder
os.makedirs(os.path.join(OUTPUT_DIR, MEAN_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, FIX_SUBDIR), exist_ok=True)

# Number of interpolation steps (including endpoints at t=0.0 and t=1.0)
NUM_STEPS = 30

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helper Functions: preprocess / postprocess / encode / decode
# ──────────────────────────────────────────────────────────────────────────────

# 2a) Preprocessing: resize + normalize to [-1, +1]
_preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),                   # [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3),  # => [-1,+1]
])

# 2b) Postprocessing: from [-1,+1] float32 => PIL Image
_postprocess = transforms.Compose([
    transforms.Normalize([-1.0]*3, [2.0]*3),         # => [0,1]
    transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0)),
    transforms.Lambda(lambda t: (t * 255.0).round().byte()),
    transforms.Lambda(lambda t: t.permute(1, 2, 0).cpu().numpy()),  # C×H×W => H×W×C
    transforms.Lambda(lambda arr: Image.fromarray(arr)),
])

def load_image_as_tensor(path: str) -> torch.Tensor:
    """
    1) Load a PIL image from disk,
    2) Resize to 512×512, normalize to [-1, +1],
    3) Return a Tensor of shape [1, 3, 512, 512], dtype=torch.float32.
    """
    img = Image.open(path).convert("RGB")
    img_t = _preprocess(img).unsqueeze(0)  # [1, 3, 512, 512], float32
    return img_t

# 2c) Encode an RGB image‐tensor => latent [1, 4, 64, 64]
def image_to_latent(image_tensor: torch.Tensor, vae) -> torch.Tensor:
    """
    Given a preprocessed image tensor [1, 3, 512, 512] in float32,
    cast it to the VAE’s dtype (float16), then run VAE.encode => latent.
    Returns [1, 4, 64, 64], dtype=float16.
    """
    image_tensor = image_tensor.to(DEVICE).to(vae.dtype)  # ensure same dtype
    with torch.no_grad():
        lat_dist = vae.encode(image_tensor).latent_dist
        latent = lat_dist.sample()            # [1, 4, 64, 64], float16
        latent = latent * LATENT_SCALING
    return latent

# 2d) Decode a latent [1, 4, 64, 64] => RGB image PIL
def latent_to_image(latent: torch.Tensor, vae) -> Image.Image:
    """
    Given a latent [1, 4, 64, 64] (dtype=float16),
    divide by LATENT_SCALING and pass to VAE.decode => [1, 3, 512, 512], dtype=float16.
    Convert to PIL.
    """
    latent_in = latent.to(DEVICE).to(vae.dtype) / LATENT_SCALING
    with torch.no_grad():
        out = vae.decode(latent_in).sample      # [1, 3, 512, 512], float16
    image_tensor = out.squeeze(0).to(torch.float32)  # [3, 512, 512], float32
    pil = _postprocess(image_tensor)                 # convert to PIL
    return pil

# ──────────────────────────────────────────────────────────────────────────────
# 3) Interpolation Functions
# ──────────────────────────────────────────────────────────────────────────────

def mean_adjusted_interpolation(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Channel‐wise mean–adjusted interpolation (NIN on noise part):
      1. d0 = channel‐wise mean of z0, broadcast to [C,H,W]
         d1 = channel‐wise mean of z1, broadcast to [C,H,W]
         e0 = z0 - d0,  e1 = z1 - d1
      2. weighted_norm = (1−t)·||e0|| + t·||e1||
         bar_e = (1−t)·e0 + t·e1
         e' = (weighted_norm / ||bar_e||) · bar_e
      3. d' = (1−t)·d0 + t·d1
      4. return z' = d' + e'
    """
    # Remove batch dimension => work in [C, H, W]
    z0_ = z0.squeeze(0)  # [C, H, W]
    z1_ = z1.squeeze(0)

    w0, w1 = 1.0 - t, t

    # 1) Channel‐wise means
    mu0 = z0_.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
    mu1 = z1_.mean(dim=(1, 2), keepdim=True)

    d0 = mu0.expand_as(z0_)  # [C, H, W]
    d1 = mu1.expand_as(z1_)

    e0 = z0_ - d0
    e1 = z1_ - d1

    # 2) Norm‐Interpolated‐Norm (NIN) on noise
    n0 = e0.norm(p=2)
    n1 = e1.norm(p=2)
    weighted_norm = w0 * n0 + w1 * n1

    bar_e = w0 * e0 + w1 * e1
    norm_bar_e = bar_e.norm(p=2).clamp_min(1e-8)
    e_prime = bar_e * (weighted_norm / norm_bar_e)

    # 3) Linear interpolation on mean parts
    d_prime = w0 * d0 + w1 * d1

    # 4) Combine and restore batch dim
    z_prime = (d_prime + e_prime).unsqueeze(0)  # [1, C, H, W]
    return z_prime

def fixed_norm_interpolation(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Fixed-norm interpolation (fFIX):
      1. bar = (1−t)·z0 + t·z1
      2. α = sqrt(L) / ||bar||
      3. return α·bar
    Where L = total #elements in bar (C*H*W).
    """
    w0, w1 = 1.0 - t, t
    bar = w0 * z0 + w1 * z1  # [1, C, H, W]
    # Compute total latent dimension L = C * H * W
    L = float(bar.numel() / bar.shape[0])  # since bar.shape[0] == 1
    # bar.norm() is the Frobenius norm over all C×H×W
    norm_bar = bar.norm(p=2).clamp_min(1e-8)
    alpha = math.sqrt(L) / norm_bar
    z_prime = bar * alpha
    return z_prime

# ──────────────────────────────────────────────────────────────────────────────
# 4) Main: load model, load images, encode, interpolate loops, decode, save
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 4a) Load Stable Diffusion Pipeline (we only need the VAE here)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)
    vae = pipe.vae

    # 4b) Load & preprocess two images
    imgA = load_image_as_tensor(IMG_PATH_A)  # [1, 3, 512, 512], float32
    imgB = load_image_as_tensor(IMG_PATH_B)

    # 4c) Encode both into latent space => [1, 4, 64, 64], float16
    zA = image_to_latent(imgA, vae)
    zB = image_to_latent(imgB, vae)

    # 4d) Loop over 30 interpolation steps from t=0.0 → t=1.0
    for step_idx, t in enumerate(np.linspace(0.0, 1.0, NUM_STEPS)):
        # 4d.1) Channel‐wise mean–adjusted interpolation
        z_mean = mean_adjusted_interpolation(zA, zB, float(t))  # [1,4,64,64]
        # 4d.2) Fixed‐norm interpolation
        z_fix  = fixed_norm_interpolation(zA, zB, float(t))     # [1,4,64,64]

        # 4e) Decode both latents back to PIL images
        with torch.no_grad():
            img_mean = latent_to_image(z_mean, vae)
            img_fix  = latent_to_image(z_fix, vae)

        # 4f) Save each frame under separate subfolders
        out_mean_path = os.path.join(
            OUTPUT_DIR, MEAN_SUBDIR, f"step_{step_idx:02d}_t{t:.2f}_mean.png"
        )
        out_fix_path  = os.path.join(
            OUTPUT_DIR, FIX_SUBDIR,  f"step_{step_idx:02d}_t{t:.2f}_fix.png"
        )
        img_mean.save(out_mean_path)
        img_fix.save(out_fix_path)

        print(f"Saved step {step_idx:02d} at t={t:.2f} →")
        print(f"    • mean‐adjusted : {out_mean_path}")
        print(f"    • fixed‐norm    : {out_fix_path}")

if __name__ == "__main__":
    main()
