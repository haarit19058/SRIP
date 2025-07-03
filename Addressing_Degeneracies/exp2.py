import os
import math
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
LATENT_SCALING = 0.18215  # latent scaling factor for SD v1.x

# Interpolation settings
NUM_STEPS = 30
PROMPT = "a frog"

# Output directories
OUTPUT_DIR = "./frog_interpolation"
MEAN_SUBDIR = "mean_adjusted"  # for channel-wise mean–adjusted results
FIX_SUBDIR  = "fixed_norm"     # for fixed-norm results

os.makedirs(os.path.join(OUTPUT_DIR, MEAN_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, FIX_SUBDIR), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Helper Functions: preprocess / postprocess / encode / decode
# ──────────────────────────────────────────────────────────────────────────────

# 2a) Preprocessing: resize → 512×512, normalize to [-1, +1]
_preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),                    # [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3),   # → [-1,+1]
])

# 2b) Postprocessing: from [-1,+1] float32 → PIL Image
_postprocess = transforms.Compose([
    transforms.Normalize([-1.0]*3, [2.0]*3),         # → [0,1]
    transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0)),
    transforms.Lambda(lambda t: (t * 255.0).round().byte()),
    transforms.Lambda(lambda t: t.permute(1, 2, 0).cpu().numpy()),  # C×H×W → H×W×C
    transforms.Lambda(lambda arr: Image.fromarray(arr)),
])

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image (RGB, 512×512) to a float32 tensor [1,3,512,512] in [-1,+1].
    """
    return _preprocess(img).unsqueeze(0)  # [1,3,512,512], float32

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a Tensor [3,512,512] in [-1,+1]/float32 to a PIL Image.
    """
    return _postprocess(tensor)

def image_to_latent(image_tensor: torch.Tensor, vae) -> torch.Tensor:
    """
    Encode a preprocessed image ([1,3,512,512], float32) into a latent [1,4,64,64], float16.
    """
    # Cast input to the VAE’s dtype
    image_tensor = image_tensor.to(DEVICE).to(vae.dtype)
    with torch.no_grad():
        lat_dist = vae.encode(image_tensor).latent_dist
        latent = lat_dist.sample()  # [1,4,64,64], float16
        latent = latent * LATENT_SCALING
    return latent  # [1,4,64,64], float16

def latent_to_image(latent: torch.Tensor, vae) -> Image.Image:
    """
    Decode a latent [1,4,64,64], float16 → [1,3,512,512], float16 → PIL (float32).
    """
    # Ensure latent is float16
    latent_in = latent.to(DEVICE).to(vae.dtype) / LATENT_SCALING
    with torch.no_grad():
        out = vae.decode(latent_in).sample  # [1,3,512,512], float16
    img_t = out.squeeze(0).to(torch.float32)  # [3,512,512], float32 in [-1,+1]
    return tensor_to_pil(img_t)  # PIL Image


# ──────────────────────────────────────────────────────────────────────────────
# 3) Interpolation Functions
# ──────────────────────────────────────────────────────────────────────────────

def mean_adjusted_interpolation(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Channel‐wise mean–adjusted interpolation between two latents z0, z1 ([1,4,64,64], float16):
      1. d0 = channel‐wise mean of z0 → [4,1,1],  broadcast to [4,64,64]
         d1 = channel‐wise mean of z1 → [4,1,1],  broadcast to [4,64,64]
         e0 = z0 - d0,  e1 = z1 - d1
      2. weighted_norm = (1−t)·||e0|| + t·||e1||
         bar_e = (1−t)·e0 + t·e1
         e' = (weighted_norm / ||bar_e||) · bar_e
      3. d' = (1−t)·d0 + t·d1
      4. return z' = d' + e', shape [1,4,64,64], float16.
    """
    # Remove batch dimension → [4,64,64]
    z0_ = z0.squeeze(0)
    z1_ = z1.squeeze(0)

    w0, w1 = 1.0 - t, t

    # 1) Channel-wise means
    mu0 = z0_.mean(dim=(1,2), keepdim=True)  # [4,1,1]
    mu1 = z1_.mean(dim=(1,2), keepdim=True)

    d0 = mu0.expand_as(z0_)  # [4,64,64]
    d1 = mu1.expand_as(z1_)

    e0 = z0_ - d0
    e1 = z1_ - d1

    # 2) Norm-Interpolated-Norm (NIN) on noise parts
    n0 = e0.norm(p=2)
    n1 = e1.norm(p=2)
    weighted_norm = w0 * n0 + w1 * n1

    bar_e = w0 * e0 + w1 * e1
    norm_bar_e = bar_e.norm(p=2).clamp_min(1e-8)
    e_prime = bar_e * (weighted_norm / norm_bar_e)

    # 3) Linear interpolation on mean parts
    d_prime = w0 * d0 + w1 * d1

    # 4) Combine → [4,64,64], then restore batch → [1,4,64,64]
    z_prime = (d_prime + e_prime).unsqueeze(0).to(z0.dtype)
    return z_prime

def fixed_norm_interpolation(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Fixed‐Norm interpolation between two latents z0, z1 ([1,4,64,64], float16):
      1. bar = (1−t)·z0 + t·z1
      2. L = total #elements per sample = 4*64*64
      3. α = sqrt(L) / ||bar||
      4. return α·bar
    """
    w0, w1 = 1.0 - t, t
    bar = w0 * z0 + w1 * z1  # [1,4,64,64]
    L = float(bar.numel() // bar.shape[0])  # e.g. 4*64*64 = 16384
    norm_bar = bar.norm(p=2).clamp_min(1e-8)
    alpha = math.sqrt(L) / norm_bar
    z_prime = bar * alpha
    return z_prime


# ──────────────────────────────────────────────────────────────────────────────
# 4) Main: generate two frog images, encode, interpolate, decode, save
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 4a) Load Stable Diffusion (we need both text2img and VAE)
    #    We’ll use a DDIMScheduler to match inversion‐style pipelines when encoding.
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        clip_sample=False, 
        set_alpha_to_one=False, 
        steps_offset=1
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, scheduler=scheduler, torch_dtype=torch.float16
    ).to(DEVICE)
    vae = pipe.vae

    # 4b) Generate two “frog” images with different random seeds
    # ──────────────────────────────────────────────────────────────────────────────
    # First frog (seed = 42)
    generatorA = torch.Generator(device=DEVICE).manual_seed(42)
    outA = pipe(
        PROMPT, 
        num_inference_steps=50, 
        guidance_scale=7.5, 
        generator=generatorA
    ).images[0]  # PIL Image

    # Second frog (seed = 1234)
    generatorB = torch.Generator(device=DEVICE).manual_seed(1234)
    outB = pipe(
        PROMPT, 
        num_inference_steps=50, 
        guidance_scale=7.5, 
        generator=generatorB
    ).images[0]  # PIL Image

    # Save the two base frogs for reference
    outA.save(os.path.join(OUTPUT_DIR, "frog_A.png"))
    outB.save(os.path.join(OUTPUT_DIR, "frog_B.png"))
    print(f"Saved base frogs → frog_A.png and frog_B.png")

    # 4c) Convert the two PIL images to preprocessed Tensors [1,3,512,512]
    imgA_t = pil_to_tensor(outA)
    imgB_t = pil_to_tensor(outB)

    # 4d) Encode each to latent space [1,4,64,64], float16
    zA = image_to_latent(imgA_t, vae)
    zB = image_to_latent(imgB_t, vae)

    # 4e) For each t in 30 steps, do both interpolations, decode, and save
    for step_idx, t in enumerate(np.linspace(0.0, 1.0, NUM_STEPS)):
        # (i) Channel-wise mean–adjusted
        z_mean = mean_adjusted_interpolation(zA, zB, float(t))
        img_mean = latent_to_image(z_mean, vae)
        out_mean = os.path.join(
            OUTPUT_DIR, MEAN_SUBDIR, f"step_{step_idx:02d}_t{t:.2f}.png"
        )
        img_mean.save(out_mean)

        # (ii) Fixed-norm
        z_fix = fixed_norm_interpolation(zA, zB, float(t))
        img_fix = latent_to_image(z_fix, vae)
        out_fix = os.path.join(
            OUTPUT_DIR, FIX_SUBDIR, f"step_{step_idx:02d}_t{t:.2f}.png"
        )
        img_fix.save(out_fix)

        print(f"Step {step_idx:02d} (t={t:.2f}):")
        print(f"  • Mean‐adjusted saved to  {out_mean}")
        print(f"  • Fixed‐norm    saved to  {out_fix}")

if __name__ == "__main__":
    main()
