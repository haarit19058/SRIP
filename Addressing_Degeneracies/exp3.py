import os
import math
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# We will scale VAE latents by this constant when encoding/decoding:
LATENT_SCALING = 0.18215

# Output directory (will contain two subfolders: "mean_adjusted/" and "fixed_norm/")
OUTPUT_DIR = "./frog_centroids"
MEAN_SUBFOLDER = "mean_adjusted"
FIX_SUBFOLDER  = "fixed_norm"
os.makedirs(os.path.join(OUTPUT_DIR, MEAN_SUBFOLDER), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, FIX_SUBFOLDER), exist_ok=True)

# Prompt and scheduler settings
PROMPT = "a photorealistic frog"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5

# The various N we want to try:
N_VALUES = [2, 4, 8, 16, 32, 128, 256]

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helper Functions: Preprocess / Postprocess / Encode / Decode
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
    Convert a Tensor [3,512,512] in [-1,+1] float32 to a PIL Image.
    """
    return _postprocess(tensor)

def image_to_latent(image_tensor: torch.Tensor, vae) -> torch.Tensor:
    """
    Encode a preprocessed image ([1,3,512,512], float32) via VAE.encode → latent [1,4,64,64], float16.
    We then multiply by LATENT_SCALING to match Stable Diffusion’s convention.
    """
    # 1) Cast to the VAE’s dtype (float16) and to DEVICE
    image_tensor = image_tensor.to(DEVICE).to(vae.dtype)
    with torch.no_grad():
        lat_dist = vae.encode(image_tensor).latent_dist  # AutoencoderKL returns a GaussianDistribution
        latent = lat_dist.sample()                       # [1,4,64,64], dtype=vae.dtype (float16)
        latent = latent * LATENT_SCALING
    return latent  # [1,4,64,64], float16

def latent_to_image(latent: torch.Tensor, vae) -> Image.Image:
    """
    Decode a latent [1,4,64,64] (float16, already scaled by LATENT_SCALING) back to a PIL image.
    Steps:
      1) Divide by LATENT_SCALING
      2) vae.decode(...) → [1,3,512,512] in float16 in [-1,+1]
      3) Convert to float32, clamp to [0,1], and to PIL.
    """
    latent_in = latent.to(DEVICE).to(vae.dtype) / LATENT_SCALING
    with torch.no_grad():
        out = vae.decode(latent_in).sample  # [1,3,512,512], float16 in [-1,+1]
    img_t = out.squeeze(0).to(torch.float32)   # [3,512,512], float32 in [-1,+1]
    return tensor_to_pil(img_t)                # convert to PIL

# ──────────────────────────────────────────────────────────────────────────────
# 3) Interpolation/Centroid Functions for N > 2
# ──────────────────────────────────────────────────────────────────────────────

def mean_adjusted_centroid(latents: torch.Tensor) -> torch.Tensor:
    """
    Given a stack of N latents z_n (shape [N, 4, 64, 64], dtype=float16),
    compute the channel-wise mean–adjusted centroid per the paper:
      1) d_n = channel‐mean of z_n → shape [N, 4, 1, 1], broadcastable
         e_n = z_n - d_n
      2) weighted_norm = (1/N)*sum_{n=1}^N ||e_n||
      3) bar_e = (1/N)*sum_{n=1}^N e_n
         e' = (weighted_norm / ||bar_e||) * bar_e
      4) d' = (1/N)*sum_{n=1}^N d_n  (broadcast to [4,64,64])
      5) return z' = d' + e', shape [1,4,64,64], float16
    """
    N, C, H, W = latents.shape
    # 1) Compute channel‐wise means for each latent: shape [N, C, 1, 1]
    mu = latents.mean(dim=(2,3), keepdim=True)  # [N, C, 1, 1]
    d = mu.expand(-1, -1, H, W)                 # [N, C, H, W]
    e = latents - d                             # [N, C, H, W]

    # 2) Compute each ||e_n|| (Frobenius norm over C×H×W), then average
    norms = e.view(N, -1).norm(dim=1, p=2)      # [N], each is ||e_n||
    weighted_norm = norms.mean()                # scalar

    # 3) Compute bar_e = (1/N)*sum e_n, then renormalize
    bar_e = e.mean(dim=0)                       # [C, H, W]
    norm_bar_e = bar_e.norm(p=2).clamp_min(1e-8)
    e_prime = bar_e * (weighted_norm / norm_bar_e)  # [C, H, W]

    # 4) Compute d' = (1/N)*sum d_n, shape [C, H, W]
    d_prime = d.mean(dim=0)                     # [C, H, W]

    # 5) Combine and re-add batch dimension to return [1, C, H, W]
    centroid = (d_prime + e_prime).unsqueeze(0)  # [1, C, H, W]
    return centroid.to(latents.dtype)            # float16

def fixed_norm_centroid(latents: torch.Tensor) -> torch.Tensor:
    """
    Given a stack of N latents z_n (shape [N, 4, 64, 64], dtype=float16),
    compute the fixed‐norm centroid (fFIX) per the paper:
      1) bar = (1/N)*sum_{n=1}^N z_n
      2) L = C * H * W = 4 * 64 * 64 = 16384
      3) α = sqrt(L) / ||bar||
      4) return z' = α * bar, shape [1,4,64,64], float16
    """
    N, C, H, W = latents.shape
    # 1) bar = average over N
    bar = latents.mean(dim=0, keepdim=True)     # [1, C, H, W]

    # 2) L = total #elements per sample
    L = float(C * H * W)                        # e.g. 4*64*64 = 16384

    # 3) compute α
    norm_bar = bar.norm(p=2).clamp_min(1e-8)
    alpha = math.sqrt(L) / norm_bar             # scalar

    # 4) scale
    centroid = bar * alpha                      # [1, C, H, W]
    return centroid.to(latents.dtype)            # float16

# ──────────────────────────────────────────────────────────────────────────────
# 4) Main: Generate, Encode, Compute Centroids, Decode, Save
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 4a) Load Stable Diffusion with a DDIMScheduler (float16)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        scheduler=scheduler,
        torch_dtype=torch.float16
    ).to(DEVICE)
    vae = pipe.vae

    # 4b) For each N in [2,4,8,16,32,128,256]:
    for N in N_VALUES:
        print(f"\n===== Computing centroids for N = {N} frog samples =====")

        # 4b.1) Generate N frog images with distinct seeds
        latents_list = []  # will hold N tensors of shape [1,4,64,64]
        for i in range(N):
            seed = 1000 + i  # arbitrary but distinct
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            out = pipe(
                PROMPT,
                num_inference_steps=NUM_DIFFUSION_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator
            ).images[0]  # PIL Image (512×512)

            # 4b.2) Convert that PIL to preprocessed Tensor [1,3,512,512]
            img_t = pil_to_tensor(out)  # float32

            # 4b.3) Encode the Tensor → latent [1,4,64,64], float16
            z = image_to_latent(img_t, vae)  # [1,4,64,64], float16
            latents_list.append(z)

        # Stack the N latents into a single tensor [N,4,64,64]
        latents_stack = torch.cat(latents_list, dim=0)  # float16

        # 4c) Compute the two centroids for these N latents
        centroid_mean = mean_adjusted_centroid(latents_stack)  # [1,4,64,64], float16
        centroid_fix  = fixed_norm_centroid(latents_stack)     # [1,4,64,64], float16

        # 4d) Decode each centroid back to PIL
        img_centroid_mean = latent_to_image(centroid_mean, vae)
        img_centroid_fix  = latent_to_image(centroid_fix, vae)

        # 4e) Save out:
        #     e.g. frog_centroid_mean_02.png and frog_centroid_fix_02.png   (if N=2)
        fname_mean = os.path.join(OUTPUT_DIR, MEAN_SUBFOLDER, f"frog_centroid_mean_{N:03d}.png")
        fname_fix  = os.path.join(OUTPUT_DIR, FIX_SUBFOLDER,  f"frog_centroid_fix_{N:03d}.png")
        img_centroid_mean.save(fname_mean)
        img_centroid_fix.save(fname_fix)

        print(f"• Channel‐wise‐mean centroid saved to: {fname_mean}")
        print(f"• Fixed‐norm centroid      saved to: {fname_fix}")

if __name__ == "__main__":
    main()
