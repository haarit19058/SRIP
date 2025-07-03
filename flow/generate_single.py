import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_inversion import NullInversion
import attention_utils
from attention_utils import AttentionStore
import attention_edit

# -----------------------------
# ─── CONFIGURATION ───────────
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
scene_name = "optical_flow_26_27"
prompt = "an image of optical flow"

image_path = "./flow_1_to_2.png"
output_dir = f"./results/{scene_name}/inversion"
os.makedirs(f"{output_dir}/inv_image", exist_ok=True)
os.makedirs(f"{output_dir}/col_image", exist_ok=True)
os.makedirs(f"{output_dir}/attn_image", exist_ok=True)
os.makedirs(f"{output_dir}/inverted_latent", exist_ok=True)

GENERATOR = torch.manual_seed(8888)

# -----------------------------
# ─── LOAD STABLE DIFFUSION ───
# -----------------------------
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False, steps_offset=1
)

ldm_stable = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    use_safetensors=True
).to(DEVICE)

tokenizer = ldm_stable.tokenizer

# -----------------------------
# ─── INVERSION ───────────────
# -----------------------------
inversion_network = NullInversion(ldm_stable, GUIDANCE_SCALE, NUM_DIFFUSION_STEPS)

# Modify offset according to your image size
# offsets = (0, 0, 0, 0)  # left, top, right, bottom    for optical flow images
offsets = (0,0,200,0) # for fern images

(image_gt, image_enc), x_t, uncond_embeddings = inversion_network.invert(
    image_path, prompt, offsets=offsets, verbose=True
)

_, text_embeddings = inversion_network.context.chunk(2)

if uncond_embeddings is None:
    uncond_embeddings, _ = inversion_network.context.chunk(2)

# -----------------------------
# ─── GENERATE FROM INVERSION ─
# -----------------------------
controller = AttentionStore()
image_inv, x_t, image_latents = attention_edit.text2image_ldm_stable(
    ldm_stable,
    [prompt],
    controller,
    latent=x_t,
    num_inference_steps=NUM_DIFFUSION_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=GENERATOR,
    uncond_embeddings=uncond_embeddings,
    text_embeddings=text_embeddings,
    task_type='inversion'
)

file_id = prompt.replace(" ", "_")
torch.save(image_latents, f"{output_dir}/inverted_latent/final_latent_{file_id}.pt")
torch.save(torch.cat(uncond_embeddings, dim=0), f"{output_dir}/inverted_latent/uncond_latents_{file_id}.pt")

# -----------------------------
# ─── SAVE VISUALS ────────────
# -----------------------------
attention_utils.view_images(
    [image_gt, image_enc, image_inv[0]],
    file_name=f"{output_dir}/col_image/col_{file_id}"
)

attention_utils.view_images(
    image_inv,
    file_name=f"{output_dir}/inv_image/inv_{file_id}"
)

attention_utils.show_cross_attention(
    attention_store=controller,
    res=16,
    from_where=["up", "down"],
    tokenizer=tokenizer,
    prompts=[prompt],
    prompt_idx=0,
    file_name=f"{output_dir}/attn_image/attn_{file_id}"
)
