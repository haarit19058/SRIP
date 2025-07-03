import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from attention_utils import AttentionStore, view_images
import attention_edit  # assumed from Prompt-to-Prompt repo
from PIL import Image

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

import random
seed = random.randint(0, 2**32 - 1)  # Generate a random seed
GENERATOR = torch.manual_seed(seed)
# GENERATOR = torch.manual_seed(8888)
SCENE_NAME = 'fern'
PROMPT = "a photo of arch linux"
SAVE_DIR = f'results/{SCENE_NAME}/generation'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the model
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)
ldm_stable = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    use_safetensors=True
).to(DEVICE)

# Text and unconditional embeddings
def get_text_embedding(ldm_stable, prompts):
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
    return text_input, text_embeddings

def get_uncond_embedding(ldm_stable, batch_size, max_length):
    uncond_input = ldm_stable.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(ldm_stable.device))[0]
    return uncond_embeddings





# Generate image
prompts = [PROMPT]
text_input, text_embeddings = get_text_embedding(ldm_stable, prompts)
uncond_embeddings = get_uncond_embedding(ldm_stable, batch_size=1, max_length=MAX_NUM_WORDS)
controller = AttentionStore()

image_list, latent_out, image_latents = attention_edit.text2image_ldm_stable(
    ldm_stable,
    prompts=prompts,
    controller=controller,
    latent=None,
    num_inference_steps=NUM_DIFFUSION_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=GENERATOR,
    uncond_embeddings=uncond_embeddings,
    text_embeddings=text_embeddings,
    task_type='generation'
)

# Display (optional)
view_images(image_list, file_name=os.path.join(SAVE_DIR, 'fern_display'))

# Save image
image_pil = image_list[0]
image_path = os.path.join(SAVE_DIR, 'fern_generated_image.png')
# image_pil.save(image_path)
import matplotlib.pyplot as plt
plt.imshow(image_pil)
plt.savefig(image_path)
print(f"[✔] Image saved at: {image_path}")

# Save latent
latent_path = os.path.join(SAVE_DIR, 'fern_latent.pt')
torch.save(image_latents, latent_path)
print(f"[✔] Latent saved at: {latent_path}")
