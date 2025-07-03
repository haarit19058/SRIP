import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
import attention_edit
import attention_utils
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

NUM_OPTIMIZATION_STEPS = 500
LEARNING_RATE = 0.01
SIMILARITY_LAMBDA = 1
NORM_LAMBDA = 0.01
GUIDANCE_SCALE = 7.5
NUM_DIFFUSION_STEPS = 50
CLIP_LAMBDA = 1000

scene_name = 'leaves'
LATENTS_DIR = f'./results/{scene_name}/inversion/inverted_latent/'
OUTPUT_DIR = f'./gradient_path_results_{scene_name}_clip/'

np.random.seed(8888)
torch.manual_seed(8888)
torch.cuda.manual_seed_all(8888)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


# def clip_image_loss(current_pil, target_emb):
#     inputs = clip_processor(images=current_pil, return_tensors="pt").to(DEVICE)
#     img_emb = clip_model.get_image_features(**inputs)
#     img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
#     return - (img_emb * target_emb).sum(dim=-1).mean()


def clip_image_loss(current_pil, target_emb, alpha=1.0, beta=0):
    inputs = clip_processor(images=current_pil, return_tensors="pt").to(DEVICE)
    img_emb = clip_model.get_image_features(**inputs)
    
    # Normalize for directional similarity (cosine similarity)
    img_emb_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
    target_emb_norm = target_emb / target_emb.norm(dim=-1, keepdim=True)
    
    # Directional loss (existing)
    directional_loss = -(img_emb_norm * target_emb_norm).sum(dim=-1).mean()
    
    # Magnitude difference loss (new)
    img_magnitude = img_emb.norm(dim=-1)
    target_magnitude = target_emb.norm(dim=-1)
    magnitude_loss = F.mse_loss(img_magnitude, target_magnitude)
    
    return alpha * directional_loss + beta * magnitude_loss



def get_text_embedding(pipe, prompts):
    ti = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    emb = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]
    return ti, emb


def get_uncond_embedding(pipe, batch_size):
    tokens = [""] * batch_size
    ui = pipe.tokenizer(tokens, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    return pipe.text_encoder(ui.input_ids.to(DEVICE))[0]


def decode_and_save_image(pipe, latent, path):
    with torch.no_grad():
        imgs = attention_edit.latent2image(pipe.vae, latent)
        if len(imgs)>0:
            img = imgs[0]
            # from PIL import Image
            plt.imshow(img)
            plt.savefig(path)
            # img.save(path)


if __name__ == '__main__':
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=scheduler, use_safetensors=True).to(DEVICE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latent_files = natsorted(glob(os.path.join(LATENTS_DIR, '*_final_latent_*.pt')))
    if not latent_files:
        raise FileNotFoundError(f"No latents in {LATENTS_DIR}")

    prompts, latents = [], []
    for path in latent_files:
        name = os.path.basename(path)
        tokens = os.path.splitext(name)[0].split('_')[3:]
        prompts.append(' '.join(tokens))
        data = torch.load(path, map_location=DEVICE)
        if data.ndim == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        latents.append(data)

    last_img = attention_edit.latent2image(pipe.vae, latents[-1].unsqueeze(0).to(DEVICE))[0]
    inp = clip_processor(images=last_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        stim_emb = clip_model.get_image_features(**inp)
        stim_emb = stim_emb / stim_emb.norm(dim=-1, keepdim=True)

    z_start = latents[0].unsqueeze(0).to(DEVICE)
    z_target = latents[3].unsqueeze(0).to(DEVICE)
    decode_and_save_image(pipe, z_start.detach(), f'{OUTPUT_DIR}/start.png')
    decode_and_save_image(pipe, z_target.detach(), f'{OUTPUT_DIR}/target.png')
    prompt = prompts[0]

    _, text_emb = get_text_embedding(pipe, [prompt])
    uncond_emb = get_uncond_embedding(pipe, 1)
    full_emb = torch.cat([uncond_emb, text_emb])

    z_current = z_start.clone().detach().requires_grad_(True)
    L = z_current.numel()
    # target_norm = torch.tensor(math.sqrt(z_target), device=DEVICE)
    target_norm = torch.norm(z_target).to(DEVICE)
    optimizer = torch.optim.Adam([z_current], lr=LEARNING_RATE)

    pipe.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)

    for step in tqdm(range(NUM_OPTIMIZATION_STEPS), desc="Optimizing"):  
        optimizer.zero_grad()  
        t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=DEVICE).long()  
        noise = torch.randn_like(z_current)
        z_t = pipe.scheduler.add_noise(z_current, noise, t)

        inp_latent = torch.cat([z_t] * 2)
        with torch.no_grad():
            pred = pipe.unet(inp_latent, t, encoder_hidden_states=full_emb).sample.chunk(2)
            noise_pred = pred[0] + GUIDANCE_SCALE * (pred[1] - pred[0])
        # z_current = (z_t - noise_pred)

        loss = F.mse_loss(noise_pred, noise)
        loss += SIMILARITY_LAMBDA * F.mse_loss(z_current, z_target)
        loss += NORM_LAMBDA * (torch.norm(z_current) - target_norm).pow(2)

        current_img = attention_edit.latent2image(pipe.vae, z_current)[0]
        loss += CLIP_LAMBDA * clip_image_loss(current_img, stim_emb)

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            decode_and_save_image(pipe, z_current.detach(), os.path.join(OUTPUT_DIR, f"step_{step:04d}.png"))

    decode_and_save_image(pipe, z_current.detach(), os.path.join(OUTPUT_DIR, f"step_{NUM_OPTIMIZATION_STEPS:04d}_final.png"))
    print(f"Done. Saved images in {OUTPUT_DIR}")
