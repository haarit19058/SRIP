import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

latent_path1 = "./results/fern_4026/inversion/inverted_latent/final_latent_an_image_of_a_fern.pt"
latent_path2 = "./results/optical_flow_26_27/inversion/inverted_latent/final_latent_an_image_of_optical_flow.pt"

scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False, steps_offset=1
)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    use_safetensors=True,
).to(DEVICE)

lat1 = torch.load(latent_path1, map_location=DEVICE) 

lat2 = torch.load(latent_path2, map_location=DEVICE)
new_latent = lat1 + lat2

new_latents = new_latent / pipe.vae.config.scaling_facto

with torch.no_grad():
    images = pipe.vae.decode(new_latents).sample
    
for i, img in enumerate(images):
    img_to_show = (img + 1) / 2
    
    plt.imshow(img_to_show.permute(1, 2, 0).cpu())
    plt.axis('off')
    
    plt.savefig(f'./results/img_{i}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
