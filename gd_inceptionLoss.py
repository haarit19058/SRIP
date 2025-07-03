# import os
# import math
# import torch
# import torch.nn.functional as F
# import numpy as np
# from glob import glob
# from natsort import natsorted
# from tqdm import tqdm
# from diffusers import StableDiffusionPipeline, DDIMScheduler
# from transformers import CLIPProcessor, CLIPModel
# import attention_edit
# import attention_utils
# import matplotlib.pyplot as plt

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "CompVis/stable-diffusion-v1-4"
# CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# NUM_OPTIMIZATION_STEPS = 500
# LEARNING_RATE = 0.007
# SIMILARITY_LAMBDA = 10
# NORM_LAMBDA = 0.0001
# GUIDANCE_SCALE = 100
# NUM_DIFFUSION_STEPS = 50
# CLIP_LAMBDA = 1

# LATENTS_DIR = './results/inversion/inverted_latent/'
# OUTPUT_DIR = f'./gradient_path_results_2/'

# np.random.seed(8888)
# torch.manual_seed(8888)
# torch.cuda.manual_seed_all(8888)

# clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
# clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


# def differentiable_decode_latent(vae, latent):
#     """Decode latent to image tensor while preserving gradients"""
#     # Ensure latent is in the right format
#     if latent.dim() == 3:
#         latent = latent.unsqueeze(0)
    
#     # Scale latent
#     latent = latent / vae.config.scaling_factor
    
#     # Decode using VAE decoder
#     with torch.enable_grad():
#         image = vae.decode(latent).sample
    
#     # Convert from [-1, 1] to [0, 1] range
#     image = (image + 1.0) / 2.0
#     image = torch.clamp(image, 0.0, 1.0)
    
#     return image


# class DifferentiableCLIPProcessor:
#     """CLIP processor that preserves gradients"""
#     def __init__(self, device="cuda"):
#         self.device = device
#         self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
#         self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        
#     def preprocess_image(self, images):
#         """Preprocess images while preserving gradients"""
#         # Ensure images are in [0, 1] range
#         if images.max() > 1.0:
#             images = images / 255.0
            
#         # Resize to 224x224
#         images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # Normalize
#         images = (images - self.image_mean) / self.image_std
        
#         return images


# def clip_image_loss_differentiable(latent, vae, target_emb, alpha=1.0, beta=10):
#     """CLIP loss that preserves gradients through the entire pipeline"""
#     # Decode latent to image tensor (preserves gradients)
#     current_img_tensor = differentiable_decode_latent(vae, latent)
    
#     # Resize to 224x224 (CLIP input size)
#     current_img_resized = F.interpolate(
#         current_img_tensor, 
#         size=(224, 224), 
#         mode='bilinear', 
#         align_corners=False
#     )
    
#     # Apply CLIP normalization
#     mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(latent.device)
#     std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(latent.device)
#     current_img_normalized = (current_img_resized - mean) / std
    
#     # Get CLIP features
#     img_emb = clip_model.vision_model(current_img_normalized).last_hidden_state[:, 0, :]
#     img_emb = clip_model.visual_projection(img_emb)
    
#     # Normalize for directional similarity
#     img_emb_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
#     target_emb_norm = target_emb / target_emb.norm(dim=-1, keepdim=True)
    
#     # Directional loss
#     directional_loss = -(img_emb_norm * target_emb_norm).sum(dim=-1).mean()
    
#     # Magnitude difference loss
#     img_magnitude = img_emb.norm(dim=-1)
#     target_magnitude = target_emb.norm(dim=-1)
#     magnitude_loss = F.mse_loss(img_magnitude, target_magnitude)
    
#     return alpha * directional_loss + beta * magnitude_loss


# def get_text_embedding(pipe, prompts):
#     ti = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
#     emb = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]
#     return ti, emb


# def get_uncond_embedding(pipe, batch_size):
#     tokens = [""] * batch_size
#     ui = pipe.tokenizer(tokens, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
#     return pipe.text_encoder(ui.input_ids.to(DEVICE))[0]


# def decode_and_save_image(pipe, latent, path):
#     with torch.no_grad():
#         imgs = attention_edit.latent2image(pipe.vae, latent)
#         if len(imgs) > 0:
#             img = imgs[0]
#             plt.imshow(img)
#             plt.savefig(path)
#             plt.close()  # Close the figure to free memory


# if __name__ == '__main__':
#     scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
#     pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=scheduler, use_safetensors=True).to(DEVICE)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     latent_files = natsorted(glob(os.path.join(LATENTS_DIR, '*_final_latent_*.pt')))
#     if not latent_files:
#         raise FileNotFoundError(f"No latents in {LATENTS_DIR}")

#     prompts, latents = [], []
#     for path in latent_files:
#         name = os.path.basename(path)
#         tokens = os.path.splitext(name)[0].split('_')[3:]
#         prompts.append(' '.join(tokens))
#         data = torch.load(path, map_location=DEVICE)
#         if data.ndim == 4 and data.shape[0] == 1:
#             data = data.squeeze(0)
#         latents.append(data)

#     # Get target embedding using the original CLIP processor for initialization
#     last_img = attention_edit.latent2image(pipe.vae, latents[-1].unsqueeze(0).to(DEVICE))[0]
#     inp = clip_processor(images=last_img, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         stim_emb = clip_model.get_image_features(**inp)
#         stim_emb = stim_emb / stim_emb.norm(dim=-1, keepdim=True)

#     z_start = latents[0].unsqueeze(0).to(DEVICE)
#     z_target = latents[3].unsqueeze(0).to(DEVICE)
#     decode_and_save_image(pipe, z_start.detach(), f'{OUTPUT_DIR}/start.png')
#     decode_and_save_image(pipe, z_target.detach(), f'{OUTPUT_DIR}/target.png')
#     prompt = prompts[0]

#     _, text_emb = get_text_embedding(pipe, [prompt])
#     uncond_emb = get_uncond_embedding(pipe, 1)
#     full_emb = torch.cat([uncond_emb, text_emb])

#     z_current = z_start.clone().detach().requires_grad_(True)
#     L = z_current.numel()
#     target_norm = torch.tensor(math.sqrt(L), device=DEVICE)
#     optimizer = torch.optim.Adam([z_current], lr=LEARNING_RATE)

#     pipe.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)

#     print(f"Starting optimization with {NUM_OPTIMIZATION_STEPS} steps...")
#     print(f"CLIP lambda: {CLIP_LAMBDA}, Similarity lambda: {SIMILARITY_LAMBDA}")

#     for step in tqdm(range(NUM_OPTIMIZATION_STEPS), desc="Optimizing"):  
#         optimizer.zero_grad()  
        
#         # Sample random timestep
#         t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=DEVICE).long()  
#         noise = torch.randn_like(z_current)
#         z_t = pipe.scheduler.add_noise(z_current, noise, t)

#         # Get noise prediction
#         inp_latent = torch.cat([z_t] * 2)
#         with torch.no_grad():
#             pred = pipe.unet(inp_latent, t, encoder_hidden_states=full_emb).sample.chunk(2)
#             noise_pred = pred[0] + GUIDANCE_SCALE * (pred[1] - pred[0])

#         # Compute losses
#         loss = F.mse_loss(noise_pred, noise)
#         loss += SIMILARITY_LAMBDA * F.mse_loss(z_current, z_target)
#         loss += NORM_LAMBDA * (torch.norm(z_current) - target_norm).pow(2)

#         # Add differentiable CLIP loss
#         clip_loss = clip_image_loss_differentiable(z_current, pipe.vae, stim_emb)
#         loss += CLIP_LAMBDA * clip_loss

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Log progress
#         if step % 100 == 0:
#             decode_and_save_image(pipe, z_current.detach(), os.path.join(OUTPUT_DIR, f"step_{step:04d}.png"))
#             print(f"Step {step}: Total loss = {loss.item():.6f}, CLIP loss = {clip_loss.item():.6f}")

#     # Save final result
#     decode_and_save_image(pipe, z_current.detach(), os.path.join(OUTPUT_DIR, f"step_{NUM_OPTIMIZATION_STEPS:04d}_final.png"))
    
#     # Save final latent
#     torch.save(z_current.detach(), os.path.join(OUTPUT_DIR, "final_optimized_latent.pt"))
    
#     print(f"Done. Saved images and final latent in {OUTPUT_DIR}")


















































# import os
# import math
# import torch
# import torch.nn.functional as F
# import numpy as np
# from glob import glob
# from natsort import natsorted
# from tqdm import tqdm
# from diffusers import StableDiffusionPipeline, DDIMScheduler
# from transformers import CLIPProcessor, CLIPModel
# from torchvision.models import inception_v3
# import torchvision.transforms as transforms
# import attention_edit
# import attention_utils
# import matplotlib.pyplot as plt

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "CompVis/stable-diffusion-v1-4"
# CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# NUM_OPTIMIZATION_STEPS = 500
# LEARNING_RATE = 0.007
# SIMILARITY_LAMBDA = 1
# NORM_LAMBDA = 0.0001
# GUIDANCE_SCALE = 10
# NUM_DIFFUSION_STEPS = 50
# INCEPTION_LAMBDA = 10000000.0  # Weight for inception loss

# LATENTS_DIR = './results/inversion/inverted_latent/'
# OUTPUT_DIR = f'./gradient_path_results_1/'

# np.random.seed(8888)
# torch.manual_seed(8888)
# torch.cuda.manual_seed_all(8888)

# clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
# clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


# def differentiable_decode_latent(vae, latent):
#     """Decode latent to image tensor while preserving gradients"""
#     # Ensure latent is in the right format
#     if latent.dim() == 3:
#         latent = latent.unsqueeze(0)
    
#     # Scale latent
#     latent = latent / vae.config.scaling_factor
    
#     # Decode using VAE decoder
#     with torch.enable_grad():
#         image = vae.decode(latent).sample
    
#     # Convert from [-1, 1] to [0, 1] range
#     image = (image + 1.0) / 2.0
#     image = torch.clamp(image, 0.0, 1.0)
    
#     return image


# def preprocess_for_inception(img_tensor):
#     """Preprocess images for inception model while preserving gradients"""
#     # img_tensor shape: (B, C, H, W), values in [0,1]
#     img_resized = F.interpolate(img_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
#     # Apply ImageNet normalization
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
#     img_normalized = (img_resized - mean) / std
    
#     return img_normalized


# class InceptionFeatureExtractor(torch.nn.Module):
#     """Inception feature extractor for perceptual loss"""
#     def __init__(self, device):
#         super().__init__()
#         self.inception = inception_v3(pretrained=True, transform_input=False).to(device).eval()
#         # Remove the final classification layer
#         self.inception.fc = torch.nn.Identity()
        
#         # Freeze all parameters
#         for param in self.inception.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         return self.inception(x)


# def inception_loss(z_current, z_target, vae, inception_extractor):
#     """Compute inception loss between current and target latents"""
#     # Decode latents to images (preserving gradients for z_current)
#     img_current = differentiable_decode_latent(vae, z_current)
    
#     # Decode target latent without gradients (since we don't optimize z_target)
#     with torch.no_grad():
#         img_target = differentiable_decode_latent(vae, z_target)
    
#     # Preprocess for inception
#     img_current_prep = preprocess_for_inception(img_current)
#     img_target_prep = preprocess_for_inception(img_target)
    
#     # Get inception features
#     feat_current = inception_extractor(img_current_prep)
    
#     # Get target features without gradients
#     with torch.no_grad():
#         feat_target = inception_extractor(img_target_prep)
    
#     # Compute MSE loss between features
#     loss = F.mse_loss(feat_current, feat_target)
#     return loss


# def get_text_embedding(pipe, prompts):
#     ti = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
#     emb = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]
#     return ti, emb


# def get_uncond_embedding(pipe, batch_size):
#     tokens = [""] * batch_size
#     ui = pipe.tokenizer(tokens, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
#     return pipe.text_encoder(ui.input_ids.to(DEVICE))[0]


# def decode_and_save_image(pipe, latent, path):
#     with torch.no_grad():
#         imgs = attention_edit.latent2image(pipe.vae, latent)
#         if len(imgs) > 0:
#             img = imgs[0]
#             plt.imshow(img)
#             plt.savefig(path)
#             plt.close()  # Close the figure to free memory


# if __name__ == '__main__':
#     scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
#     pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=scheduler, use_safetensors=True).to(DEVICE)

#     # Initialize inception feature extractor
#     inception_extractor = InceptionFeatureExtractor(DEVICE)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     latent_files = natsorted(glob(os.path.join(LATENTS_DIR, '*_final_latent_*.pt')))
#     if not latent_files:
#         raise FileNotFoundError(f"No latents in {LATENTS_DIR}")

#     prompts, latents = [], []
#     for path in latent_files:
#         name = os.path.basename(path)
#         tokens = os.path.splitext(name)[0].split('_')[3:]
#         prompts.append(' '.join(tokens))
#         data = torch.load(path, map_location=DEVICE)
#         if data.ndim == 4 and data.shape[0] == 1:
#             data = data.squeeze(0)
#         latents.append(data)

#     z_start = latents[0].unsqueeze(0).to(DEVICE)
#     z_target = latents[3].unsqueeze(0).to(DEVICE)
#     decode_and_save_image(pipe, z_start.detach(), f'{OUTPUT_DIR}/start.png')
#     decode_and_save_image(pipe, z_target.detach(), f'{OUTPUT_DIR}/target.png')
#     prompt = prompts[0]

#     _, text_emb = get_text_embedding(pipe, [prompt])
#     uncond_emb = get_uncond_embedding(pipe, 1)
#     full_emb = torch.cat([uncond_emb, text_emb])

#     z_current = z_start.clone().detach().requires_grad_(True)
#     L = z_current.numel()
#     target_norm = torch.tensor(math.sqrt(L), device=DEVICE)
#     optimizer = torch.optim.Adam([z_current], lr=LEARNING_RATE)

#     pipe.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)

#     print(f"Starting optimization with {NUM_OPTIMIZATION_STEPS} steps...")
#     print(f"Inception lambda: {INCEPTION_LAMBDA}, Similarity lambda: {SIMILARITY_LAMBDA}")

#     for step in tqdm(range(NUM_OPTIMIZATION_STEPS), desc="Optimizing"):  
#         optimizer.zero_grad()  
        
#         # Sample random timestep
#         t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=DEVICE).long()  
#         noise = torch.randn_like(z_current)
#         z_t = pipe.scheduler.add_noise(z_current, noise, t)

#         # Get noise prediction
#         inp_latent = torch.cat([z_t] * 2)
#         with torch.no_grad():
#             pred = pipe.unet(inp_latent, t, encoder_hidden_states=full_emb).sample.chunk(2)
#             noise_pred = pred[0] + GUIDANCE_SCALE * (pred[1] - pred[0])

#         # Compute losses
#         loss = F.mse_loss(noise_pred, noise)
#         loss += SIMILARITY_LAMBDA * F.mse_loss(z_current, z_target)
#         loss += NORM_LAMBDA * (torch.norm(z_current) - target_norm).pow(2)

#         # Add inception loss
#         inception_loss_value = inception_loss(z_current, z_target, pipe.vae, inception_extractor)
#         loss += INCEPTION_LAMBDA * inception_loss_value

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Log progress
#         if step % 10 == 0:
#             decode_and_save_image(pipe, z_current.detach(), os.path.join(OUTPUT_DIR, f"step_{step:04d}.png"))
#             print(f"Step {step}: Total loss = {loss.item():.6f}, Inception loss = {inception_loss_value.item():.6f}")

#     # Save final result
#     decode_and_save_image(pipe, z_current.detach(), os.path.join(OUTPUT_DIR, f"step_{NUM_OPTIMIZATION_STEPS:04d}_final.png"))
    
#     # Save final latent
#     torch.save(z_current.detach(), os.path.join(OUTPUT_DIR, "final_optimized_latent.pt"))
    
#     print(f"Done. Saved images and final latent in {OUTPUT_DIR}")
