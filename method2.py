import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt

# Import your existing modules
import attention_edit
import attention_utils

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INTERPOLATION_POINTS = 50
NUM_DIFFUSION_STEPS = 1000
GUIDANCE_SCALE = 20
REFINEMENT_STEPS = 1000  # Number of optimization steps for each point
LEARNING_RATE = 0.0003
DIFFUSION_LAMBDA = 1000.0  # Weight for diffusion loss
NORM_LAMBDA = 0.001      # Weight for norm regularization

def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two latent vectors.
    """
    # Flatten to 1D for computation
    z0_flat = z0.view(-1)
    z1_flat = z1.view(-1)
    
    # Compute cosine of angle between vectors
    denom = (z0_flat.norm() * z1_flat.norm()).clamp_min(1e-8)
    cos_omega = torch.dot(z0_flat, z1_flat) / denom
    cos_omega = cos_omega.clamp(-1 + 1e-7, 1 - 1e-7)
    
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega).clamp_min(1e-8)
    
    # Fall back to linear interpolation if vectors are nearly collinear
    if sin_omega < 1e-7:
        return (1 - t) * z0 + t * z1
    
    # SLERP formula
    factor0 = torch.sin((1 - t) * omega) / sin_omega
    factor1 = torch.sin(t * omega) / sin_omega
    
    z_flat = factor0 * z0_flat + factor1 * z1_flat
    return z_flat.view_as(z0)

def get_text_embedding(ldm_stable, prompts):
    """Generate text embeddings for given prompts."""
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(DEVICE))[0]
    return text_input, text_embeddings

def get_uncond_embedding(ldm_stable, batch_size):
    """Generate unconditional embeddings."""
    max_length = ldm_stable.tokenizer.model_max_length
    uncond_input = ldm_stable.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    return uncond_embeddings

def refine_latent_with_diffusion(ldm_stable, z_init, text_embeddings, uncond_embeddings, 
                                num_steps=300, lr=0.001):
    """
    Refine a latent vector using diffusion loss to ensure it stays on the learned manifold.
    """
    # Clone and enable gradients
    z_refined = z_init.clone().detach().requires_grad_(True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([z_refined], lr=lr)
    
    # Precompute target norm
    L = z_refined.numel()
    target_norm = torch.tensor(math.sqrt(L), device=DEVICE)
    
    # Combine embeddings for classifier-free guidance
    full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Sample random timestep
        t = torch.randint(0, ldm_stable.scheduler.config.num_train_timesteps, (1,)).long().to(DEVICE)
        
        # Add noise to current latent
        noise = torch.randn_like(z_refined)
        z_t = ldm_stable.scheduler.add_noise(z_refined, noise, t)
        
        # Predict noise with classifier-free guidance
        latent_model_input = torch.cat([z_t] * 2)
        with torch.no_grad():
            noise_pred_uncond, noise_pred_text = ldm_stable.unet(
                latent_model_input, t, encoder_hidden_states=full_text_embeddings
            ).sample.chunk(2)
        
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        
        # Compute diffusion loss
        loss_diffusion = F.mse_loss(noise_pred, noise)
        
        # Compute norm regularization loss
        latent_norm = torch.norm(z_refined)
        loss_norm = (latent_norm - target_norm).pow(2)
        
        # Total loss
        total_loss = DIFFUSION_LAMBDA * loss_diffusion + NORM_LAMBDA * loss_norm
        
        total_loss.backward()
        optimizer.step()
    
    return z_refined.detach()

def generate_smooth_trajectory(ldm_stable, z_start, z_end, prompt, output_dir):
    """
    Generate a smooth trajectory between two latent points using SLERP and diffusion refinement.
    """
    # Prepare text embeddings
    _, text_embeddings = get_text_embedding(ldm_stable, [prompt])
    uncond_embeddings = get_uncond_embedding(ldm_stable, batch_size=1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate interpolation points
    t_values = np.linspace(0, 1, NUM_INTERPOLATION_POINTS)
    refined_latents = []
    
    print("Generating and refining interpolation points...")
    for i, t in enumerate(tqdm(t_values, desc="Processing points")):
        # Step 1: SLERP interpolation
        z_interp = slerp(z_start, z_end, float(t))
        
        # Step 2: Refine with diffusion
        z_refined = refine_latent_with_diffusion(
            ldm_stable, 
            z_interp.unsqueeze(0),  # Add batch dimension
            text_embeddings, 
            uncond_embeddings,
            num_steps=REFINEMENT_STEPS,
            lr=LEARNING_RATE
        )
        
        refined_latents.append(z_refined)
        
        # Step 3: Generate and save image
        with torch.no_grad():
            images = attention_edit.latent2image(ldm_stable.vae, z_refined)
            if len(images)>0:
                # Save individual frame
                frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
                attention_utils.view_images(images[0], file_name=frame_path)
    
    # Save all refined latents for future use
    refined_latents_tensor = torch.cat(refined_latents, dim=0)
    torch.save(refined_latents_tensor, os.path.join(output_dir, "refined_trajectory_latents.pt"))
    
    # Generate a comparison grid showing original SLERP vs refined trajectory
    print("Generating comparison images...")
    original_images = []
    refined_images = []
    
    # Sample every 4th frame for comparison
    sample_indices = range(0, NUM_INTERPOLATION_POINTS, 4)
    
    for i in sample_indices:
        t = t_values[i]
        
        # Original SLERP interpolation
        z_orig = slerp(z_start, z_end, float(t))
        with torch.no_grad():
            orig_imgs = attention_edit.latent2image(ldm_stable.vae, z_orig.unsqueeze(0))
            if len(orig_imgs)>0:
                original_images.append(orig_imgs[0])
        
        # Refined version
        with torch.no_grad():
            refined_imgs = attention_edit.latent2image(ldm_stable.vae, refined_latents[i])
            if refined_imgs:
                refined_images.append(refined_imgs[0])
    
    # Save comparison grids
    if original_images:
        attention_utils.view_images(original_images, 
                                  file_name=os.path.join(output_dir, "original_slerp_comparison"))
    if refined_images:
        attention_utils.view_images(refined_images, 
                                  file_name=os.path.join(output_dir, "refined_trajectory_comparison"))
    
    return refined_latents_tensor

if __name__ == '__main__':
    # Load Stable Diffusion model
    print("Loading Stable Diffusion model...")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )
    ldm_stable = (
        StableDiffusionPipeline
        .from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_safetensors=True)
        .to(DEVICE)
    )
    
    # Load existing latents (from your reference code)
    prompt_lst = []
    latents_lst = []
    
    latent_files = natsorted(glob('./results/inversion/inverted_latent/*_final_latent_*.pt'))
    
    for latent_path in latent_files:
        basename = os.path.basename(latent_path)
        tokens = os.path.splitext(basename)[0].split('_')
        prompt_str = ' '.join(tokens[3:])
        prompt_lst.append(prompt_str)
        
        loaded = torch.load(latent_path, map_location=DEVICE)
        if loaded.ndim == 4 and loaded.shape[0] == 1:
            loaded = loaded.squeeze(0)
        latents_lst.append(loaded)
    
    # Select start and end points
    start_idx = 0
    end_idx = 3
    z_start = latents_lst[start_idx]
    z_end = latents_lst[end_idx]
    prompt = prompt_lst[start_idx]  # Use the prompt from start image
    
    print(f"Generating trajectory from latent {start_idx} to {end_idx}")
    print(f"Using prompt: '{prompt}'")
    
    # Generate smooth trajectory
    output_directory = f"./smooth_trajectory_results/{start_idx}_to_{end_idx}/"
    refined_trajectory = generate_smooth_trajectory(
        ldm_stable, z_start, z_end, prompt, output_directory
    )
    
    print(f"Trajectory generation complete! Results saved in: {output_directory}")
    print(f"Generated {NUM_INTERPOLATION_POINTS} refined frames")
