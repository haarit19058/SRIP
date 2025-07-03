import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms as tfms
import numpy as np
from tqdm.auto import tqdm

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pipeline with DDIM scheduler
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def load_and_encode_image(image_path, size=(512, 512)):
    """Load and encode image to latent space"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    image = image.resize(size)
    
    # Convert to tensor and normalize
    image_tensor = tfms.functional.to_tensor(image).unsqueeze(0).to(device) * 2 - 1
    
    # Encode to latent space
    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor)
        latent = 0.18215 * latent.latent_dist.sample()
    
    return latent, image

@torch.no_grad()
def ddim_inversion(start_latents, prompt, guidance_scale=7.5, num_inference_steps=50, 
                   negative_prompt=""):
    """Perform DDIM inversion to get noisy latents"""
    
    # Encode prompts
    text_embeddings = pipe._encode_prompt(
        prompt, device, 1, True, negative_prompt
    )
    
    latents = start_latents.clone()
    intermediate_latents = []
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)
    
    for i in tqdm(range(1, num_inference_steps), desc="DDIM Inversion"):
        if i >= num_inference_steps - 1:
            continue
            
        t = timesteps[i]
        
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Apply guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Inversion step
        current_t = max(0, t.item() - (1000 // num_inference_steps))
        next_t = t
        
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        
        # Inverted update step
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred
        
        intermediate_latents.append(latents.clone())
    
    return torch.stack(intermediate_latents)

def shift_latents_left(latents, shift_pixels=20, fill_method='noise'):
    """Shift latents left and fill empty space"""
    # Handle different tensor shapes
    if isinstance(latents, tuple):
        latents = latents[0]  # Take first element if it's a tuple
    
    # Ensure latents is 4D [batch, channels, height, width]
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)  # Add batch dimension
    elif latents.dim() == 5:
        latents = latents.squeeze(0)    # Remove extra dimension
    elif latents.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {latents.dim()}D tensor with shape {latents.shape}")
    
    batch_size, channels, height, width = latents.shape
    
    # Calculate shift in latent space (latents are 8x downsampled)
    latent_shift = shift_pixels // 8
    
    if latent_shift == 0:
        latent_shift = max(1, shift_pixels // 4)  # Ensure at least some shift
    
    # Create shifted latents
    shifted_latents = torch.zeros_like(latents)
    
    if latent_shift < width:
        # Shift existing content left
        shifted_latents[:, :, :, :-latent_shift] = latents[:, :, :, latent_shift:]
        
        # Fill the empty space on the right
        if fill_method == 'noise':
            # Fill with random noise matching the distribution
            noise_std = latents.std()
            shifted_latents[:, :, :, -latent_shift:] = torch.randn(
                batch_size, channels, height, latent_shift, 
                device=latents.device, dtype=latents.dtype
            ) * noise_std
            
        elif fill_method == 'edge':
            # Extend the leftmost pixels
            shifted_latents[:, :, :, -latent_shift:] = latents[:, :, :, :1].repeat(1, 1, 1, latent_shift)
            
        elif fill_method == 'mirror':
            # Mirror the leftmost part
            mirror_width = min(latent_shift, width - latent_shift)
            shifted_latents[:, :, :, -latent_shift:-latent_shift+mirror_width] = torch.flip(
                latents[:, :, :, :mirror_width], dims=[3]
            )
            if latent_shift > mirror_width:
                # Fill remaining with noise
                noise_std = latents.std()
                shifted_latents[:, :, :, -latent_shift+mirror_width:] = torch.randn(
                    batch_size, channels, height, latent_shift - mirror_width,
                    device=latents.device, dtype=latents.dtype
                ) * noise_std
    
    return shifted_latents

# Also update the prompt encoding function to handle the new format
@torch.no_grad()
def ddim_inversion(start_latents, prompt, guidance_scale=7.5, num_inference_steps=50, 
                   negative_prompt=""):
    """Perform DDIM inversion to get noisy latents"""
    
    # Use the new encode_prompt method instead of deprecated _encode_prompt
    try:
        # Try new format first
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
    except:
        # Fallback to old format
        text_embeddings = pipe._encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
    
    latents = start_latents.clone()
    intermediate_latents = []
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)
    
    for i in tqdm(range(1, num_inference_steps), desc="DDIM Inversion"):
        if i >= num_inference_steps - 1:
            continue
            
        t = timesteps[i]
        
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Apply guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Inversion step
        current_t = max(0, t.item() - (1000 // num_inference_steps))
        next_t = t
        
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        
        # Inverted update step
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred
        
        intermediate_latents.append(latents.clone())
    
    return torch.stack(intermediate_latents)

@torch.no_grad()
def generate_shifted_view(inverted_latents, prompt, shift_pixels=20, fill_method='noise',
                         guidance_scale=7.5, num_inference_steps=50, start_step=0,
                         negative_prompt=""):
    """Generate image with shifted view using inverted latents"""
    
    # Use the new encode_prompt method
    try:
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
    except:
        text_embeddings = pipe._encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps[start_step:]
    
    # Start from inverted latents
    latents = inverted_latents[-(start_step + 1)].unsqueeze(0)
    
    for i, t in enumerate(tqdm(timesteps, desc="Generating shifted view")):
        # Apply shift at each step
        latents = shift_latents_left(latents, shift_pixels, fill_method)
        
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Apply guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Denoising step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode to image
    with torch.no_grad():
        image = pipe.decode_latents(latents)
    
    return pipe.numpy_to_pil(image)[0]


@torch.no_grad()
def generate_shifted_view(inverted_latents, prompt, shift_pixels=20, fill_method='noise',
                         guidance_scale=7.5, num_inference_steps=50, start_step=0,
                         negative_prompt=""):
    """Generate image with shifted view using inverted latents"""
    
    # Encode prompts
    text_embeddings = pipe._encode_prompt(
        prompt, device, 1, True, negative_prompt
    )
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps[start_step:]
    
    # Start from inverted latents
    latents = inverted_latents[-(start_step + 1)].unsqueeze(0)
    
    for i, t in enumerate(tqdm(timesteps, desc="Generating shifted view")):
        # Apply shift at each step
        latents = shift_latents_left(latents, shift_pixels, fill_method)
        
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Apply guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Denoising step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode to image
    with torch.no_grad():
        image = pipe.decode_latents(latents)
    
    return pipe.numpy_to_pil(image)[0]

# Main execution function
def create_shifted_view_image(image_path, prompt, shift_pixels=20, fill_method='noise'):
    """Complete pipeline to create shifted view image"""
    
    print("1. Loading and encoding image...")
    start_latents, original_image = load_and_encode_image(image_path)
    
    print("2. Performing DDIM inversion...")
    inverted_latents = ddim_inversion(start_latents, prompt)
    
    print("3. Generating shifted view...")
    shifted_image = generate_shifted_view(
        inverted_latents, 
        prompt, 
        shift_pixels=shift_pixels,
        fill_method=fill_method,
        start_step=10  # Start from partially denoised state for better results
    )
    
    return original_image, shifted_image

# Example usage
if __name__ == "__main__":
    # Load your image
    image_path = "./flow/IMG_4026.JPG"  # Replace with your image path
    prompt = "A photo of fern"  # Describe your image
    
    # Create shifted view
    original, shifted = create_shifted_view_image(
        image_path, 
        prompt, 
        shift_pixels=1,  # Adjust shift amount
        fill_method='noise'  # Options: 'noise', 'edge', 'mirror'
    )
    
    # Save results
    original.save("original.jpg")
    shifted.save("shifted_view.jpg")
    
    print("Images saved!")
