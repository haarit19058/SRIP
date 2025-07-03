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

def shift_image_pixels(image, shift_pixels=50, fill_method='edge'):
    """Shift image left by specified pixels and fill empty space"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    shifted_image = np.zeros_like(image)
    
    # Ensure shift_pixels doesn't exceed image width
    shift_pixels = min(shift_pixels, width - 1)
    
    if shift_pixels < width and shift_pixels > 0:
        # Shift existing content left
        shifted_image[:, :-shift_pixels] = image[:, shift_pixels:]
        
        # Fill the empty space on the right
        if fill_method == 'edge':
            # Extend the leftmost pixels
            shifted_image[:, -shift_pixels:] = image[:, :1]
            
        elif fill_method == 'mirror':
            # Calculate safe mirror width
            available_source = width - shift_pixels  # Available pixels to mirror from
            mirror_width = min(shift_pixels, available_source)
            
            if mirror_width > 0:
                # Mirror the leftmost part of the shifted content
                source_start = 0
                source_end = mirror_width
                target_start = -shift_pixels
                target_end = -shift_pixels + mirror_width
                
                shifted_image[:, target_start:target_end] = np.flip(
                    image[:, source_start:source_end], axis=1
                )
                
                # Fill any remaining space with edge pixels
                if mirror_width < shift_pixels:
                    remaining_pixels = shift_pixels - mirror_width
                    shifted_image[:, -remaining_pixels:] = image[:, :1]
            else:
                # Fallback to edge fill if no pixels available for mirroring
                shifted_image[:, -shift_pixels:] = image[:, :1]
                
        elif fill_method == 'black':
            # Fill with black (already zeros from initialization)
            pass
            
        elif fill_method == 'wrap':
            # Wrap around from the right side
            wrap_pixels = min(shift_pixels, width - shift_pixels)
            if wrap_pixels > 0:
                shifted_image[:, -shift_pixels:-shift_pixels+wrap_pixels] = image[:, -wrap_pixels:]
            # Fill remaining with edge if needed
            if wrap_pixels < shift_pixels:
                remaining = shift_pixels - wrap_pixels
                shifted_image[:, -remaining:] = image[:, :1]
    else:
        # If shift is too large, fill entire image with edge pixels
        shifted_image[:, :] = image[:, :1]
    
    return Image.fromarray(shifted_image.astype(np.uint8))

# Alternative safer implementation with explicit bounds checking
def shift_image_pixels_safe(image, shift_pixels=50, fill_method='edge'):
    """Safer version with explicit bounds checking"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    # Clamp shift_pixels to valid range
    shift_pixels = max(0, min(shift_pixels, width))
    
    # Create output array
    if len(image.shape) == 3:
        shifted_image = np.zeros((height, width, channels), dtype=image.dtype)
    else:
        shifted_image = np.zeros((height, width), dtype=image.dtype)
    
    if shift_pixels == 0:
        return Image.fromarray(image)
    
    if shift_pixels >= width:
        # Entire image is shifted out, fill with edge
        if len(image.shape) == 3:
            shifted_image[:, :] = image[:, 0:1]
        else:
            shifted_image[:, :] = image[:, 0:1]
        return Image.fromarray(shifted_image)
    
    # Copy shifted content
    shifted_image[:, :-shift_pixels] = image[:, shift_pixels:]
    
    # Fill the empty space based on method
    if fill_method == 'edge':
        shifted_image[:, -shift_pixels:] = image[:, 0:1]
        
    elif fill_method == 'mirror':
        # Calculate how much we can mirror
        available_to_mirror = width - shift_pixels
        mirror_amount = min(shift_pixels, available_to_mirror)
        
        if mirror_amount > 0:
            # Mirror from the beginning of the original image
            mirror_source = image[:, :mirror_amount]
            mirror_flipped = np.flip(mirror_source, axis=1)
            shifted_image[:, -shift_pixels:-shift_pixels+mirror_amount] = mirror_flipped
            
            # Fill any remaining space with edge
            if mirror_amount < shift_pixels:
                remaining = shift_pixels - mirror_amount
                shifted_image[:, -remaining:] = image[:, 0:1]
        else:
            # No space to mirror, use edge fill
            shifted_image[:, -shift_pixels:] = image[:, 0:1]
            
    elif fill_method == 'black':
        # Already filled with zeros
        pass
        
    elif fill_method == 'wrap':
        # Wrap from the end of the image
        wrap_amount = min(shift_pixels, width - shift_pixels)
        if wrap_amount > 0:
            shifted_image[:, -shift_pixels:-shift_pixels+wrap_amount] = image[:, -wrap_amount:]
            if wrap_amount < shift_pixels:
                remaining = shift_pixels - wrap_amount
                shifted_image[:, -remaining:] = image[:, 0:1]
        else:
            shifted_image[:, -shift_pixels:] = image[:, 0:1]
    
    return Image.fromarray(shifted_image)

# Updated main processing function with error handling
def process_with_advanced_options_safe(image_path, prompt, shift_pixels=50, 
                                     noise_strength=0.1, fill_method='edge',
                                     guidance_scale=7.5, num_inference_steps=50,
                                     start_step=5, negative_prompt=""):
    """Advanced version with error handling"""
    
    print("0. Loading original image...")
    if isinstance(image_path, str):
        original_image = Image.open(image_path).convert("RGB")
    else:
        original_image = image_path
    
    # Validate shift_pixels against image dimensions
    img_width = original_image.size[0]
    if shift_pixels >= img_width:
        print(f"Warning: shift_pixels ({shift_pixels}) >= image width ({img_width}). Reducing to {img_width//2}")
        shift_pixels = img_width // 2
    
    print(f"1. Shifting image left by {shift_pixels} pixels...")
    try:
        shifted_image = shift_image_pixels_safe(original_image, shift_pixels, fill_method)
    except Exception as e:
        print(f"Error in shifting: {e}")
        print("Falling back to edge fill method...")
        shifted_image = shift_image_pixels_safe(original_image, shift_pixels, 'edge')
    
    print("2. Encoding shifted image to latent space...")
    start_latents, _ = load_and_encode_image(shifted_image)
    
    print("3. Performing DDIM inversion with noise injection...")
    inverted_latents = ddim_inversion_with_noise_injection(
        start_latents, 
        prompt, 
        shift_pixels=shift_pixels,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        noise_strength=noise_strength
    )
    
    print("4. Regenerating image from latents...")
    regenerated_image = regenerate_from_latents(
        inverted_latents, 
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        start_step=start_step,
        negative_prompt=negative_prompt
    )
    
    return original_image, shifted_image, regenerated_image


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

def get_shifted_mask(latents_shape, shift_pixels, device):
    """Create a mask for the shifted region in latent space"""
    batch_size, channels, height, width = latents_shape
    
    # Calculate shift in latent space (latents are 8x downsampled)
    latent_shift = max(1, shift_pixels // 8)
    
    # Create mask: 1 for shifted region, 0 for original region
    mask = torch.zeros((batch_size, 1, height, width), device=device)
    if latent_shift < width:
        mask[:, :, :, -latent_shift:] = 1.0
    
    return mask

@torch.no_grad()
def ddim_inversion_with_noise_injection(start_latents, prompt, shift_pixels=50, 
                                      guidance_scale=7.5, num_inference_steps=50, 
                                      negative_prompt="", noise_strength=0.1):
    """Perform DDIM inversion with Gaussian noise added to shifted region"""
    
    # Encode prompts
    try:
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
    except:
        text_embeddings = pipe._encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
    
    latents = start_latents.clone()
    intermediate_latents = []
    
    # Get mask for shifted region
    shift_mask = get_shifted_mask(latents.shape, shift_pixels, device)
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)
    
    for i in tqdm(range(1, num_inference_steps), desc="DDIM Inversion with Noise Injection"):
        if i >= num_inference_steps - 1:
            continue
            
        t = timesteps[i]
        
        # Add Gaussian noise to shifted region before processing
        noise_scale = noise_strength * (t.item() / 1000.0)  # Scale noise with timestep
        gaussian_noise = torch.randn_like(latents) * noise_scale
        latents = latents + gaussian_noise * shift_mask
        
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
def regenerate_from_latents(inverted_latents, prompt, guidance_scale=7.5, 
                           num_inference_steps=50, start_step=0, negative_prompt=""):
    """Regenerate image from inverted latents"""
    
    # Encode prompts
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
    
    # Fix: Properly handle the latents dimensions
    latents = inverted_latents[-(start_step + 1)]
    
    # Ensure latents is 4D [batch, channels, height, width]
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)  # Add batch dimension
    elif latents.dim() == 5:
        # If 5D, remove the extra dimension
        latents = latents.squeeze(1) if latents.shape[1] == 1 else latents.squeeze(0)
    elif latents.dim() != 4:
        raise ValueError(f"Expected 4D latents, got {latents.dim()}D with shape {latents.shape}")
    
    print(f"Latents shape for regeneration: {latents.shape}")
    
    for i, t in enumerate(tqdm(timesteps, desc="Regenerating image")):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        print(f"Step {i}: latent_model_input shape: {latent_model_input.shape}")
        
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

# Also fix the DDIM inversion function to ensure consistent tensor shapes
@torch.no_grad()
def ddim_inversion_with_noise_injection(start_latents, prompt, shift_pixels=50, 
                                      guidance_scale=7.5, num_inference_steps=50, 
                                      negative_prompt="", noise_strength=0.1):
    """Perform DDIM inversion with Gaussian noise added to shifted region"""
    
    # Encode prompts
    try:
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
    except:
        text_embeddings = pipe._encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
    
    # Ensure start_latents is 4D
    latents = start_latents.clone()
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    elif latents.dim() == 5:
        latents = latents.squeeze(1) if latents.shape[1] == 1 else latents.squeeze(0)
    
    print(f"Initial latents shape: {latents.shape}")
    intermediate_latents = []
    
    # Get mask for shifted region
    shift_mask = get_shifted_mask(latents.shape, shift_pixels, device)
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)
    
    for i in tqdm(range(1, num_inference_steps), desc="DDIM Inversion with Noise Injection"):
        if i >= num_inference_steps - 1:
            continue
            
        t = timesteps[i]
        
        # Add Gaussian noise to shifted region before processing
        noise_scale = noise_strength * (t.item() / 1000.0)
        gaussian_noise = torch.randn_like(latents) * noise_scale
        latents = latents + gaussian_noise * shift_mask
        
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
        
        # Store latents ensuring consistent shape
        intermediate_latents.append(latents.clone())
    
    # Stack latents properly
    stacked_latents = torch.stack(intermediate_latents)
    print(f"Stacked latents shape: {stacked_latents.shape}")
    
    return stacked_latents

# Add a debug function to check tensor shapes
def debug_tensor_shapes(tensor, name="tensor"):
    """Debug function to print tensor information"""
    if isinstance(tensor, (list, tuple)):
        print(f"{name} is {type(tensor).__name__} with {len(tensor)} elements:")
        for i, t in enumerate(tensor):
            if hasattr(t, 'shape'):
                print(f"  Element {i}: {t.shape}")
            else:
                print(f"  Element {i}: {type(t)}")
    elif hasattr(tensor, 'shape'):
        print(f"{name} shape: {tensor.shape}")
    else:
        print(f"{name} type: {type(tensor)}")


def process_image_with_shift_and_inversion(image_path, prompt, shift_pixels=50, 
                                         noise_strength=0.1, fill_method='edge'):
    """Complete pipeline: shift image, DDIM inversion with noise, regenerate"""
    
    print("0. Loading original image...")
    if isinstance(image_path, str):
        original_image = Image.open(image_path).convert("RGB")
    else:
        original_image = image_path
    
    print(f"1. Shifting image left by {shift_pixels} pixels...")
    shifted_image = shift_image_pixels(original_image, shift_pixels, fill_method)
    
    print("2. Encoding shifted image to latent space...")
    start_latents, _ = load_and_encode_image(shifted_image)
    
    print("3. Performing DDIM inversion with noise injection...")
    inverted_latents = ddim_inversion_with_noise_injection(
        start_latents, 
        prompt, 
        shift_pixels=shift_pixels,
        noise_strength=noise_strength
    )
    
    print("4. Regenerating image from latents...")
    regenerated_image = regenerate_from_latents(
        inverted_latents, 
        prompt,
        start_step=5  # Start from partially denoised state
    )
    
    return original_image, shifted_image, regenerated_image

# Advanced version with more control
def process_with_advanced_options(image_path, prompt, shift_pixels=50, 
                                noise_strength=0.1, fill_method='edge',
                                guidance_scale=7.5, num_inference_steps=50,
                                start_step=5, negative_prompt=""):
    """Advanced version with more control parameters"""
    
    print("0. Loading original image...")
    if isinstance(image_path, str):
        original_image = Image.open(image_path).convert("RGB")
    else:
        original_image = image_path
    
    print(f"1. Shifting image left by {shift_pixels} pixels...")
    shifted_image = shift_image_pixels(original_image, shift_pixels, fill_method)
    
    print("2. Encoding shifted image to latent space...")
    start_latents, _ = load_and_encode_image(shifted_image)
    
    print("3. Performing DDIM inversion with noise injection...")
    inverted_latents = ddim_inversion_with_noise_injection(
        start_latents, 
        prompt, 
        shift_pixels=shift_pixels,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        noise_strength=noise_strength
    )
    
    print("4. Regenerating image from latents...")
    regenerated_image = regenerate_from_latents(
        inverted_latents, 
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        start_step=start_step,
        negative_prompt=negative_prompt
    )
    
    return original_image, shifted_image, regenerated_image

# Example usage
if __name__ == "__main__":
    # Basic usage
    image_path = "./flow/IMG_4026.JPG"  # Replace with your image path
    prompt = "A photo of a fern"  # Describe your image
    
    # Process with default settings
    original, shifted, regenerated = process_image_with_shift_and_inversion(
        image_path, 
        prompt, 
        shift_pixels=50,
        noise_strength=0.1,
        fill_method='edge'
    )
    
    # Save results
    original.save("original.jpg")
    shifted.save("shifted.jpg")
    regenerated.save("regenerated.jpg")
    
    print("Processing complete! Images saved.")
    
    # Advanced usage with more control
    print("\nRunning advanced version...")
    original_adv, shifted_adv, regenerated_adv = process_with_advanced_options(
        image_path,
        prompt,
        shift_pixels=50,
        noise_strength=0.15,  # Higher noise for more variation
        fill_method='mirror',
        guidance_scale=7.5,
        num_inference_steps=50,
        start_step=10,
        negative_prompt="blurry, low quality"
    )
    
    # Save advanced results
    regenerated_adv.save("regenerated_advanced.jpg")
    
    print("Advanced processing complete!")
