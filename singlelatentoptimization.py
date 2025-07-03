import os
import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# --- Device and Models ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
# This version uses the default VAE included with the Stable Diffusion v1.4 model.
# For potentially sharper results, a fine-tuned VAE can be loaded separately.

# --- Paths ---
SCENE_NAME = 'f1ern_shifted'
SOURCE_LATENT_PATH = f'./results/{SCENE_NAME}/inversion/inverted_latent/shifted.pt'
OUTPUT_DIR = f'./denoise_results_{SCENE_NAME}_v3/' # Use a new folder for new results

# --- Generation Parameters ---
# The prompt to guide the denoising. An empty prompt is fine for re-rendering.
PROMPT = "photo of a fern"
CFG_SCALE = 7.5               # Classifier-Free Guidance scale. 7-8 is a good default.
                                # Lower values are more creative, higher values follow the prompt more strictly.

# --- Diffusion Settings ---
# Timestep to add noise. This determines how much the original latent is altered.
# A value of 50 is very low, meaning little noise. Try values like 200, 400, or 600 for more variation.
NOISE_TIMESTEP = 100

# Total number of steps for the denoising process. 50 is a good standard value for DDIM.
NUM_INFERENCE_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# HELPER: Decode latents to an image and save it
# -----------------------------------------------------------------------------
def decode_latents_to_image(pipe, latents, save_path):
    """Decodes latents into a PIL image and saves it using the pipeline's VAE."""
    # The VAE is usually best in float32 for quality. Ensure latents match its dtype.
    latents = latents.to(
        device=pipe.vae.device,
        dtype=pipe.vae.dtype
    )
    # The latent space is scaled. We need to un-scale it before decoding.
    latents = 1 / pipe.vae.config.scaling_factor * latents

    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype(np.uint8)
    Image.fromarray(image).save(save_path)
    print(f"Saved image to {save_path}")

# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------
def main():
    # 1. Load models and scheduler
    # We will use the default DDIM scheduler configuration from the model repo
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    print("Loading Stable Diffusion Pipeline with its default VAE...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=torch.float16, # Load UNet and Text Encoder in float16 for performance
        safety_checker=None,
        feature_extractor=None,
    ).to(DEVICE)
    
    # For best quality, it's often recommended to run the VAE in float32.
    pipe.vae.to(dtype=torch.float32)

    pipe.enable_attention_slicing()

    # 2. Prepare text embeddings for CFG
    print(f"Preparing embeddings for prompt: '{PROMPT}'")
    # Conditional embedding (for the prompt)
    text_input = pipe.tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(DEVICE))[0]

    # Unconditional embedding (for guidance)
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(DEVICE))[0]

    # Concatenate for single-pass processing
    guidance_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(DEVICE, dtype=torch.float16)

    # 3. Load and prepare the source latent
    print(f"Loading source latent from: {SOURCE_LATENT_PATH}")
    # Ensure latent is loaded onto the correct device and in the correct format
    source_latent = torch.load(SOURCE_LATENT_PATH, map_location=DEVICE)
    if source_latent.ndim == 3:
        source_latent = source_latent.unsqueeze(0) # Ensure [1, C, H, W]
    
    # Cast to the same dtype as the UNet for processing
    source_latent = source_latent.to(DEVICE, dtype=torch.float16)


    # 4. Add noise to the latent at the chosen timestep
    # Set the scheduler's timesteps for the diffusion process
    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    
    print(f"Adding noise at timestep {NOISE_TIMESTEP}...")
    noise = torch.randn_like(source_latent)
    
    # Find the closest actual timestep in the schedule to add noise at.
    # The timesteps are in descending order, e.g., [981, 961, ..., 1].
    # We find the timestep corresponding to our desired noise level.
    noise_level_percent = NOISE_TIMESTEP / scheduler.config.num_train_timesteps
    start_step_index = int(len(scheduler.timesteps) * (1 - noise_level_percent))
    add_noise_timestep = scheduler.timesteps[start_step_index]

    noisy_latent = scheduler.add_noise(source_latent, noise, add_noise_timestep.unsqueeze(0))

    # Save the noisy image for inspection
    decode_latents_to_image(pipe, noisy_latent, os.path.join(OUTPUT_DIR, '00_noisy_latent.png'))

    # 5. Denoise the latent (Reverse Diffusion)
    # We will start the denoising loop from where we added the noise.
    timesteps_to_run = scheduler.timesteps[start_step_index:]
    
    print(f"Starting denoising from step {start_step_index} ({len(timesteps_to_run)} steps total)...")
    current_latent = noisy_latent

    for i, t in enumerate(tqdm(timesteps_to_run, desc="Denoising")):
        # Expand the latent input for CFG (run two inferences in one pass)
        latent_model_input = torch.cat([current_latent] * 2)

        with torch.no_grad():
            # Predict the noise residual
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=guidance_embeddings).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_guided = noise_pred_uncond + CFG_SCALE * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        step_output = scheduler.step(noise_pred_guided, t, current_latent)
        current_latent = step_output.prev_sample

        # Optional: Save intermediate steps
        if (i + 1) % 10 == 0:
            decode_latents_to_image(pipe, current_latent, os.path.join(OUTPUT_DIR, f'denoised_step_{start_step_index+i+1:03d}.png'))

    # 6. Decode the final denoised latent and save
    print("Denoising complete. Decoding and saving final image...")
    decode_latents_to_image(pipe, current_latent, os.path.join(OUTPUT_DIR, 'denoised_final.png'))
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()