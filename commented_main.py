# References: https://github.com/google/prompt-to-prompt/tree/main

import os                        # For interacting with the operating system (e.g., creating directories)
import numpy as np               # Numerical operations, random seeding, array manipulation
from glob import glob            # File pattern matching to list files in directories
from natsort import natsorted    # Natural string sorting for file paths (e.g., "file2" before "file10")
from typing import Optional, Union, Tuple, List, Callable, Dict  # Type hinting
from tqdm import tqdm            # Progress bar utility for loops
import torch                     # PyTorch: tensor operations, GPU support, model inference
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP text encoder and tokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler  # Stable Diffusion pipeline and scheduler
from diffusion_inversion import NullInversion  # Null-inversion module for inverting images into diffusion latents

# Modules implementing attention-based editing for Prompt-to-Prompt
import attention_edit
import attention_utils
from attention_utils import AttentionControl, EmptyControl, AttentionStore  # Various attention controllers

# Choose device: use GPU ("cuda") if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of diffusion timesteps for inference/inversion
NUM_DIFFUSION_STEPS = 50

# Guidance scale for classifier-free guidance during sampling
GUIDANCE_SCALE = 7.5

# Maximum number of tokens (words) supported by the text encoder (CLIP)
MAX_NUM_WORDS  = 77

# Option to reduce resource usage (e.g., for lower-memory environments). Not used in this script.
LOW_RESOURCE = False

# Fix random seeds for reproducibility (both NumPy and PyTorch)
np.random.seed(8888)
GENERATOR = torch.manual_seed(8888)


def run_and_display(
    prompts: List[str],
    controller: AttentionControl,
    latent: torch.Tensor = None,
    run_baseline: bool = False,
    generator: torch.Generator = None,
    text_embeddings: torch.Tensor = None,
    uncond_embeddings: torch.Tensor = None,
    verbose: bool = True,
    file_name: str = 'tmp',
    task_type: Optional[str] = None
) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
    """
    Runs the Prompt-to-Prompt text-to-image pipeline with or without attention editing,
    then displays (and optionally saves) the resulting images.

    Args:
        prompts: A list of text prompts to guide generation.
        controller: An AttentionControl instance (e.g., AttentionStore, EmptyControl).
        latent: Input latent tensor to start from (for inversion or editing).
        run_baseline: If True, first run generation without prompt-to-prompt modifications.
        generator: A PyTorch random number generator for reproducibility.
        text_embeddings: Precomputed conditional text embeddings.
        uncond_embeddings: Precomputed unconditional text embeddings for classifier-free guidance.
        verbose: Whether to print progress messages.
        file_name: Base filename or path prefix to save images.
        task_type: A string indicating the type of task (e.g., 'inversion' or 'editing').

    Returns:
        images: A list of generated images as numpy arrays.
        x_t: The final latent tensor after diffusion steps.
        image_latents: Latents saved at each denoising timestep (for analysis).
    """
    if run_baseline:
        # If we want to see how generation looks without any prompt-to-prompt editing:
        print('without prompt2prompt')
        images, latent, image_latents = run_and_display(
            prompts=prompts,
            controller=EmptyControl(),      # EmptyControl means no attention editing
            latent=latent,
            run_baseline=False,
            generator=generator,
            file_name=file_name,
            task_type=task_type
        )
        print('with prompt2prompt')
        # Continue below to run with the actual controller passed in

    # Run the text-to-image generation with attention editing applied
    # attention_edit.text2image_ldm_stable wraps around StableDiffusionPipeline.safety checker
    images, x_t, image_latents = attention_edit.text2image_ldm_stable(
        ldm_stable,                 # The loaded Stable Diffusion pipeline
        prompts,                    # Text prompts
        controller,                 # Attention editing controller (e.g., AttentionStore)
        latent=latent,              # Starting latent: None for fresh generation, or inverted latents for editing
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,        # PyTorch RNG for deterministic outputs
        uncond_embeddings=uncond_embeddings,  # Unconditional embeddings for guidance
        text_embeddings=text_embeddings,      # Conditional embeddings for the prompts
        task_type=task_type         # 'inversion' or 'editing' (controls architecture of attention mask)
    )

    # Display (and save) generated images using attention_utils utility
    attention_utils.view_images(images, file_name=file_name)

    # Return the generated images, final latent after diffusion, and all saved latents
    return images, x_t, image_latents


def get_text_embedding(
    ldm_stable: StableDiffusionPipeline,
    prompts: List[str]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Tokenizes prompts and obtains conditional text embeddings from the CLIP text encoder.

    Args:
        ldm_stable: The Stable Diffusion pipeline (contains the tokenizer and text encoder).
        prompts: A list of text prompts.

    Returns:
        text_input: A dictionary containing tokenized input_ids, attention_mask, etc.
        text_embeddings: A tensor of shape [batch_size, seq_len, embedding_dim].
    """
    # Tokenize the prompts to fixed length (pad or truncate to max_length)
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Move token IDs to the same device as the model and feed into the CLIP text encoder
    # The [0] extracts the last hidden state (embeddings) from the model output
    text_embeddings = ldm_stable.text_encoder(
        text_input.input_ids.to(ldm_stable.device)
    )[0]

    return text_input, text_embeddings


def get_uncond_embedding(
    ldm_stable: StableDiffusionPipeline,
    batch_size: int,
    max_length: int
) -> torch.Tensor:
    """
    Generates embeddings for an "empty" or null prompt, used for classifier-free guidance.

    Args:
        ldm_stable: The Stable Diffusion pipeline containing the tokenizer and encoder.
        batch_size: Number of examples in the batch (usually 1 or more).
        max_length: Maximum token sequence length for padding/truncating.

    Returns:
        uncond_embeddings: A tensor of shape [batch_size, seq_len, embedding_dim], for null prompts.
    """
    # Create a batch of empty strings (one per sample in the batch)
    uncond_input = ldm_stable.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Encode the null prompts (empty text) to produce "unconditional" embeddings
    uncond_embeddings = ldm_stable.text_encoder(
        uncond_input.input_ids.to(ldm_stable.device)
    )[0]

    return uncond_embeddings


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Main script: inversion + editing over a dataset of NeRF-rendered images.
    #--------------------------------------------------------------------------

    task_type = 'inversion'
    # To switch to editing based on inverted latents, set: task_type = 'editing'

    # Create a DDIM scheduler with specific beta schedule parameters
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )

    # Load the pretrained Stable Diffusion v1.4 model with the custom scheduler,
    # and move it to the selected device (GPU/CPU).
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=scheduler,
        use_safetensors=True
    ).to(DEVICE)

    # The tokenizer is needed later for attention visualization
    tokenizer  = ldm_stable.tokenizer

    # -------------------------------------------------------------------------
    # Define scene-specific parameters and create output folders
    # -------------------------------------------------------------------------
    scene_name  = 'flower'   # Example scene: 'flower' (could also be 'leaves')
    # Base folder where inversion results will be saved
    file_name = f'results/{scene_name}/inversion'

    # Create necessary subdirectories if they don't already exist
    os.makedirs(file_name + '/inv_image', exist_ok=True)    # Inverted image outputs
    os.makedirs(file_name + '/col_image', exist_ok=True)    # Side-by-side comparison images
    os.makedirs(file_name + '/attn_image', exist_ok=True)   # Attention maps visualization
    os.makedirs(file_name + '/inverted_latent', exist_ok=True)  # Saved latent tensors

    # Optional prompt template that includes camera parameters (unused here)
    base_prompt = "a image of {} from camera location azimuthal {} elevation {} and distance {}"

    # -------------------------------------------------------------------------
    # Load the dataset of NeRF-rendered images and camera poses
    # (Using natsorted to ensure consistent ordering)
    # -------------------------------------------------------------------------
    images_dir = natsorted(glob(f'../../../dataset/nerf_llff_data/{scene_name}/images/*'))
    pose_file  = np.load(f'../../../dataset/nerf_llff_data/{scene_name}/2_poses.npy')

    # -------------------------------------------------------------------------
    # Loop through each image and its corresponding camera pose
    # -------------------------------------------------------------------------
    for path_idx, (image_path, camera_pose) in enumerate(zip(images_dir, pose_file)):
        """
        For each step:
         1. Build a simple prompt describing the scene (no camera parameters here).
         2. Initialize NullInversion to invert the real image into diffusion latents.
         3. Save the inverted latents and the reconstructed images.
         4. (Optionally) run the Prompt-to-Prompt editing pipeline on the inverted latents.
         5. Save attention maps and side-by-side comparisons.
        """

        # Extract camera position (x, y, z) from the 4x4 pose matrix, if desired:
        # x, y, z  = camera_pose[:, 3]
        # Compute spherical coordinates or other parameters if needed:
        # distance = np.round(np.sqrt(x**2 + y**2 + z**2), 2)
        # theta    = np.round(np.arctan(z / x), 2)
        # phi      = np.round(np.arccos(y / distance), 2)
        # prompt   = base_prompt.format(scene_name, int(theta*100), int(phi*100), int(distance*100))
        
        # For simplicity, use a generic prompt: "an image of a <scene_name>"
        prompt = f'an image of a {scene_name}'
        prompts = [prompt]

        # Initialize the NullInversion network for this Stable Diffusion model
        # This network handles encoding the real image into a starting latent x_t and returning embeddings
        inversion_network = NullInversion(ldm_stable, GUIDANCE_SCALE, NUM_DIFFUSION_STEPS)

        # Perform null-text inversion:
        #   - image_path: path to the ground-truth image from the NeRF dataset
        #   - prompt: text prompt to condition on during inversion
        #   - offsets: cropping/padding offsets for the image inversion process (subject to user tweaking)
        #   - verbose=True prints progress during inversion
        (image_gt, image_enc), x_t, uncond_embeddings = inversion_network.invert(
            image_path,
            prompt,
            offsets=(0, 0, 200, 0),  # Example offsets; adjust based on image size/cropping needs
            verbose=True
        )

        # The inversion network stores "context" which contains the concatenated [uncond, cond] embeddings.
        # We split them into two equal halves: unconditional and text embeddings.
        _, text_embeddings = inversion_network.context.chunk(2)

        # If get_uncond_embedding failed inside NullInversion, fallback:
        if uncond_embeddings is None:
            # Split the context to obtain unconditional embeddings
            uncond_embeddings, _ = inversion_network.context.chunk(2)

        print("Modify or remove offsets according to your image!")

        # Define folder and filename ID prefix for saving results for this iteration
        folder_loc = f'results/{scene_name}/inversion/'
        file_id = prompt.replace(' ', '_')  # e.g., "an_image_of_a_flower"

        # Prepare an attention controller (AttentionStore) to capture cross-attention maps during generation
        controller = AttentionStore()

        # Run the Prompt-to-Prompt pipeline on the inverted latent x_t
        # This produces:
        #   - image_inv: new generated images (numpy arrays) from x_t, with attention edits
        #   - x_t: updated latent after generation (unused here)
        #   - image_latents: saved latents at each denoising step
        image_inv, x_t, image_latents = run_and_display(
            prompts,
            controller,
            run_baseline=False,          # Skip baseline/no-edit run; we only want the edited output
            latent=x_t,                  # Start from the inverted latent
            text_embeddings=text_embeddings,
            uncond_embeddings=uncond_embeddings,
            verbose=False,               # Suppress extra printouts here
            file_name=f'{folder_loc}/inv_image/{path_idx}_{file_id}',
            generator=GENERATOR,         # For reproducibility
            task_type=task_type
        )

        # Save the final latent tensor (after diffusion) to disk
        torch.save(
            image_latents,
            f'{folder_loc}inverted_latent/{path_idx}_final_latent_{file_id}.pt'
        )

        # Save the unconditional latents separately (concatenate if list of tensors)
        torch.save(
            torch.cat(uncond_embeddings, dim=0),
            f'{folder_loc}inverted_latent/{path_idx}_uncond_latents_{file_id}.pt'
        )

        # Print a summary of what images will be shown in comparison:
        print("showing from left to right: the ground truth image, "
              "the vq-autoencoder reconstruction, the null-text inverted image")
        
        # Display (and save) a side-by-side comparison of:
        #   1. ground truth image
        #   2. the reconstructed image from the autoencoder during inversion
        #   3. the final inverted image produced by NullInversion (after diffusion steps)
        attention_utils.view_images(
            [image_gt, image_enc, image_inv[0]],
            file_name=f'{folder_loc}/col_image/{path_idx}_col_{file_id}'
        )

        # Show and save cross-attention maps for the inverted generation:
        #   - attention_store: controller containing stored attention weights
        #   - res: resolution for attention map visualization (here 16x16 patches)
        #   - from_where: which blocks to visualize ("up" and "down" sampler stages)
        #   - tokenizer: needed to map token indices to words
        #   - prompts: original list of text prompts
        #   - prompt_idx: which prompt to visualize attention for (0 since batch size=1)
        attention_utils.show_cross_attention(
            attention_store=controller,
            res=16,
            from_where=["up", "down"],
            tokenizer=tokenizer,
            prompts=prompts,
            prompt_idx=0,
            file_name=f'{folder_loc}/attn_image/{path_idx}_attn_{file_id}'
        )
