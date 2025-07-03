import os
import torch
import numpy as np
from glob import glob
from typing import List, Optional, Tuple
from natsort import natsorted
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.io as io
from torchvision.utils import save_image

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDIMScheduler

from diffusion_inversion import NullInversion
import attention_edit
import attention_utils
from attention_utils import EmptyControl, AttentionStore

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 50
GUIDANCE_SCALE = 7.5
SEED = 8888
SCENE_NAME = 'f1ern_shifted'
OUTPUT_DIR = os.path.join('results', SCENE_NAME)
PROMPT = 'a photo of a fern'
SHIFT_PIXELS = 1000
# change this two for di
INPUT_IMAGE = './images/fern_1.JPG'
save_name = 'shifted.pt'

np.random.seed(SEED)
torch.manual_seed(SEED)


def shift_image(tensor: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shifts the image tensor left by `shift` pixels, filling vacated area with zeros.
    """
    _, _, width = tensor.shape
    if shift >= width:
        return torch.zeros_like(tensor)
    output = torch.zeros_like(tensor)
    output[:, :, : width - shift] = tensor[:, :, shift:]
    return output

def shift_image_with_padding(tensor: torch.Tensor, shift: int, mode: str = 'replicate') -> torch.Tensor:
    """
    A more idiomatic and efficient way to shift with filling using F.pad.
    
    Args:
        tensor (torch.Tensor): The input image tensor (C, H, W).
        shift (int): The number of pixels to shift left.
        mode (str): The padding mode. 'replicate' or 'reflect' are most common.
    
    Returns:
        torch.Tensor: The shifted and padded image tensor.
    """
    c, h, w = tensor.shape
    if shift <= 0:
        return tensor
    if shift >= w:
        return torch.zeros_like(tensor)

    # We want to shift left, which is equivalent to adding padding on the left
    # and then cropping the original width from the left.
    # Padding format is (pad_left, pad_right, pad_top, pad_bottom)
    padding = (shift, 0, 0, 0)
    
    # Add padding to the left. The input to pad needs a batch dimension.
    padded_tensor = F.pad(tensor.unsqueeze(0), padding, mode=mode).squeeze(0)
    
    # Crop the tensor to its original width, effectively removing pixels from the right
    shifted_tensor = padded_tensor[:, :, :w]
    
    return shifted_tensor


def load_and_shift_image(path: str, shift: int) -> torch.Tensor:
    """
    Loads an image as a tensor and shifts it left.
    """
    tensor = io.read_image(path)
    return shift_image_with_padding(tensor, shift)


def prepare_pipeline(model_id: str = "CompVis/stable-diffusion-v1-4") -> StableDiffusionPipeline:
    """
    Initializes the Stable Diffusion pipeline with DDIM scheduler.
    """
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        use_safetensors=True,
    ).to(DEVICE)
    return pipe


def create_output_dirs(base: str) -> None:
    """Creates required directories for outputs."""
    for sub in ['inv_image', 'col_image', 'attn_image', 'inverted_latent']:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def invert_image(
    pipe: StableDiffusionPipeline,
    image_path: str,
    prompt: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Performs inversion to latent space and returns ground truth, encoded image, latent x_t, and image latents.
    """
    inversion = NullInversion(pipe, GUIDANCE_SCALE, NUM_STEPS)
    (img_gt, img_enc), x_t, uncond = inversion.invert(
        image_path, prompt, offsets=(0, 0, 0, 0), verbose=True
    )
    # Split embeddings
    _, text_emb = inversion.context.chunk(2)
    if uncond is None:
        uncond, _ = inversion.context.chunk(2)
    return img_gt, img_enc, x_t, uncond, text_emb


def run_generation(
    pipe: StableDiffusionPipeline,
    prompt: str,
    x_t: torch.Tensor,
    uncond_emb: torch.Tensor,
    text_emb: torch.Tensor,
    controller: AttentionStore,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List, torch.Tensor, List]:
    """
    Generates images with or without Prompt2Prompt editing.
    """
    images, latent, latents = attention_edit.text2image_ldm_stable(
        pipe,
        [prompt],
        controller,
        latent=x_t,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        uncond_embeddings=uncond_emb,
        text_embeddings=text_emb,
        task_type='inversion',
    )
    return images, latent, latents


def main():
    # Prepare outputs
    inv_dir = os.path.join(OUTPUT_DIR, 'inversion')
    create_output_dirs(inv_dir)

    # Load and shift
    shifted = load_and_shift_image(INPUT_IMAGE, SHIFT_PIXELS)
    shifted_path = os.path.join('images', 'fern_shifted.png')
    save_image(shifted.float() / 255.0, shifted_path)

    # Setup model
    pipe = prepare_pipeline()

    # Invert image
    img_gt, img_enc, x_t, uncond_emb, text_emb = invert_image(pipe, shifted_path, PROMPT)

    # Run generation
    store = AttentionStore()
    images, _, latents = run_generation(
        pipe,
        PROMPT,
        x_t,
        uncond_emb,
        text_emb,
        store,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    )

    # Save latents
    torch.save(latents, os.path.join(inv_dir, 'inverted_latent', save_name))
    torch.save(torch.cat(uncond_emb, dim=0), os.path.join(inv_dir, 'inverted_latent', 'uncond_latents.pt'))

    # Visualizations
    attention_utils.view_images(
        [img_gt, img_enc, images[0]],
        file_name=os.path.join(inv_dir, 'col_image', 'comparison.png'),
    )
    attention_utils.show_cross_attention(
        attention_store=store,
        res=16,
        from_where=["up", "down"],
        tokenizer=pipe.tokenizer,
        prompts=[PROMPT],
        prompt_idx=0,
        file_name=os.path.join(inv_dir, 'attn_image', 'attention.png'),
    )


if __name__ == '__main__':
    main()
