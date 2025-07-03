# References: https://github.com/google/prompt-to-prompt/tree/main

import os
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusion_inversion import NullInversion

import attention_edit
import attention_utils
from attention_utils import AttentionControl, EmptyControl, AttentionStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS  = 77
LOW_RESOURCE = False

np.random.seed(8888)
GENERATOR = torch.manual_seed(8888)

def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, text_embeddings=None, uncond_embeddings=None, verbose=True, file_name='tmp', task_type=None):

    if run_baseline:
        print('without prompt2prompt')
        images, latent, image_latents = run_and_display(prompts=prompts,\
                                         controller=EmptyControl(),\
                                         latent=latent,\
                                         run_baseline=False,\
                                         generator=generator,\
                                         file_name=file_name,\
                                         task_type=task_type)
        print('with prompt2prompt')
    
    images, x_t, image_latents = attention_edit.text2image_ldm_stable(ldm_stable,\
                                                    prompts,\
                                                    controller,\
                                                    latent=latent,\
                                                    num_inference_steps=NUM_DIFFUSION_STEPS,\
                                                    guidance_scale=GUIDANCE_SCALE,\
                                                    generator=generator,\
                                                    uncond_embeddings=uncond_embeddings,\
                                                    text_embeddings=text_embeddings,\
                                                    task_type=task_type)
    
    attention_utils.view_images(images, file_name=file_name)

    return images, x_t, image_latents


def get_text_embedding(ldm_stable, prompts):
    # Generating Text Prompt Embedding
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]

    return text_input, text_embeddings

def get_uncond_embedding(ldm_stable, batch_size, max_length):
    # Generating Null Prompt Embedding
    uncond_input    = ldm_stable.tokenizer([""] * batch_size,\
                                        padding="max_length",\
                                        max_length=max_length,\
                                        return_tensors="pt")    
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(ldm_stable.device))[0]
    return uncond_embeddings


if __name__ == '__main__':

    task_type = 'inversion'
    # task_type = 'editing'
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_safetensors=True).to(DEVICE)
    tokenizer  = ldm_stable.tokenizer

    file_name = 'results_rerender/inversion/'
    os.makedirs(file_name+'/inv_image', exist_ok=True)
    os.makedirs(file_name+'/col_image', exist_ok=True)
    os.makedirs(file_name+'/attn_image', exist_ok=True)
    os.makedirs(file_name+'/inverted_latent', exist_ok=True)

    prompt_lst = []
    uncond_embeddings_lst = []
    
    for uncond_latent_path in natsorted(glob('./results/inversion/inverted_latent/*_uncond_latents_*.pt')):
        prompt_lst.append(' '.join(os.path.splitext(os.path.basename(uncond_latent_path))[0].split('_')[3:]))
        uncond_embeddings_lst.append(torch.load(uncond_latent_path).unsqueeze(dim=0))

    uncond_embeddings_lst = torch.cat(uncond_embeddings_lst, dim=0)

    base_image  = '../../../dataset/nerf_llff_data/fern/images/IMG_4026.JPG'
    base_prompt = 'an image of a fern'
    inversion_network_base = NullInversion(ldm_stable, GUIDANCE_SCALE, NUM_DIFFUSION_STEPS)
    (_, _), x_t, _ = inversion_network_base.invert(base_image, base_prompt, offsets=(0,0,200,0), verbose=True)

    for path_idx, (prompt, uncond_embeddings) in enumerate(zip(prompt_lst, uncond_embeddings_lst)):

        # import pdb; pdb.set_trace()

        prompts = [prompt]
        inversion_network = NullInversion(ldm_stable, GUIDANCE_SCALE, NUM_DIFFUSION_STEPS)
        inversion_network.init_prompt(prompt)
        _, text_embeddings = inversion_network.context.chunk(2)

        if path_idx == 0:
            base_uncond_embeddings = uncond_embeddings
        else:
            uncond_embeddings  = 0.2 * uncond_embeddings + 0.8 * base_uncond_embeddings

        uncond_embeddings  = [tmp_embd for tmp_embd in uncond_embeddings]

        folder_loc = 'results_rerender/inversion/'
        file_id    = prompt.replace(' ', '_')

        controller = AttentionStore()
        image_inv, x_t_c, image_latents = run_and_display(prompts, controller, run_baseline=False,\
                                                            latent=x_t, text_embeddings=text_embeddings,\
                                                            uncond_embeddings=uncond_embeddings,\
                                                            verbose=False,\
                                                            file_name=folder_loc+'/inv_image/'+f'{path_idx}_'+file_id,\
                                                            generator=GENERATOR,\
                                                            task_type=task_type)
        

        print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
        attention_utils.view_images(image_inv[0], file_name=folder_loc+'/col_image/'+f'{path_idx}_col_'+file_id)
        
        attention_utils.show_cross_attention(attention_store=controller,\
                                            res=16,\
                                            from_where=["up", "down"],\
                                            tokenizer=tokenizer,\
                                            prompts=prompts,\
                                            prompt_idx=0,\
                                            file_name=folder_loc+'/attn_image/'+f'{path_idx}_attn_'+file_id)