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
    # 1) Ensure directories exist
    os.makedirs("latents", exist_ok=True)
    os.makedirs("latents/images", exist_ok=True)

    # 2) Define 20 “similar” prompts
    prompt_list = [
        "a fern",
        "a close-up of a fern",
        "a picture of a fern",
        "a green plant with fronds",
        "a botanical fern portrait",
        "a macro shot of a fern",
        "a lone fern leaf",
        "a cluster of ferns in the forest",
        "a potted fern",
        "a small wild fern",
        "a fern under sunlight",
        "a fresh green fern",
        "a detailed fern plant",
        "a fern against a white background",
        "a fern in a forest",
        "a young fern shoot",
        "a fern sprouting",
        "a delicate fern frond",
        "a tropical fern plant",
        "a single fern leaf on wood",
    ]

    for prompt in prompt_list:
        task_type = 'inversion'
        # task_type = 'editing'
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_safetensors=True).to(DEVICE)
        tokenizer  = ldm_stable.tokenizer

        scene_name  = f'fern_{prompt}'   # 'leaves' 
        file_name = 'results/{}/inversion'.format(scene_name)
        os.makedirs(file_name+'/inv_image', exist_ok=True)
        os.makedirs(file_name+'/col_image', exist_ok=True)
        os.makedirs(file_name+'/attn_image', exist_ok=True)
        os.makedirs(file_name+'/inverted_latent', exist_ok=True)

        base_prompt = "a image of {} from camera location azimuthal {} elevation {} and distance {}"
        images_dir  = natsorted(glob('../../../dataset/nerf_llff_data/fern/images/*'))
        pose_file   = np.load('../../../dataset/nerf_llff_data/fern/1_poses.npy')
        # images_dir  = natsorted(glob('../../../dataset/nerf_llff_data/{}/images/*'.format(scene_name)))
        # pose_file   = np.load('../../../dataset/nerf_llff_data/{}/2_poses.npy'.format(scene_name))


        for path_idx, (image_path, camera_pose) in enumerate(zip(images_dir, pose_file)):

            # x, y, z  = camera_pose[:, 3]
            # distance = np.round(np.sqrt(x**2 + y**2 + z**2), 2)
            # theta    = np.round(np.arctan(z/x), 2)
            # phi      = np.round(np.arccos(y/distance), 2)
            # prompt   = base_prompt.format(int(theta*100), int(phi*100), int(distance*100))
            # prompt = 'an image of a {}'.format(scene_name)

            inversion_network = NullInversion(ldm_stable, GUIDANCE_SCALE, NUM_DIFFUSION_STEPS)

            # import pdb; pdb.set_trace()

            (image_gt, image_enc), x_t, uncond_embeddings = inversion_network.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True)

            _, text_embeddings = inversion_network.context.chunk(2)

            if uncond_embeddings == None:
                uncond_embeddings, _ = inversion_network.context.chunk(2)

            print("Modify or remove offsets according to your image!")
            folder_loc = 'results/{}/inversion/'.format(scene_name)
            file_id    = prompt.replace(' ', '_')

            controller = AttentionStore()
            prompts = [prompt]
            image_inv, x_t, image_latents = run_and_display(prompts, controller, run_baseline=False,\
                                                                latent=x_t, text_embeddings=text_embeddings,\
                                                                uncond_embeddings=uncond_embeddings,\
                                                                verbose=False,\
                                                                file_name=folder_loc+'/inv_image/'+f'{path_idx}_'+file_id,\
                                                                generator=GENERATOR,\
                                                                task_type=task_type)
            
            torch.save(image_latents, folder_loc+'inverted_latent/'+f'{path_idx}_final_latent_'+file_id+'.pt')
            torch.save(torch.cat(uncond_embeddings, dim=0), folder_loc+'inverted_latent/'+f'{path_idx}_uncond_latents_'+file_id+'.pt')

            print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
            attention_utils.view_images([image_gt, image_enc, image_inv[0]], file_name=folder_loc+'/col_image/'+f'{path_idx}_col_'+file_id)
            
            attention_utils.show_cross_attention(attention_store=controller,\
                                                res=16,\
                                                from_where=["up", "down"],\
                                                tokenizer=tokenizer,\
                                                prompts=prompts,\
                                                prompt_idx=0,\
                                                file_name=folder_loc+'/attn_image/'+f'{path_idx}_attn_'+file_id)