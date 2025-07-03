import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from glob import glob
from natsort import natsorted
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import Attention, AttnProcessor
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import cv2
import math
from typing import Dict, Tuple
# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
GUIDANCE_SCALE = 7.5
NUM_DIFFUSION_STEPS = 200
NOISE_LEVEL = 0.8
NUM_EPIPOLAR_SAMPLES = 128
FEATURE_SIMILARITY_THRESHOLD = 0.1

scene_name = 'fern'
LATENTS_DIR = f'./results/{scene_name}/inversion/inverted_latent/'
OUTPUT_DIR = f'./denoise_epipolar_{scene_name}/'

# Seeds for reproducibility
np.random.seed(8888)
torch.manual_seed(8888)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(8888)




# -------------------------------------Epipolar geometry functions--------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------



def skew_symmetric_matrix(t: torch.Tensor) -> torch.Tensor:
    """This constructs a skew symmetric matrix tx associated with a 3 vector t
        so that for any vector v: [t]xv = t v that is left multiplying by the skew symmetric matrix
        implements the cross product.
    """

    if t.numel() == 1:
        # If somebody passed in a scalar tensor, 
        # we fall back to the zero‐vector [0,0,0].
        t = torch.zeros(3, device=t.device, dtype=t.dtype)
    elif t.numel() != 3:
        raise ValueError(f"Translation vector must have 3 elements, got {t.numel()}")
    
    t = t.flatten()
    
    if t.dim() == 1:
        # A single 3-vector → treat as a batch of size 1. shape(1,3)
        batch_size = 1
        t = t.unsqueeze(0)
    else:
        # If already (B,3) → batch_size = B
        batch_size = t.shape[0]
    
    t = t.view(batch_size, 3)

    # zeros = torch.zeros(batch_size, 1, device=t.device, dtype=t.dtype)

    # Prepare an output tensor of shape (B,3,3)
    skew = torch.zeros(batch_size, 3, 3, device=t.device, dtype=t.dtype)
    skew[:, 0, 1] = -t[:, 2]
    skew[:, 0, 2] = t[:, 1]
    skew[:, 1, 0] = t[:, 2]
    skew[:, 1, 2] = -t[:, 0]
    skew[:, 2, 0] = -t[:, 1]
    skew[:, 2, 1] = t[:, 0]
    
    return skew.squeeze(0) if batch_size == 1 else skew




def compute_epipolar_line(point: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Given a point p in one image(target view), you want the corresponding epipolar line L in the other image(source view)
    In homogenous coordinates l=Fp. F is a fundamental matrix.
    """

    if point.dim() == 1:
        point = point.unsqueeze(0)
    
    ones = torch.ones(point.shape[0], 1, device=point.device, dtype=point.dtype)
    point_homo = torch.cat([point, ones], dim=1)
    
    K_inv = torch.inverse(K).to(dtype=point.dtype, device=point.device)
    t_skew = skew_symmetric_matrix(t).to(dtype=point.dtype, device=point.device)
    
    # Fundamental matrix: F = K^-T * R * [t]_x * K^-1
    F = K_inv.T @ t_skew @ R @ K_inv
    
    # Compute epipolar line: l = F * p
    epipolar_lines = (F @ point_homo.T).T

    # normalizing the lines so that distance from a pixel to the like is |ax+by+c|
    norm = torch.sqrt(epipolar_lines[:,0]**2 + epipolar_lines[:,1]**2).unsqueeze(1)
    epipolar_lines = epipolar_lines / norm

    
    return epipolar_lines.squeeze(0) if epipolar_lines.shape[0] == 1 else epipolar_lines





def sample_points_on_epipolar_line(line: torch.Tensor, image_size: Tuple[int, int], 
                                 num_samples: int = NUM_EPIPOLAR_SAMPLES) -> torch.Tensor:
    """Sample points along epipolar line within image bounds."""

    # Unpacking the inputs
    a, b, c = line
    h, w = image_size
    
    points = []
    
    # parameterize by x or y
    # This avoids dividing by a very small coefficient (which would amplify noise).

    if abs(b) > abs(a):
        # line is steeper in y than in x => we step along x
        # sampling and filtering
        x_coords = torch.linspace(0, w-1, num_samples, device=line.device)
        for x in x_coords:
            if abs(b) > 1e-6:
                y = -(a * x + c) / b
                if 0 <= y < h:
                    points.append([x, y])
    else:
        # line is steeper in x than along y => we step along y
        # sampling and filtering
        y_coords = torch.linspace(0, h-1, num_samples, device=line.device)
        for y in y_coords:
            if abs(a) > 1e-6:
                x = -(b * y + c) / a
                if 0 <= x < w:
                    points.append([x, y])
    
    if len(points) == 0:
        return torch.tensor([[w/2, h/2]], device=line.device, dtype=line.dtype)
    
    return torch.stack([torch.tensor(p, device=line.device, dtype=line.dtype) for p in points])








# -------------------------------------Epipolar Attention--------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


class EpipolarAttentionProcessor:
    """True epipolar attention via geometric line sampling."""

    def __init__(self,
                 reference_features_cache: Dict[str, torch.Tensor] = None,
                 camera_params: Dict[str, torch.Tensor] = None,
                 num_samples: int = NUM_EPIPOLAR_SAMPLES):
        self.reference_features_cache = reference_features_cache or {}
        self.num_samples = num_samples
        # camera_params: expects 'intrinsics', 'R', 't'
        if camera_params:
            self.update_camera_params(**camera_params)
        else:
            self.K_inv = None
            self.F = None

    def update_camera_params(self,
                             intrinsics: torch.Tensor,
                             R: torch.Tensor,
                             t: torch.Tensor):
        """Compute and store inverse intrinsics and fundamental matrix."""
        K = intrinsics
        self.K_inv = torch.inverse(K)
        # essential and fundamental
        E = skew_symmetric_matrix(t) @ R
        self.F = self.K_inv.T @ E @ self.K_inv

    def update_reference_features(self, layer_name: str, features: torch.Tensor):
        """Cache a reference feature map for a given layer."""
        self.reference_features_cache[layer_name] = features.detach().clone()

    def __call__(self,
                 attn,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: torch.Tensor = None,
                 attention_mask: torch.Tensor = None,
                 layer_name: str = None,
                 **kwargs) -> torch.Tensor:
        # cross-attn: standard
        if encoder_hidden_states is not None:
            return self._standard_attention(attn, hidden_states, encoder_hidden_states, attention_mask)

        # self-attn: if we have reference features and camera params
        if layer_name and layer_name in self.reference_features_cache and self.F is not None:
            return self._epipolar_attention(attn, hidden_states, layer_name)
        # otherwise fallback
        return self._standard_attention(attn, hidden_states)

    def _standard_attention(self, attn, hidden_states: torch.Tensor,
                            encoder_hidden_states: torch.Tensor = None,
                            attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Standard (cross/self) attention computation."""
        q = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)
        attn_probs = attn.get_attention_scores(q, k, attention_mask)
        out = torch.bmm(attn_probs, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    def _epipolar_attention(self, attn, hidden_states: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Geometry-driven epipolar attention."""
        # shapes: hidden_states [B, L, D], reference [B, L, D]
        B, L, D = hidden_states.shape
        ref_feats = self.reference_features_cache[layer_name]  # [B, L, D]
        H = W = int(math.sqrt(L))
        assert H*W == L, "Hidden states must form square feature map"

        # 1) query projection
        Q = attn.to_q(hidden_states).view(B, L, D)

        # 2) construct image grid [B, L, 2]
        xs = torch.arange(W, device=Q.device)
        ys = torch.arange(H, device=Q.device)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, L, 2).repeat(B,1,1).float()

        # 3) homogeneous pts & epipolar lines
        ones = torch.ones(B, L, 1, device=Q.device)
        pts_h = torch.cat([grid, ones], dim=2)        # [B,L,3]
        lines = (self.F[None] @ pts_h.unsqueeze(-1)).squeeze(-1)  # [B,L,3]
        norms = torch.sqrt(lines[:,:,0]**2 + lines[:,:,1]**2).unsqueeze(-1)
        lines = lines / (norms + 1e-8)                 # normalize

        # 4) sample along line via grid_sample
        S = self.num_samples
        t = torch.linspace(-1,1,S,device=Q.device).view(1,1,S,1)
        t = t.expand(B, L, S, 1)
        a, b, c = lines.unbind(-1)                    # each [B,L]
        dir_x =  b.view(B,L,1)
        dir_y = -a.view(B,L,1)
        dx_ndc = dir_x / ((W-1)/2)
        dy_ndc = dir_y / ((H-1)/2)
        cx_ndc = (grid[:,:,0] / ((W-1)/2) - 1).view(B,L,1)
        cy_ndc = (grid[:,:,1] / ((H-1)/2) - 1).view(B,L,1)
        u = cx_ndc + t * dx_ndc
        v = cy_ndc + t * dy_ndc
        sample_grid = torch.stack([u, v], dim=-1)    # [B,L,S,2]

        ref = ref_feats.permute(0,2,1).view(B, D, H, W)  # [B,D,H,W]
        ref_rep = ref.unsqueeze(1).repeat(1,L,1,1,1).view(B*L,D,H,W)
        grid_rep = sample_grid.view(B*L,S,1,2)
        samples = F.grid_sample(ref_rep, grid_rep, align_corners=True)  # [B*L,D,S,1]
        samples = samples.view(B,L,D,S).permute(0,1,3,2)  # [B,L,S,D]

        # 5) dot+softmax
        Qq = Q.view(B, L, 1, D)
        attn_logits = (Qq * samples).sum(-1) / math.sqrt(D)
        attn_w = F.softmax(attn_logits, dim=-1)       # [B,L,S]
        context = (attn_w.unsqueeze(-1) * samples).sum(2)  # [B,L,D]

        # 6) final projection
        out = context.view(B, L, D)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out



# -------------------------------------Multiview Generator--------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

class MultiViewEpipolarGenerator:
    """Multi-view generator using EpipolarAttentionProcessor."""

    def __init__(self, ldm_stable, camera_poses: List[np.ndarray], intrinsics: np.ndarray):
        self.ldm_stable = ldm_stable
        self.camera_poses = camera_poses
        self.intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=DEVICE)

        # Target layers for epipolar attention
        self.target_layers = [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn1",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
            "up_blocks.2.attentions.0.transformer_blocks.0.attn1"
        ]

        # Epipolar attention processor (shared instance)
        self.epipolar_processor = EpipolarAttentionProcessor()
        self.original_processors = {}

    def _assign_processor(self):
        """Assign EpipolarAttentionProcessor to target layers, saving originals."""
        self.original_processors.clear()
        for name, module in self.ldm_stable.unet.named_modules():
            if name in self.target_layers and hasattr(module, "processor"):
                self.original_processors[name] = module.processor
                module.processor = self.epipolar_processor

    def _restore_processors(self):
        """Restore original attention processors."""
        for name, module in self.ldm_stable.unet.named_modules():
            if name in self.original_processors and hasattr(module, "processor"):
                module.processor = self.original_processors[name]
        self.original_processors.clear()

    def extract_reference_features(self, reference_latent: torch.Tensor, prompt: str) -> Dict[str, torch.Tensor]:
        """Extract reference features by running a forward pass."""
        # (same as before)
        text_input = self.ldm_stable.tokenizer(
            [prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(DEVICE))[0]

        captured_features = {}
        def capture_hook(layer_name):
            def hook_fn(module, input, output):
                captured_features[layer_name] = output.detach().clone()
            return hook_fn

        capture_hooks = []
        for name, module in self.ldm_stable.unet.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(capture_hook(name))
                capture_hooks.append(hook)

        timestep = torch.tensor([20], device=DEVICE)
        with torch.no_grad():
            _ = self.ldm_stable.unet(reference_latent, timestep, encoder_hidden_states=text_embeddings).sample

        for hook in capture_hooks:
            hook.remove()
        return captured_features

    def generate_novel_view(self, reference_latent: torch.Tensor, reference_pose: np.ndarray,
                            target_pose: np.ndarray, prompt: str) -> torch.Tensor:
        """Generate novel view using epipolar attention processor."""

        # 1. Extract reference features
        print("Extracting reference features...")
        reference_features = self.extract_reference_features(reference_latent, prompt)
        print(f"Extracted features for {len(reference_features)} layers")

        # 2. Compute relative pose and update processor
        reference_pose = reference_pose.astype(np.float32)
        target_pose = target_pose.astype(np.float32)
        relative_pose = np.linalg.inv(reference_pose) @ target_pose
        R = torch.tensor(relative_pose[:3, :3], dtype=torch.float32, device=DEVICE)
        t = torch.tensor(relative_pose[:3, 3], dtype=torch.float32, device=DEVICE)

        # 3. Update processor with features and camera params
        for layer_name, feats in reference_features.items():
            self.epipolar_processor.update_reference_features(layer_name, feats)
        self.epipolar_processor.update_camera_params(self.intrinsics, R, t)

        # 4. Assign processor
        self._assign_processor()

        try:
            # 5. Generate novel view
            novel_view_latent = self._generate_with_noise_injection(reference_latent, prompt)
        finally:
            # 6. Restore original processors
            self._restore_processors()
            # 7. Clear reference features from processor
            self.epipolar_processor.reference_features_cache.clear()

        return novel_view_latent

    def _generate_with_noise_injection(self, reference_latent: torch.Tensor, prompt: str) -> torch.Tensor:
        # (identical to previous version)
        text_input = self.ldm_stable.tokenizer(
            [prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(DEVICE))[0]
        uncond_input = self.ldm_stable.tokenizer(
            [""], padding="max_length", max_length=77, return_tensors="pt"
        )
        uncond_embeddings = self.ldm_stable.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
        self.ldm_stable.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
        timesteps = self.ldm_stable.scheduler.timesteps
        noise_timestep_idx = int(NOISE_LEVEL * len(timesteps))
        noise_timestep = timesteps[noise_timestep_idx:noise_timestep_idx+1]
        noise = torch.randn_like(reference_latent)
        noisy_latent = self.ldm_stable.scheduler.add_noise(reference_latent, noise, noise_timestep)
        latent = noisy_latent.clone()
        relevant_timesteps = timesteps[noise_timestep_idx:]
        with torch.no_grad():
            for i, t in enumerate(tqdm(relevant_timesteps, desc="Denoising with Epipolar Attention")):
                latent_model_input = torch.cat([latent] * 2)
                text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])
                noise_pred = self.ldm_stable.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings_cfg
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
                latent = self.ldm_stable.scheduler.step(noise_pred, t, latent).prev_sample
        return latent







# -------------------------------------Helper Functions--------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------



def create_camera_poses(num_views: int = 4, radius: float = 100.0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Create camera poses in a circle around the object."""
    poses = []
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
    
    for angle in angles:
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.0
        
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    intrinsics = np.array([
        [20.0, 0.0, 10.0],
        [0.0, 20.0, 10.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return poses, intrinsics




def get_text_embedding(ldm_stable, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates text embeddings for a list of prompts."""
    text_input = ldm_stable.tokenizer(
        prompts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(DEVICE))[0]
    return text_input, text_embeddings

def get_uncond_embedding(ldm_stable, batch_size: int) -> torch.Tensor:
    """Generates null-prompt embeddings."""
    max_length = ldm_stable.tokenizer.model_max_length
    uncond_input = ldm_stable.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    return uncond_embeddings

def latent_to_image(vae, latent: torch.Tensor) -> np.ndarray:
    """Convert latent to image."""
    with torch.no_grad():
        latent = latent / vae.config.scaling_factor
        image = vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image[0]

def save_image(image: np.ndarray, path: str):
    """Save image to file."""
    Image.fromarray(image).save(path)








# -------------------------------------Main Execution--------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
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
    
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        scheduler=scheduler,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_safetensors=True
    ).to(DEVICE)
    
    # Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Images will be saved in: {OUTPUT_DIR}")
    
    # Create camera poses
    camera_poses, intrinsics = create_camera_poses(num_views=5)
    print(f"Created {len(camera_poses)} camera poses")
    
    # Load or create reference latent
    prompt = 'a photo of a fern'
    print(f"Using prompt: '{prompt}'")
    
    latent_files = natsorted(glob(os.path.join(LATENTS_DIR, '*.pt')))
    if latent_files:
        reference_latent = torch.load(latent_files[0], map_location=DEVICE)
        if reference_latent.dim() == 3:
            reference_latent = reference_latent.unsqueeze(0)
        reference_latent = reference_latent.to(dtype=ldm_stable.unet.dtype)
    else:
        # Generate reference latent if none exists
        print("Generating reference latent...")
        os.makedirs(LATENTS_DIR, exist_ok=True)
        reference_latent = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=ldm_stable.unet.dtype)
        torch.save(reference_latent.cpu(), os.path.join(LATENTS_DIR, 'reference_latent.pt'))
    
    # Initialize multi-view generator
    mv_generator = MultiViewEpipolarGenerator(ldm_stable, camera_poses, intrinsics)
    
    # Generate novel views
    print("Generating novel views with epipolar attention...")
    reference_pose = camera_poses[0]
    
    # Save reference view
    ref_image = latent_to_image(ldm_stable.vae, reference_latent)
    save_image(ref_image, os.path.join(OUTPUT_DIR, "reference_view.png"))
    
    # Generate novel views
    for i, target_pose in enumerate(camera_poses[1:], 1):
        print(f"Generating view {i}/{len(camera_poses)-1}")
        
        novel_view_latent = mv_generator.generate_novel_view(
            reference_latent, reference_pose, target_pose, prompt
        )
        
        # Convert and save image
        novel_image = latent_to_image(ldm_stable.vae, novel_view_latent)
        save_image(novel_image, os.path.join(OUTPUT_DIR, f"novel_view_{i:02d}.png"))
        
        # Save latent
        torch.save(novel_view_latent.cpu(), os.path.join(OUTPUT_DIR, f"novel_view_latent_{i:02d}.pt"))
    
    print(f"Multi-view generation complete! Results saved in: {OUTPUT_DIR}")
