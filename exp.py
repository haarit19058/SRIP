import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# -------- Epipolar Geometry Helpers --------
def compute_epipolar_line(K, R, t, pixel_coords):
    ones = torch.ones(pixel_coords.shape[0], 1, device=pixel_coords.device)
    pts = torch.cat([pixel_coords, ones], dim=-1)
    invK = torch.inverse(K)
    norm_pts = (invK @ pts.T).T
    t_x = torch.tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], device=t.device)
    E = R @ t_x
    return (E @ norm_pts.T).T

def sample_points_on_line(line, H, W, num_samples=32):
    a, b, c = line
    u = torch.linspace(0, W-1, num_samples, device=line.device)
    v = -(a*u + c) / (b + 1e-8)
    mask = (v >= 0) & (v < H)
    u_norm = u[mask]/(W-1)*2 - 1
    v_norm = v[mask]/(H-1)*2 - 1
    return torch.stack([u_norm, v_norm], dim=-1)

# -------- Epipolar Attention Processor --------
class EpipolarAttention(AttnProcessor):
    def __init__(self, base, K, R, t, alpha=0.5, num_samples=32):
        super().__init__()
        for attr in ['to_q','to_k','to_v','to_out']:
            layer = getattr(base, attr)
            new = nn.Linear(layer.in_features, layer.out_features).to(layer.weight.device)
            new.load_state_dict(layer.state_dict())
            setattr(self, attr, new)
        self.scale = base.scale
        self.K, self.R, self.t = K, R, t
        self.alpha, self.num_samples = alpha, num_samples

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        B, N, C = hidden_states.shape
        H = W = int(N**0.5)
        Q = self.to_q(hidden_states)
        Kf = self.to_k(hidden_states)
        Vf = self.to_v(hidden_states)
        coords = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H)), dim=-1).reshape(-1,2).to(hidden_states.device)
        lines = compute_epipolar_line(self.K, self.R, self.t, coords.float())
        feat_map = hidden_states.view(B,H,W,C).permute(0,3,1,2)
        ref_feats = []
        for b in range(B):
            per_b = []
            for i, line in enumerate(lines):
                samples = sample_points_on_line(line, H, W, self.num_samples)
                if samples.numel() == 0:
                    per_b.append(torch.zeros(C, device=hidden_states.device))
                    continue
                grid = samples.view(1,-1,1,2)
                sampled = F.grid_sample(feat_map[b:b+1], grid, align_corners=True)
                sampled = sampled.view(C,-1).T
                q_vec = Q[b,i]
                sims = (sampled @ q_vec)/self.scale
                w = torch.softmax(sims, dim=0)
                per_b.append((w.unsqueeze(1)*sampled).sum(dim=0))
            ref_feats.append(torch.stack(per_b))
        ref_feats = torch.stack(ref_feats).view(B,N,C)
        scores = torch.softmax((Q @ Kf.transpose(-2,-1))*self.scale, dim=-1)
        std_out = scores @ Vf
        out = self.alpha*ref_feats + (1-self.alpha)*std_out
        return self.to_out(out)

# -------- Patch UNet --------
def patch_unet(pipe, K, R, t, alpha=0.5, num_samples=32, layers=None):
    if layers is None:
        layers = ['mid_block.attentions.0']
    for name, proc in pipe.unet.attn_processors.items():
        if name in layers:
            pipe.unet.attn_processors[name] = EpipolarAttention(proc, K, R, t, alpha, num_samples)
    return pipe

# -------- Novel View Synthesis --------
def generate_novel_view(init_img_path, angle_deg, baseline=0.1,
                        model_id='runwayml/stable-diffusion-v1-5',
                        output_path='novel_view.png', num_inference_steps=50):
    device = 'cuda'
    # Load and preprocess image
    img = Image.open(init_img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device).half()

    # Load pipeline and configure scheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    pipe.scheduler = scheduler
    pipe.enable_attention_slicing()

    with torch.no_grad():
        # Encode
        latents = pipe.vae.encode(tensor * 2 - 1).latent_dist.sample() * 0.18215

        # Decode
        image_tensor = pipe.vae.decode(latents / 0.18215).sample  # shape: [B, C, H, W]

    # Convert to [0, 1] and move to CPU
    image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2.0
    image_tensor = image_tensor[0].cpu()  # get first image in batch

    # Convert to PIL and display
    img_pil = to_pil_image(image_tensor)
    plt.imshow(img_pil)
    plt.axis('off')
    plt.savefig('temp.png')

    # return
    # Camera geometry
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device))
    K = torch.eye(3, device=device)
    R = torch.tensor([
        [torch.cos(theta), 0, torch.sin(theta)],
        [0, 1, 0],
        [-torch.sin(theta), 0, torch.cos(theta)]
    ], device=device)
    t = torch.tensor([baseline, 0, 0], device=device)

    # Patch UNet for epipolar
    pipe = patch_unet(pipe, K, R, t, alpha=0.7, num_samples=64)


if __name__ == '__main__':
    generate_novel_view('./flow/IMG_4026.JPG', angle_deg=30, baseline=0.1)
