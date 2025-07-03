import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict
from torchvision import transforms
from PIL import Image
import os

def train_lora_on_image(image_path, prompt, output_dir,
                         base_model_id="runwayml/stable-diffusion-v1-5",
                         device="cuda", lr=1e-4, num_train_steps=500):
    """
    Train a LoRA adapter on a single image for a given prompt.
    Automatically selects cross-attention modules in the UNet.
    """
    # Load base diffusion model and scheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Detect UNet cross-attention submodules
    target_modules = [
        name for name, module in pipe.unet.named_modules()
        if any(tok in name for tok in ("to_q", "to_k", "to_v"))
    ]
    if not target_modules:
        raise ValueError("No cross-attention modules found: check your UNet structure.")

    # Prepare PEFT LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="SCHEDULER"
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_t = preprocess(image).unsqueeze(0).to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=lr)

    # Training loop
    for step in range(num_train_steps):
        # Sample noise and timestep
        noise = torch.randn_like(image_t)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps,
            (1,), device=device
        )

        # Encode latents and add noise
        latents = pipe.vae.encode(image_t * 2 - 1).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise residual
        noise_pred = pipe.unet(
            noisy_latents, timesteps,
            encoder_hidden_states=pipe._encode_prompt(prompt)
        )[0]

        # Compute and backprop loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}/{num_train_steps}, loss: {loss.item():.4f}")

    # Save LoRA adapter weights
    os.makedirs(output_dir, exist_ok=True)
    pipe.unet.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")
    return output_dir


def interpolate_lora_and_generate(prompt, lora_dirs, output_dir,
                                  num_frames=30, seed=42,
                                  base_model_id="runwayml/stable-diffusion-v1-5",
                                  device="cuda"):
    """
    Interpolate between two LoRA adapters and generate frames.
    lora_dirs: list of two LoRA adapter dirs.
    """
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load LoRA states
    states = [torch.load(os.path.join(d, "pytorch_model.bin"), map_location=device)
              for d in lora_dirs]

    # Identify target modules
    target_modules = [
        name for name, module in pipe.unet.named_modules()
        if any(tok in name for tok in ("to_q", "to_k", "to_v"))
    ]
    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1, bias="none",
        task_type="SCHEDULER"
    )

    os.makedirs(output_dir, exist_ok=True)
    generator = torch.Generator(device).manual_seed(seed)

    for i, alpha in enumerate(torch.linspace(0, 1, num_frames)):
        # Interpolate state dict
        interp = {
            k: (1 - alpha) * states[0][k] + alpha * states[1][k]
            for k in states[0]
        }
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        set_peft_model_state_dict(pipe.unet, interp)

        # Generate & save frame
        img = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
        print(f"Saved frame {i+1}/{num_frames}")

    print(f"All frames saved to {output_dir}")


if __name__ == "__main__":
    prompt = "A serene landscape with mountains and a lake"
    # Train LoRAs
    loraA = train_lora_on_image("path/to/imageA.png", prompt, "loras/loraA")
    loraB = train_lora_on_image("path/to/imageB.png", prompt, "loras/loraB")

    # Interpolate & generate
    interpolate_lora_and_generate(
        prompt, [loraA, loraB], "outputs/morph_frames",
        num_frames=60, seed=1234
    )
