"""
lora/train_lora.py
─────────────────────────────────────────────────────────────────────────────
LoRA fine-tuning trainer for SDXL and SD2.

Implements:
  • Low-rank adaptor injection into UNet attention layers
  • Gradient checkpointing and mixed precision
  • DreamBooth-style instance + class prompt dataset
  • Cosine LR schedule with warmup

Usage:
    python -m lora.train_lora --config_preset style

References
----------
Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
Ruiz et al. 2023 — "DreamBooth: Fine Tuning Text-to-Image Diffusion Models"
    https://arxiv.org/abs/2208.12242
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import LoRAConfig, SDXL_MODEL_ID, SD2_MODEL_ID
from pipeline import get_device

logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class StyleDataset(Dataset):
    """
    Simple image dataset for LoRA style fine-tuning.
    Loads all images from a directory and pairs them with an instance prompt.

    Args:
        image_dir:       Directory containing training images.
        instance_prompt: Prompt associated with the style (e.g. "a photo of sks style").
        size:            Training resolution.
    """

    def __init__(
        self,
        image_dir:       str,
        instance_prompt: str,
        size:            int = 1024,
    ):
        self.image_dir       = Path(image_dir)
        self.instance_prompt = instance_prompt
        self.size            = size

        self.paths = [
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in EXTENSIONS
        ]
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")
        logger.info(f"StyleDataset: {len(self.paths)} images from {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "prompt":       self.instance_prompt,
        }


# ─── LoRA layer ───────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptor.

        y = W₀x + (α/r) * BAx

    where W₀ is frozen, B ∈ R^{d×r}, A ∈ R^{r×k} are trained.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank:   int   = 16,
        alpha:  int   = 32,
    ):
        super().__init__()
        d, k        = linear.out_features, linear.in_features
        self.weight = linear.weight        # frozen original weight
        self.bias   = linear.bias

        self.lora_A = nn.Parameter(torch.randn(rank, k) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d, rank))
        self.scale  = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + lora


# ─── LoRA injection ───────────────────────────────────────────────────────────

def inject_lora(
    unet:           nn.Module,
    rank:           int         = 16,
    alpha:          int         = 32,
    target_modules: list[str]   = ("to_q", "to_k", "to_v", "to_out.0"),
) -> list[nn.Parameter]:
    """
    Replace target Linear layers in UNet with LoRALinear modules.
    Freezes all original weights; only LoRA A/B matrices are trainable.

    Returns:
        List of trainable LoRA parameters.
    """
    # First freeze all params
    for p in unet.parameters():
        p.requires_grad_(False)

    lora_params = []
    replaced    = 0

    for parent_name, parent_module in list(unet.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if (
                isinstance(child_module, nn.Linear)
                and any(t in child_name for t in target_modules)
            ):
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha)
                setattr(parent_module, child_name, lora_layer)
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
                replaced += 1

    logger.info(f"LoRA injected into {replaced} Linear layers (rank={rank}, alpha={alpha})")
    for p in lora_params:
        p.requires_grad_(True)
    return lora_params


# ─── Trainer ──────────────────────────────────────────────────────────────────

class LoRATrainer:
    """
    LoRA fine-tuning trainer.

    Args:
        cfg: LoRAConfig with all hyperparameters.
    """

    def __init__(self, cfg: LoRAConfig):
        self.cfg    = cfg
        self.device = get_device()

    def _load_components(self):
        """Load tokeniser, text encoder, VAE, UNet, scheduler."""
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionPipeline,
            DDPMScheduler,
        )
        cfg  = self.cfg
        dtype = torch.float32   # always train in fp32, cast outputs in fp16

        if cfg.model == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_MODEL_ID, torch_dtype=dtype
            )
            self.tokenizer  = pipe.tokenizer
            self.tokenizer2 = pipe.tokenizer_2
            self.text_enc   = pipe.text_encoder.to(self.device)
            self.text_enc2  = pipe.text_encoder_2.to(self.device)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                SD2_MODEL_ID, torch_dtype=dtype
            )
            self.tokenizer  = pipe.tokenizer
            self.tokenizer2 = None
            self.text_enc   = pipe.text_encoder.to(self.device)
            self.text_enc2  = None

        self.vae       = pipe.vae.to(self.device)
        self.unet      = pipe.unet.to(self.device)
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Freeze VAE and text encoders
        for m in [self.vae, self.text_enc]:
            for p in m.parameters():
                p.requires_grad_(False)
        if self.text_enc2:
            for p in self.text_enc2.parameters():
                p.requires_grad_(False)

        logger.info("All base components loaded and frozen.")

    def _encode_prompt_batch(self, prompts: list[str]) -> torch.Tensor:
        """Encode a list of prompts to embeddings."""
        ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_enc(ids)[0]                # [B, L, D]

    def train(self) -> Path:
        """
        Run the full LoRA training loop.

        Returns:
            Path to the saved LoRA weights directory.
        """
        cfg = self.cfg
        self._load_components()

        # Inject LoRA
        lora_params = inject_lora(
            self.unet,
            rank           = cfg.lora_rank,
            alpha          = cfg.lora_alpha,
            target_modules = cfg.target_modules,
        )

        # Enable gradient checkpointing to save VRAM
        self.unet.enable_gradient_checkpointing()

        # Dataset & dataloader
        dataset = StyleDataset(
            cfg.dataset_dir,
            cfg.instance_prompt,
            size=1024 if cfg.model == "sdxl" else 768,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # Optimiser & LR scheduler
        optimiser = torch.optim.AdamW(lora_params, lr=cfg.learning_rate)
        total_steps = cfg.max_train_steps

        def _lr_lambda(step):
            # Linear warmup then cosine decay
            if step < cfg.lr_warmup_steps:
                return step / cfg.lr_warmup_steps
            progress = (step - cfg.lr_warmup_steps) / max(total_steps - cfg.lr_warmup_steps, 1)
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimiser, _lr_lambda)

        scaler  = torch.cuda.amp.GradScaler(enabled=(cfg.mixed_precision == "fp16"))
        step    = 0
        accum   = cfg.gradient_accumulation_steps

        logger.info(f"Starting LoRA training for {total_steps} steps …")

        self.unet.train()

        while step < total_steps:
            for batch in loader:
                if step >= total_steps:
                    break

                pixels  = batch["pixel_values"].to(self.device)  # [B,3,H,W]
                prompts = batch["prompt"]

                use_amp = (
                    cfg.mixed_precision in ("fp16", "bf16")
                    and self.device.type == "cuda"
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16,
                    enabled=use_amp,
                ):
                    # ── Encode images to latents ──
                    # SDXL's VAE scaling factor is 0.13025, not the 0.18215 that
                    # SD1.x/SD2 use — read it off the model rather than hardcoding.
                    latents = self.vae.encode(pixels).latent_dist.sample() * getattr(
                        self.vae.config, "scaling_factor", 0.18215
                    )

                    # ── Sample noise and timestep ──
                    noise = torch.randn_like(latents)
                    t     = torch.randint(
                        0, self.scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=self.device, dtype=torch.long
                    )
                    z_t = self.scheduler.add_noise(latents, noise, t)

                    # ── Text conditioning ──
                    text_emb = self._encode_prompt_batch(prompts)

                    # ── UNet forward ──
                    pred = self.unet(z_t, t, encoder_hidden_states=text_emb).sample

                    # ── Loss (epsilon prediction) ──
                    loss = F.mse_loss(pred, noise) / accum

                scaler.scale(loss).backward()

                if (step + 1) % accum == 0:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    lr_sched.step()
                    optimiser.zero_grad()

                if (step + 1) % 50 == 0:
                    logger.info(
                        f"Step {step+1:04d}/{total_steps} | "
                        f"loss={loss.item() * accum:.4f} | "
                        f"lr={lr_sched.get_last_lr()[0]:.2e}"
                    )

                if (step + 1) % cfg.save_every == 0:
                    self._save_lora(lora_params, step + 1)

                step += 1

        out_path = self._save_lora(lora_params, step)
        logger.info(f"Training complete. LoRA saved to {out_path}")
        return out_path

    def _save_lora(self, lora_params: list, step: int) -> Path:
        """Save LoRA A/B matrices as a .pt checkpoint."""
        out_dir = Path(self.cfg.output_dir) / f"step_{step:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {f"param_{i}": p.detach().cpu() for i, p in enumerate(lora_params)}
        torch.save(state, out_dir / "lora_weights.pt")
        logger.info(f"LoRA checkpoint → {out_dir}")
        return out_dir


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model",            default="sdxl")
    p.add_argument("--dataset_dir",      default="data/lora_train")
    p.add_argument("--instance_prompt",  default="a photo of sks style")
    p.add_argument("--rank",             type=int, default=16)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--steps",            type=int, default=1000)
    p.add_argument("--batch_size",       type=int, default=1)
    args = p.parse_args()

    cfg = LoRAConfig(
        model            = args.model,
        dataset_dir      = args.dataset_dir,
        instance_prompt  = args.instance_prompt,
        lora_rank        = args.rank,
        learning_rate    = args.lr,
        max_train_steps  = args.steps,
        train_batch_size = args.batch_size,
    )
    trainer = LoRATrainer(cfg)
    trainer.train()
