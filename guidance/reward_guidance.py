"""
guidance/reward_guidance.py
─────────────────────────────────────────────────────────────────────────────
Inference-time reward guidance using CLIP-based aesthetic scoring.

Implements:
  • CLIPAestheticReward — lightweight MLP on top of CLIP vision features
                          trained to predict human aesthetic scores
  • LatentOptimiser     — guides diffusion by back-propagating reward
                          gradient through the VAE decoder into the latent

References
----------
Black et al. 2023 — "Training Diffusion Models with RL (DDPO)"
    https://arxiv.org/abs/2305.13301
Clark et al. 2023 — "Directly Fine-Tuning Diffusion Models on Differentiable Rewards"
    https://arxiv.org/abs/2309.12407
Schuhmann et al. 2022 — "LAION-Aesthetics"
    https://laion.ai/blog/laion-aesthetics/
"""

from __future__ import annotations

import logging
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

logger = logging.getLogger(__name__)


# ─── CLIP aesthetic reward model ──────────────────────────────────────────────

class AestheticMLP(nn.Module):
    """
    Tiny MLP that maps a CLIP embedding → scalar aesthetic score.
    Architecture mirrors the LAION aesthetic predictor (Schuhmann et al., 2022).
    """

    def __init__(self, clip_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Normalise the CLIP embedding before scoring
        x = F.normalize(x, dim=-1)
        return self.net(x).squeeze(-1)            # [B]


class CLIPAestheticReward:
    """
    Differentiable aesthetic reward using CLIP + a trained MLP head.

    The MLP weights can be loaded from the public LAION checkpoint or
    trained from scratch on freely-available data (no paid data).

    Args:
        clip_model_name: HuggingFace CLIP model name.
        mlp_ckpt_path:   Path to pretrained MLP weights (optional).
        device:          Compute device.
    """

    CLIP_PREPROCESS = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    def __init__(
        self,
        clip_model_name: str                 = "openai/clip-vit-large-patch14",
        mlp_ckpt_path:   Optional[str]       = None,
        device:          torch.device        = torch.device("cuda"),
    ):
        self.device = device

        # ── Load CLIP ──
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip      = CLIPModel.from_pretrained(clip_model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
            clip_dim       = self.clip.config.vision_config.hidden_size
            logger.info(f"Loaded CLIP ({clip_model_name}), dim={clip_dim}")
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}. Reward will return zeros.")
            self.clip    = None
            self.processor = None
            clip_dim     = 768

        # Freeze CLIP
        if self.clip is not None:
            for p in self.clip.parameters():
                p.requires_grad_(False)

        # ── Load MLP ──
        self.mlp = AestheticMLP(clip_dim).to(device)
        if mlp_ckpt_path:
            state = torch.load(mlp_ckpt_path, map_location=device)
            self.mlp.load_state_dict(state)
            logger.info(f"Loaded aesthetic MLP from {mlp_ckpt_path}")
        else:
            logger.info("No MLP checkpoint provided — using random aesthetic weights.")

    @torch.no_grad()
    def _get_clip_features(self, images: Tensor) -> Tensor:
        """
        Extract CLIP image features.

        Args:
            images: [B, 3, H, W] normalised to [0, 1] or [-1, 1].
        Returns:
            features: [B, clip_dim]
        """
        if self.clip is None:
            return torch.zeros(images.shape[0], 768, device=self.device)

        # Rescale from [-1,1] to [0,1] if needed
        if images.min() < -0.1:
            images = (images + 1.0) / 2.0

        images = self.CLIP_PREPROCESS(images)
        feats  = self.clip.get_image_features(pixel_values=images)
        return feats

    def score(self, images: Tensor) -> Tensor:
        """
        Compute aesthetic score for a batch of images.

        Args:
            images: [B, 3, H, W] float tensor.
        Returns:
            scores: [B] scalar scores ∈ (−∞, +∞), higher = better.
        """
        feats = self._get_clip_features(images)
        return self.mlp(feats)

    def reward(self, images: Tensor) -> Tensor:
        """Alias for score — for use as a reward signal."""
        return self.score(images)


# ─── Latent optimiser ─────────────────────────────────────────────────────────

class LatentOptimiser:
    """
    Inference-time latent optimisation guided by a differentiable reward.

    After standard diffusion denoising, we take a few gradient steps
    directly on the final latent z₀ to maximise reward(decode(z₀)).

    This is lightweight (only a handful of gradient steps, no UNet involved)
    and can noticeably improve aesthetic quality.

    Args:
        vae:             VAE decoder (from the diffusion pipeline).
        reward_fn:       Callable: image [B,3,H,W] → scalar reward [B].
        num_steps:       Number of latent gradient steps.
        lr:              Adam learning rate on the latent.
        reward_scale:    Weight of reward gradient vs. identity regularisation.
        vae_scale:       VAE latent scale factor (0.18215 for SD/SDXL).
        device:          Compute device.
    """

    def __init__(
        self,
        vae,
        reward_fn:    Callable[[Tensor], Tensor],
        num_steps:    int           = 5,
        lr:           float         = 0.05,
        reward_scale: float         = 0.15,
        vae_scale:    float         = 0.18215,
        device:       torch.device  = torch.device("cuda"),
    ):
        self.vae          = vae
        self.reward_fn    = reward_fn
        self.num_steps    = num_steps
        self.lr           = lr
        self.reward_scale = reward_scale
        self.vae_scale    = vae_scale
        self.device       = device

        # Keep decoder gradients enabled but freeze encoder/other parts
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def _decode(self, latent: Tensor) -> Tensor:
        """Decode latent [B,4,h,w] → image [B,3,H,W] ∈ [-1,1]."""
        return self.vae.decode(latent / self.vae_scale).sample

    def optimise(self, latent: Tensor) -> tuple[Tensor, list[float]]:
        """
        Optimise latent z to maximise reward.

        Args:
            latent: Initial latent from diffusion denoising [B,4,h,w].
        Returns:
            (optimised_latent, reward_history)
        """
        z     = latent.detach().clone().requires_grad_(True)
        opt   = torch.optim.Adam([z], lr=self.lr)
        z_ref = latent.detach().clone()   # regularisation anchor

        rewards = []

        for step in range(self.num_steps):
            opt.zero_grad()

            # Decode to pixel space
            image = self._decode(z)

            # Reward (higher = better, so we minimise -reward)
            r     = self.reward_fn(image)
            r_loss = -r.mean() * self.reward_scale

            # L2 regularisation to stay close to the denoised latent
            reg_loss = F.mse_loss(z, z_ref)

            loss = r_loss + reg_loss
            loss.backward()
            opt.step()

            rewards.append(r.mean().item())
            logger.debug(f"LatentOpt step {step+1}: reward={rewards[-1]:.4f}, reg={reg_loss.item():.4f}")

        return z.detach(), rewards


# ─── Reward-guided generation ─────────────────────────────────────────────────

def reward_guided_generation(
    pipe,
    reward_model:   CLIPAestheticReward,
    latent_opt:     LatentOptimiser,
    prompt:         str,
    negative_prompt: str           = "",
    num_inference_steps: int       = 50,
    guidance_scale:  float         = 7.5,
    seed:            Optional[int] = None,
    output_dir                     = None,
):
    """
    Full pipeline: diffusion denoising → latent reward optimisation → decode.

    Returns the optimised PIL image.
    """
    from PIL import Image
    from pathlib import Path
    import numpy as np

    device    = next(pipe.base.unet.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    # ── Step 1: run base diffusion (get latents) ──
    out = pipe.base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
    )
    latent = out.images                           # [1,4,h,w] float

    # ── Step 2: latent reward optimisation ──
    latent_opt_result, reward_history = latent_opt.optimise(latent)
    logger.info(f"Reward trajectory: {[f'{r:.3f}' for r in reward_history]}")

    # ── Step 3: decode ──
    with torch.no_grad():
        image_tensor = pipe.base.vae.decode(
            latent_opt_result / 0.18215
        ).sample                                  # [1,3,H,W] in [-1,1]

    image_np = ((image_tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype("uint8")
    pil_image = Image.fromarray(image_np)

    if output_dir:
        out_path = Path(output_dir) / "reward_guided.png"
        pil_image.save(out_path)
        logger.info(f"Saved → {out_path}")

    return pil_image, reward_history
