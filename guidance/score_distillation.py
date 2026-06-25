"""
guidance/score_distillation.py
─────────────────────────────────────────────────────────────────────────────
Score Distillation Sampling (SDS) and Variational Score Distillation (VSD).

Implements:
  • SDSLoss   — original SDS gradient (Poole et al., DreamFusion 2022)
  • VSDLoss   — improved VSD gradient (Wang et al., ProlificDreamer 2023)

References
----------
Poole et al. 2022 — "DreamFusion: Text-to-3D using 2D Diffusion"
    https://arxiv.org/abs/2209.14988
Wang et al. 2023 — "ProlificDreamer: High-Fidelity and Diverse Text-to-3D
    Generation with Variational Score Distillation"
    https://arxiv.org/abs/2305.16213
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ─── Noise-level sampling ─────────────────────────────────────────────────────

def sample_timestep(
    scheduler,
    batch_size: int,
    t_min: float = 0.02,
    t_max: float = 0.98,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Uniformly sample a timestep index in [t_min, t_max] * T.

    Args:
        scheduler:   Diffusers noise scheduler.
        batch_size:  Number of timesteps to sample.
        t_min/t_max: Fractional range to sample from.
    """
    T    = scheduler.config.num_train_timesteps
    low  = int(t_min * T)
    high = int(t_max * T)
    t    = torch.randint(low, high, (batch_size,), device=device)
    return t


# ─── SDS Loss ─────────────────────────────────────────────────────────────────

class SDSLoss:
    """
    Score Distillation Sampling loss.

    Given a differentiable renderer that produces images x = g(θ),
    SDS distils the diffusion prior into θ by following the gradient:

        ∇_θ L_SDS ≈ E_{t,ε}[w(t) (ε_φ(z_t; y, t) − ε) ∂x/∂θ]

    where z_t = αx + σε is the noised latent and ε_φ is the UNet prediction.

    In practice this is used to optimise a 2-D texture or a NeRF parameter set.

    Args:
        unet:            The diffusion UNet (frozen).
        scheduler:       Noise scheduler.
        vae:             VAE encoder (to convert pixel images → latents).
        text_embeddings: Concatenated [uncond, cond] embeddings [2B, L, D].
        guidance_scale:  CFG weight (high values work better for SDS, e.g. 100).
        t_min/t_max:     Noise level range.
        device:          Compute device.
    """

    def __init__(
        self,
        unet,
        scheduler,
        vae,
        text_embeddings: Tensor,
        guidance_scale:  float          = 100.0,
        t_min:           float          = 0.02,
        t_max:           float          = 0.98,
        device:          torch.device   = torch.device("cuda"),
        vae_scale_factor: float         = 0.18215,
    ):
        self.unet              = unet
        self.scheduler         = scheduler
        self.vae               = vae
        self.text_embeddings   = text_embeddings.to(device)
        self.guidance_scale    = guidance_scale
        self.t_min             = t_min
        self.t_max             = t_max
        self.device            = device
        self.vae_scale_factor  = vae_scale_factor

        # Freeze diffusion model
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_image(self, image: Tensor) -> Tensor:
        """Encode image [B,3,H,W] ∈ [-1,1] into latent [B,4,H/8,W/8]."""
        dist   = self.vae.encode(image).latent_dist
        latent = dist.sample() * self.vae_scale_factor
        return latent

    def __call__(self, image: Tensor) -> tuple[Tensor, dict]:
        """
        Compute SDS gradient and loss.

        Args:
            image: Rendered image [B, 3, H, W] with values in [-1, 1].
                   Must have `requires_grad=True` or be produced by an
                   autograd-compatible renderer.
        Returns:
            (loss, info_dict)
        """
        B = image.shape[0]

        # ── 1. Encode to latent space ──
        latents = self._encode_image(image)          # [B,4,h,w]

        # ── 2. Sample noise and timestep ──
        noise = torch.randn_like(latents)
        t     = sample_timestep(
            self.scheduler, B,
            self.t_min, self.t_max,
            device=self.device,
        )

        # ── 3. Forward diffusion: z_t ──
        z_t = self.scheduler.add_noise(latents, noise, t)   # [B,4,h,w]

        # ── 4. UNet noise prediction with CFG ──
        z_t_in   = torch.cat([z_t] * 2)                     # [2B,4,h,w]
        t_in     = torch.cat([t]   * 2)
        emb_in   = self.text_embeddings.expand(2 * B, -1, -1)

        with torch.no_grad():
            noise_pred = self.unet(z_t_in, t_in, encoder_hidden_states=emb_in).sample

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_guided = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)

        # ── 5. SDS gradient (stop-grad on model output) ──
        # w(t) = σ_t²  (SNR weighting)
        alphas     = self.scheduler.alphas_cumprod.to(self.device)
        alpha_t    = alphas[t][:, None, None, None]          # [B,1,1,1]
        sigma_t    = (1.0 - alpha_t).sqrt()
        w_t        = sigma_t ** 2                             # SNR weight

        grad       = w_t * (noise_guided - noise)            # [B,4,h,w]
        grad        = grad.detach()                          # stop gradient through model

        # Reconstruct scalar loss so autograd flows through `latents`
        loss = (grad * latents).sum()

        info = {
            "t_mean":     t.float().mean().item(),
            "grad_norm":  grad.norm().item(),
            "w_t_mean":   w_t.mean().item(),
        }
        return loss, info


# ─── VSD Loss ─────────────────────────────────────────────────────────────────

class VSDLoss(SDSLoss):
    """
    Variational Score Distillation (ProlificDreamer).

    Extends SDS by maintaining a separate "phi" LoRA adaptor on the UNet
    that models the distribution of rendered images.  The VSD gradient is:

        ∇_θ L_VSD ≈ E[w(t) (ε_φ_pretrained(z_t; y, t) − ε_φ_lora(z_t; c, t)) ∂x/∂θ]

    The phi network is updated with a score-matching objective each step.

    Args:
        phi_lr:    Learning rate for the LoRA phi network.
        phi_steps: UNet phi update steps per outer optimisation step.
    """

    def __init__(
        self,
        *args,
        phi_lr:    float = 1e-4,
        phi_steps: int   = 1,
        lora_rank: int   = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.phi_steps = phi_steps

        # Build a minimal LoRA on top of the frozen UNet
        self.lora_params = self._init_lora(lora_rank)
        self.phi_opt     = torch.optim.AdamW(self.lora_params, lr=phi_lr)

    def _init_lora(self, rank: int) -> list[nn.Parameter]:
        """
        Attach low-rank matrices to the UNet attention projections.
        Returns list of trainable parameters.
        """
        params = []
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Linear) and any(
                k in name for k in ("to_q", "to_k", "to_v", "to_out")
            ):
                d_in, d_out = module.in_features, module.out_features
                A = nn.Parameter(torch.randn(d_in,  rank, device=self.device) * 0.01)
                B = nn.Parameter(torch.zeros(rank, d_out, device=self.device))
                module._lora_A   = A
                module._lora_B   = B
                module._lora_on  = True
                params.extend([A, B])

                # Monkey-patch the forward to add LoRA delta
                orig_fwd = module.forward
                def _lora_fwd(x, _orig=orig_fwd, _m=module):
                    out = _orig(x)
                    if getattr(_m, "_lora_on", False):
                        delta = (x @ _m._lora_A) @ _m._lora_B
                        out   = out + delta
                    return out

                module.forward = _lora_fwd

        logger.info(f"VSD: attached LoRA to {len(params)//2} linear layers (rank={rank})")
        return params

    def _phi_forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        """Run UNet with LoRA active (phi distribution)."""
        for module in self.unet.modules():
            if hasattr(module, "_lora_on"):
                module._lora_on = True
        with torch.enable_grad():
            pred = self.unet(
                z_t,
                t,
                encoder_hidden_states=self.text_embeddings[:z_t.shape[0]],
            ).sample
        return pred

    def _update_phi(self, z_t: Tensor, noise: Tensor, t: Tensor):
        """Score-match the phi distribution to the rendered latents."""
        for _ in range(self.phi_steps):
            self.phi_opt.zero_grad()
            noise_pred = self._phi_forward(z_t, t)
            loss       = F.mse_loss(noise_pred, noise)
            loss.backward()
            self.phi_opt.step()

    def __call__(self, image: Tensor) -> tuple[Tensor, dict]:
        B       = image.shape[0]
        latents = self._encode_image(image)
        noise   = torch.randn_like(latents)
        t       = sample_timestep(self.scheduler, B, self.t_min, self.t_max, self.device)
        z_t     = self.scheduler.add_noise(latents, noise, t)

        # ── Pretrained score (frozen) ──
        z_t_in = torch.cat([z_t] * 2)
        t_in   = torch.cat([t]   * 2)
        emb_in = self.text_embeddings.expand(2 * B, -1, -1)
        with torch.no_grad():
            for m in self.unet.modules():
                if hasattr(m, "_lora_on"):
                    m._lora_on = False
            noise_pred  = self.unet(z_t_in, t_in, encoder_hidden_states=emb_in).sample
        noise_u, noise_c = noise_pred.chunk(2)
        noise_pretrained = noise_u + self.guidance_scale * (noise_c - noise_u)

        # ── Phi (LoRA) score ──
        self._update_phi(z_t.detach(), noise, t)
        with torch.no_grad():
            noise_phi = self._phi_forward(z_t, t)

        # ── VSD gradient ──
        alphas   = self.scheduler.alphas_cumprod.to(self.device)
        alpha_t  = alphas[t][:, None, None, None]
        sigma_t  = (1.0 - alpha_t).sqrt()
        w_t      = sigma_t ** 2

        grad     = w_t * (noise_pretrained - noise_phi)
        grad     = grad.detach()
        loss     = (grad * latents).sum()

        info = {
            "t_mean":    t.float().mean().item(),
            "grad_norm": grad.norm().item(),
        }
        return loss, info


# ─── Texture optimisation loop ─────────────────────────────────────────────────

def run_sds_optimisation(
    sds_loss:    Union[SDSLoss, VSDLoss],
    texture:     nn.Parameter,
    renderer,                                   # callable: texture → image [B,3,H,W]
    num_iter:    int           = 500,
    lr:          float         = 1e-2,
    log_every:   int           = 50,
    save_every:  int           = 100,
    output_dir   = None,
) -> nn.Parameter:
    """
    Main SDS/VSD optimisation loop.

    Args:
        sds_loss:  SDSLoss or VSDLoss instance.
        texture:   Learnable parameter (e.g. a [1,3,H,W] image or NeRF weights).
        renderer:  Function mapping texture → rendered image tensor.
        num_iter:  Number of gradient steps.
        lr:        Adam learning rate.
        log_every: Print loss every N steps.
        save_every:Save intermediate texture every N steps.
        output_dir:Path to save intermediate PNGs.
    """
    from pathlib import Path
    from torchvision.utils import save_image

    optimiser = torch.optim.Adam([texture], lr=lr)
    output_dir = Path(output_dir) if output_dir else Path("outputs/sds")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_iter):
        optimiser.zero_grad()

        # Render image from current texture
        image = renderer(texture)                # [B,3,H,W] in [-1,1]

        loss, info = sds_loss(image)
        loss.backward()
        optimiser.step()

        # Clamp texture to valid range
        with torch.no_grad():
            texture.clamp_(-1.0, 1.0)

        if (i + 1) % log_every == 0:
            logger.info(
                f"[SDS] step {i+1:04d}/{num_iter} | "
                f"loss={loss.item():.4f} | "
                f"t={info['t_mean']:.1f} | "
                f"grad_norm={info['grad_norm']:.4f}"
            )

        if (i + 1) % save_every == 0 and output_dir:
            img_out = (texture.detach() * 0.5 + 0.5).clamp(0, 1)
            save_image(img_out, output_dir / f"sds_step_{i+1:04d}.png")

    return texture
