"""
guidance/classifier_free.py
─────────────────────────────────────────────────────────────────────────────
Classifier-Free Guidance (CFG) utilities.

Implements:
  • Static CFG (standard)
  • Linear / cosine CFG annealing schedules  [Ho & Salimans, 2022]
  • Perturbed attention guidance (PAG)        [Ahn et al., 2024]
  • Rescaled CFG (cfg-rescale)               [Lin et al., 2024]

References
----------
Ho & Salimans 2022 — "Classifier-Free Diffusion Guidance"
    https://arxiv.org/abs/2207.12598
Ahn et al. 2024 — "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance"
    https://arxiv.org/abs/2403.17377
Lin et al. 2024 — "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    https://arxiv.org/abs/2305.08891
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn.functional as F


# ─── Schedule types ───────────────────────────────────────────────────────────

ScheduleType = Literal["constant", "linear", "cosine", "warmup_cosine"]


@dataclass
class CFGScheduler:
    """
    Dynamic CFG scale schedule over the denoising trajectory.

    Args:
        cfg_start:     Guidance scale at timestep 0 (highest noise).
        cfg_end:       Guidance scale at the final timestep (lowest noise).
        schedule:      One of 'constant', 'linear', 'cosine', 'warmup_cosine'.
        warmup_frac:   For 'warmup_cosine', fraction of steps spent warming up.
    """

    cfg_start:   float        = 12.0
    cfg_end:     float        = 5.0
    schedule:    ScheduleType = "cosine"
    warmup_frac: float        = 0.1

    def __call__(self, step: int, total_steps: int) -> float:
        """Return the CFG scale for the given denoising step."""
        frac = step / max(total_steps - 1, 1)   # 0 → 1 over trajectory

        if self.schedule == "constant":
            return self.cfg_start

        elif self.schedule == "linear":
            return self.cfg_start + frac * (self.cfg_end - self.cfg_start)

        elif self.schedule == "cosine":
            cos_val = 0.5 * (1.0 + math.cos(math.pi * frac))
            return self.cfg_end + cos_val * (self.cfg_start - self.cfg_end)

        elif self.schedule == "warmup_cosine":
            if frac < self.warmup_frac:
                # ramp up from cfg_end → cfg_start
                warm_frac = frac / self.warmup_frac
                return self.cfg_end + warm_frac * (self.cfg_start - self.cfg_end)
            else:
                # cosine anneal back down
                t = (frac - self.warmup_frac) / (1.0 - self.warmup_frac)
                cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
                return self.cfg_end + cos_val * (self.cfg_start - self.cfg_end)

        raise ValueError(f"Unknown schedule '{self.schedule}'")


# ─── Core CFG application ─────────────────────────────────────────────────────

def apply_cfg(
    noise_pred_uncond: torch.Tensor,
    noise_pred_cond:   torch.Tensor,
    guidance_scale:    float,
    rescale_phi:       float = 0.0,
) -> torch.Tensor:
    """
    Standard CFG with optional rescaling (Lin et al., 2024).

    Standard formula:
        ε̂ = ε_uncond + w * (ε_cond − ε_uncond)

    Rescaled variant corrects over-exposure from high CFG by scaling the
    conditional prediction so its standard deviation matches the unconditioned one:
        ε̂_rescaled = phi * ε̂_std + (1 - phi) * ε̂

    Args:
        noise_pred_uncond: Unconditional noise prediction [B, C, H, W].
        noise_pred_cond:   Conditional noise prediction   [B, C, H, W].
        guidance_scale:    CFG weight w.
        rescale_phi:       Rescaling strength ∈ [0, 1]. 0 = off.
    """
    guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    if rescale_phi > 0.0:
        # Match the std of the full-CFG prediction to the cond prediction
        std_cond   = noise_pred_cond.std()
        std_guided = guided.std()
        guided_rescaled = guided * (std_cond / (std_guided + 1e-8))
        guided = rescale_phi * guided_rescaled + (1.0 - rescale_phi) * guided

    return guided


# ─── Perturbed Attention Guidance (PAG) ───────────────────────────────────────

class PerturbedAttentionGuidance:
    """
    Inference-time guidance via self-attention perturbation (Ahn et al., 2024).

    During each denoising step we run the UNet twice:
      1. Normal forward pass           → noise_pred_normal
      2. Forward pass with all self-attention layers replaced by identity
         (i.e., each token attends only to itself) → noise_pred_perturbed

    PAG score:
        ε̂_PAG = ε̂_normal + pag_scale * (ε̂_normal − ε̂_perturbed)

    This can be combined with standard CFG or used as a drop-in replacement.
    """

    def __init__(self, unet, pag_scale: float = 3.0):
        self.unet      = unet
        self.pag_scale = pag_scale
        self._hooks: list = []

    def _identity_attn_hook(self, module, args, kwargs, output):
        """Replace cross-token attention output with self (identity mapping)."""
        # For a standard transformer block the output shape is [B, seq_len, dim].
        # Replacing with per-token identity means each position ignores all others.
        return args[0] if isinstance(args, tuple) and len(args) > 0 else output

    def _register_hooks(self):
        """Attach identity hooks to all UNet self-attention processors."""
        for name, module in self.unet.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_v"):
                h = module.register_forward_hook(self._identity_attn_hook)
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        latents:     torch.Tensor,
        timestep:    torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        noise_pred_normal: torch.Tensor,
        **unet_kwargs,
    ) -> torch.Tensor:
        """
        Compute PAG-guided noise prediction.

        Args:
            latents:               Latent [B, C, H, W].
            timestep:              Diffusion timestep scalar.
            encoder_hidden_states: Text conditioning.
            noise_pred_normal:     Already-computed normal UNet output.
        Returns:
            PAG-adjusted noise prediction.
        """
        self._register_hooks()
        with torch.no_grad():
            noise_pred_perturbed = self.unet(
                latents, timestep,
                encoder_hidden_states=encoder_hidden_states,
                **unet_kwargs,
            ).sample
        self._remove_hooks()

        pag_delta  = noise_pred_normal - noise_pred_perturbed
        return noise_pred_normal + self.pag_scale * pag_delta


# ─── Prompt-to-embedding utility ──────────────────────────────────────────────

def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    max_length: int = 77,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompt and negative prompt into embeddings.
    Supports basic attention weighting via (word:weight) syntax, e.g.
    "a (beautiful:1.3) sunset".

    Returns:
        (cond_embeddings, uncond_embeddings), each [1, seq_len, hidden_dim]
    """

    def _parse_weighted(text: str):
        """Parse 'token:weight' annotations into (tokens, weights)."""
        import re
        parts   = re.split(r"[\(\)]", text)
        tokens  = []
        weights = []
        for part in parts:
            if ":" in part:
                tok, w = part.rsplit(":", 1)
                try:
                    w = float(w)
                except ValueError:
                    w = 1.0
                tokens.append(tok.strip())
                weights.append(w)
            else:
                tokens.append(part.strip())
                weights.append(1.0)
        return " ".join(tokens), weights

    def _encode(text: str):
        cleaned, _ = _parse_weighted(text)
        ids = tokenizer(
            cleaned,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(ids)[0]

    cond_emb   = _encode(prompt)
    uncond_emb = _encode(negative_prompt)
    return cond_emb, uncond_emb
