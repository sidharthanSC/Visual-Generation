"""
techniques/multidiffusion.py
─────────────────────────────────────────────────────────────────────────────
MultiDiffusion: tiled generation for arbitrarily large images.

Each denoising step is split across overlapping tiles.
Overlapping regions are averaged (weighted by a cosine window),
creating seamless, globally coherent large-resolution outputs.

Reference
---------
Bar-Tal et al. 2023 — "MultiDiffusion: Fusing Diffusion Paths for
    Controlled Image Generation"
    https://arxiv.org/abs/2302.08113
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from pipeline import decode_latents

logger = logging.getLogger(__name__)


# ─── Cosine tile weight ───────────────────────────────────────────────────────

def cosine_tile_weight(h: int, w: int, device: torch.device) -> Tensor:
    """
    2-D cosine window for soft tile blending.
    Values peak at the tile centre and fall off toward edges.
    Shape: [1, 1, h, w]
    """
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    weight = wy[:, None] * wx[None, :]          # [h, w]
    return weight[None, None]                   # [1, 1, h, w]


# ─── MultiDiffusion core ──────────────────────────────────────────────────────

class MultiDiffusion:
    """
    MultiDiffusion wrapper.

    Supports SD2 and SDXL; uses the pipeline's UNet and scheduler directly
    to run a custom denoising loop with tiled latent averaging.

    Args:
        pipe:        SD2Pipeline or SDXLPipeline (from pipeline.py).
        tile_h:      Tile height in pixels (pre-VAE, so ×8 in latents).
        tile_w:      Tile width  in pixels.
        overlap:     Pixel overlap between adjacent tiles.
        stride_h:    Step between tile tops  (default: tile_h - overlap).
        stride_w:    Step between tile lefts (default: tile_w - overlap).
    """

    LATENT_FACTOR = 8   # VAE spatial downscale

    def __init__(
        self,
        pipe,
        tile_h:   int = 1024,
        tile_w:   int = 1024,
        overlap:  int = 256,
        stride_h: Optional[int] = None,
        stride_w: Optional[int] = None,
    ):
        self.pipe     = pipe
        self.tile_lh  = tile_h  // self.LATENT_FACTOR
        self.tile_lw  = tile_w  // self.LATENT_FACTOR
        self.overlap_l = overlap // self.LATENT_FACTOR
        self.stride_lh = stride_h // self.LATENT_FACTOR if stride_h else (self.tile_lh - self.overlap_l)
        self.stride_lw = stride_w // self.LATENT_FACTOR if stride_w else (self.tile_lw - self.overlap_l)

        # Determine device and dtype from the UNet
        unet_param   = next(pipe.get_unet().parameters())
        self.device  = unet_param.device
        self.dtype   = unet_param.dtype

    # ── Tile position generation ──────────────────────────────────────────────

    def _tile_positions(self, lh: int, lw: int) -> list[tuple[int, int, int, int]]:
        """
        Return (y0, y1, x0, x1) for every tile in the latent grid.
        Tiles are padded at the right/bottom edge to stay within bounds.
        """
        positions = []
        y = 0
        while True:
            y1 = min(y + self.tile_lh, lh)
            y0 = max(0, y1 - self.tile_lh)
            x  = 0
            while True:
                x1 = min(x + self.tile_lw, lw)
                x0 = max(0, x1 - self.tile_lw)
                positions.append((y0, y1, x0, x1))
                if x1 >= lw:
                    break
                x += self.stride_lw
            if y1 >= lh:
                break
            y += self.stride_lh
        return positions

    # ── Prompt encoding ───────────────────────────────────────────────────────

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> tuple[Tensor, ...]:
        """Encode prompts to embeddings; handles both SD2 and SDXL."""
        is_sdxl = hasattr(self.pipe, "base")
        if is_sdxl:
            return self.pipe.encode_prompt(prompt, negative_prompt)
        else:
            tok  = self.pipe.get_tokenizer()
            tenc = self.pipe.get_text_encoder()
            from guidance.classifier_free import get_weighted_text_embeddings
            c, u = get_weighted_text_embeddings(
                tok, tenc, prompt, negative_prompt, self.device
            )
            return c, u, None, None

    # ── Main generation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        prompt:              str,
        negative_prompt:     str   = "",
        canvas_h:            int   = 2048,
        canvas_w:            int   = 4096,
        num_inference_steps: int   = 50,
        guidance_scale:      float = 8.0,
        seed:                Optional[int] = 0,
    ) -> Image.Image:
        """
        Generate a large image via MultiDiffusion.

        Args:
            canvas_h/w:  Target pixel dimensions (should be multiples of 8).
            Other args:  Same as a standard diffusion call.
        Returns:
            PIL Image of size (canvas_w, canvas_h).
        """
        lh = canvas_h // self.LATENT_FACTOR
        lw = canvas_w // self.LATENT_FACTOR

        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None else None
        )

        # ── Encode prompts ──
        is_sdxl            = hasattr(self.pipe, "base")
        cond_emb, uncond_emb, pooled_cond, pooled_uncond = self._encode_prompt(
            prompt, negative_prompt
        )
        # Stack for CFG: [uncond, cond]
        text_embs = torch.cat([uncond_emb, cond_emb], dim=0)  # [2, L, D]

        # ── Initialise canvas latent ──
        latents = torch.randn(
            1, 4, lh, lw,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        # ── Setup scheduler ──
        scheduler = (
            self.pipe.get_scheduler()
            if not is_sdxl
            else self.pipe.base.scheduler
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        latents   = latents * scheduler.init_noise_sigma

        # ── Precompute tile positions and cosine weights ──
        positions = self._tile_positions(lh, lw)
        logger.info(
            f"MultiDiffusion: canvas {canvas_h}×{canvas_w} | "
            f"{len(positions)} tiles per step | "
            f"{num_inference_steps} steps"
        )

        # ── Denoising loop ──
        unet = self.pipe.get_unet() if not is_sdxl else self.pipe.base.unet

        for step_idx, t in enumerate(timesteps):
            noise_preds   = torch.zeros_like(latents)
            weight_sums   = torch.zeros(1, 1, lh, lw, device=self.device, dtype=self.dtype)

            for (y0, y1, x0, x1) in positions:
                tile     = latents[:, :, y0:y1, x0:x1]               # [1,4,th,tw]
                th, tw   = tile.shape[2], tile.shape[3]
                tile_in  = torch.cat([tile] * 2)                      # [2,4,th,tw]
                t_in     = torch.cat([t.unsqueeze(0)] * 2)

                # Build SDXL additional embeddings if needed
                unet_kwargs = {}
                if is_sdxl and pooled_cond is not None:
                    added_cond_kwargs = {
                        "text_embeds":  torch.cat([pooled_uncond, pooled_cond]),
                        "time_ids":     self._get_sdxl_time_ids(th, tw).expand(2, -1),
                    }
                    unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                pred = unet(
                    tile_in,
                    t_in,
                    encoder_hidden_states=text_embs.expand(2, -1, -1),
                    **unet_kwargs,
                ).sample                                               # [2,4,th,tw]

                pred_u, pred_c = pred.chunk(2)
                guided = pred_u + guidance_scale * (pred_c - pred_u)  # [1,4,th,tw]

                # Weighted accumulation
                w = cosine_tile_weight(th, tw, self.device).to(self.dtype)
                noise_preds[:, :, y0:y1, x0:x1]  += guided * w
                weight_sums[:, :,  y0:y1, x0:x1] += w

            # Normalise by overlap weights
            noise_avg = noise_preds / (weight_sums + 1e-8)

            # Scheduler step
            latents = scheduler.step(noise_avg, t, latents).prev_sample

        # ── Decode ──
        logger.info("Decoding final latent …")
        vae = self.pipe.get_vae() if not is_sdxl else self.pipe.base.vae
        image = decode_latents(vae, latents)                          # [1,3,H,W] in [0,1]

        # Convert to PIL
        image = image[0].cpu().permute(1, 2, 0).float().numpy()
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

    def _get_sdxl_time_ids(self, h: int, w: int) -> Tensor:
        """Build SDXL's added time embeddings for a given tile size."""
        ph, pw = h * self.LATENT_FACTOR, w * self.LATENT_FACTOR
        ids = torch.tensor(
            [[ph, pw, 0, 0, ph, pw]],
            device=self.device, dtype=self.dtype,
        )
        return ids
