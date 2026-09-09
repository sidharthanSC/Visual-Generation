"""
postprocess/upscale.py
─────────────────────────────────────────────────────────────────────────────
Upscaling utilities — no external pretrained upscaler required.

Methods:
  1. LatentUpscaler  — SDXL-native latent-space img2img upscaling.
     Encodes → adds noise → denoises at higher resolution.
  2. tile_upscale    — Tile-based 4× upscale using SD img2img.
     Splits a large image into overlapping tiles, upscales each,
     and blends with cosine weights (no additional model needed).

Both methods use only the permitted SD/SDXL models.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from pipeline import get_device, seed_generator

logger = logging.getLogger(__name__)


# ─── SDXL latent upscaler ─────────────────────────────────────────────────────

class LatentUpscaler:
    """
    2× upscale using SDXL img2img at a low denoising strength.

    The image is bicubic-upscaled, encoded to latent space,
    lightly noised, and then refined by the SDXL UNet.
    Strength < 0.5 ensures the structure is preserved while
    high-frequency details are added.

    Args:
        pipe:           SDXLPipeline (from pipeline.py).
        strength:       img2img denoising strength ∈ [0,1]. 0.3–0.4 is ideal.
        scale_factor:   Upscale factor (2 recommended for quality).
    """

    def __init__(self, pipe, strength: float = 0.35, scale_factor: int = 2):
        self.pipe         = pipe
        self.strength     = strength
        self.scale_factor = scale_factor
        is_sdxl = hasattr(pipe, "base")
        assert is_sdxl, "LatentUpscaler requires an SDXLPipeline."

    def __call__(
        self,
        image:      Image.Image,
        prompt:     str,
        negative_prompt: str           = "blurry, low quality, artifacts",
        seed:       Optional[int]      = None,
        steps:      int                = 40,
    ) -> Image.Image:
        """
        Upscale an image.

        Args:
            image:   Input PIL image.
            prompt:  Original prompt (helps guide detail synthesis).
        Returns:
            Upscaled PIL image at scale_factor × the input resolution.
        """
        w, h     = image.size
        new_w    = w * self.scale_factor
        new_h    = h * self.scale_factor
        # Bicubic preupscale
        upsampled = image.resize((new_w, new_h), Image.BICUBIC)

        # Previously gated on torch.cuda.is_available(), which silently dropped
        # the seed on MPS and made every Mac upscale non-reproducible.
        generator = seed_generator(seed, get_device())

        result = self.pipe.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upsampled,
            strength=self.strength,
            num_inference_steps=steps,
            generator=generator,
        )
        return result.images[0]


# ─── Tile-based upscaling ─────────────────────────────────────────────────────

def tile_upscale(
    pipe,
    image:      Image.Image,
    prompt:     str,
    negative_prompt: str           = "blurry, low quality",
    scale_factor:    int           = 2,
    tile_size:       int           = 512,
    overlap:         int           = 64,
    strength:        float         = 0.4,
    steps:           int           = 30,
    guidance_scale:  float         = 7.5,
    seed:            Optional[int] = None,
) -> Image.Image:
    """
    Tile-based 4× upscale using SD/SDXL img2img.

    Procedure:
      1. Bicubic upscale the full image by scale_factor.
      2. Split into overlapping tiles.
      3. Run img2img on each tile at low strength.
      4. Blend tiles using cosine weights.

    No external model beyond the two allowed is required.
    """
    is_sdxl = hasattr(pipe, "base")

    # ── Pre-upscale ──
    w, h   = image.size
    big    = image.resize((w * scale_factor, h * scale_factor), Image.BICUBIC)
    big_np = np.array(big).astype(np.float32)
    out_np = np.zeros_like(big_np)
    wgt_np = np.zeros((*big_np.shape[:2], 1), dtype=np.float32)

    bw, bh = big.size
    step   = tile_size - overlap
    xs     = list(range(0, bw - tile_size + 1, step)) + [max(0, bw - tile_size)]
    ys     = list(range(0, bh - tile_size + 1, step)) + [max(0, bh - tile_size)]
    xs     = sorted(set(xs))
    ys     = sorted(set(ys))

    # Cosine blend weight for a tile
    def _weight_2d(th, tw):
        wy = np.hanning(th)[:, None]
        wx = np.hanning(tw)[None, :]
        return (wy * wx)[:, :, None]           # [th, tw, 1]

    gen     = seed_generator(seed, get_device())

    logger.info(f"Tile upscale: {bh}×{bw} | {len(ys)*len(xs)} tiles")

    for y in ys:
        for x in xs:
            y1 = min(y + tile_size, bh)
            x1 = min(x + tile_size, bw)
            y0 = y1 - tile_size
            x0 = x1 - tile_size

            tile_pil = big.crop((x0, y0, x1, y1))

            # Run img2img on the tile
            if is_sdxl:
                result = pipe.base(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=tile_pil,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                ).images[0]
            else:
                # SD2 img2img
                from diffusers import StableDiffusionImg2ImgPipeline
                result = pipe.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=tile_pil,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                ).images[0]

            tile_np = np.array(result).astype(np.float32)
            w_tile  = _weight_2d(tile_np.shape[0], tile_np.shape[1])

            out_np[y0:y1, x0:x1] += tile_np * w_tile
            wgt_np[y0:y1, x0:x1] += w_tile

    result_np = (out_np / (wgt_np + 1e-8)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result_np)
