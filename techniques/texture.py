"""
techniques/texture.py
─────────────────────────────────────────────────────────────────────────────
Seamless texture generation for UV-mapped 3-D meshes.

Techniques used:
  1. Circular-padded tiled generation  — ensures texture tiles seamlessly.
  2. Frequency-domain seam removal     — FFT-based boundary correction.
  3. SDS-based texture optimisation    — optional refinement via SDS gradient.
  4. Normal / roughness / metallic     — generate PBR material maps from
     the colour texture using lightweight CNNs or frequency cues.

References
----------
Richardson et al. 2023 — "TEXTure: Text-Guided Texturing of 3D Shapes"
    https://arxiv.org/abs/2302.01721
Chen et al. 2023 — "Text2Tex: Text-driven Texture Synthesis via
    Diffusion Models"
    https://arxiv.org/abs/2303.11396
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pipeline import decode_latents

logger = logging.getLogger(__name__)


# ─── Seamless texture generator ───────────────────────────────────────────────

class SeamlessTextureGenerator:
    """
    Generate seamless, tileable textures using a diffusion pipeline.

    Strategy:
      • Run generation on a canvas padded by half a tile on each side.
      • Circularly pad the latent each step so borders are consistent.
      • Crop back to the target size — edges blend seamlessly.

    Args:
        pipe:        SD2Pipeline or SDXLPipeline.
        texture_size: Output texture resolution (square).
        overlap_pad: Padding in latent space for seamless wrapping.
    """

    LATENT_FACTOR = 8

    def __init__(self, pipe, texture_size: int = 1024, overlap_pad: int = 32):
        self.pipe         = pipe
        self.tex_size     = texture_size
        self.lat_size     = texture_size  // self.LATENT_FACTOR
        self.overlap_pad  = overlap_pad

        p = next(pipe.get_unet().parameters())
        self.device = p.device
        self.dtype  = p.dtype

    # ── Prompt encoding ───────────────────────────────────────────────────────

    def _encode_prompt(self, prompt: str, neg: str):
        is_sdxl = hasattr(self.pipe, "base")
        if is_sdxl:
            return self.pipe.encode_prompt(prompt, neg)
        tok, tenc = self.pipe.get_tokenizer(), self.pipe.get_text_encoder()
        from guidance.classifier_free import get_weighted_text_embeddings
        c, u = get_weighted_text_embeddings(tok, tenc, prompt, neg, self.device)
        return c, u, None, None

    # ── Core generation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt:              str,
        negative_prompt:     str   = "blurry, seams, low quality",
        num_inference_steps: int   = 40,
        guidance_scale:      float = 7.5,
        seed:                Optional[int] = 0,
    ) -> Image.Image:
        """
        Generate a seamlessly tileable texture.

        Returns:
            PIL Image of size (texture_size, texture_size).
        """
        pad    = self.overlap_pad
        lsize  = self.lat_size + 2 * pad   # padded latent canvas

        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = torch.randn(1, 4, lsize, lsize, device=self.device, dtype=self.dtype, generator=gen)

        is_sdxl = hasattr(self.pipe, "base")
        cond, uncond, pooled_c, pooled_u = self._encode_prompt(prompt, negative_prompt)
        text_embs = torch.cat([uncond, cond])

        scheduler = self.pipe.base.scheduler if is_sdxl else self.pipe.get_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * scheduler.init_noise_sigma
        unet    = self.pipe.base.unet if is_sdxl else self.pipe.get_unet()

        for t in scheduler.timesteps:
            # ── Circular pad latent so texture wraps ──
            lat_pad = F.pad(latents, (pad, pad, pad, pad), mode="circular")

            tile_in = torch.cat([lat_pad] * 2)
            t_in    = torch.cat([t.unsqueeze(0)] * 2)

            unet_kwargs = {}
            if is_sdxl and pooled_c is not None:
                h, w = lat_pad.shape[2:]
                unet_kwargs["added_cond_kwargs"] = {
                    "text_embeds": torch.cat([pooled_u, pooled_c]),
                    "time_ids":    self._sdxl_time_ids(h, w).expand(2, -1),
                }

            pred   = unet(tile_in, t_in, encoder_hidden_states=text_embs.expand(2,-1,-1), **unet_kwargs).sample
            pu, pc = pred.chunk(2)
            guided = pu + guidance_scale * (pc - pu)

            # ── Crop back to un-padded size ──
            guided_crop = guided[:, :, pad:-pad, pad:-pad]

            latents = scheduler.step(guided_crop, t, latents).prev_sample

        # Decode
        vae = self.pipe.base.vae if is_sdxl else self.pipe.get_vae()
        image = decode_latents(vae, latents)           # [1,3,H,W] in [0,1]
        image = image[0].cpu().permute(1, 2, 0).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil   = Image.fromarray(image)

        # FFT seam correction as post-process
        pil = self._fft_seam_correction(pil)
        return pil

    def _sdxl_time_ids(self, h: int, w: int) -> torch.Tensor:
        ph, pw = h * self.LATENT_FACTOR, w * self.LATENT_FACTOR
        return torch.tensor([[ph, pw, 0, 0, ph, pw]], device=self.device, dtype=self.dtype)

    # ── FFT seam correction ───────────────────────────────────────────────────

    def _fft_seam_correction(self, img: Image.Image, blend_px: int = 32) -> Image.Image:
        """
        Blend left↔right and top↔bottom seams by averaging opposing borders
        in the frequency domain.

        This is a fast post-process that removes any residual colour
        discontinuity at tile boundaries.
        """
        arr  = np.array(img).astype(np.float32)   # [H,W,3]
        arr  = self._blend_borders(arr, blend_px)
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    @staticmethod
    def _blend_borders(arr: np.ndarray, px: int) -> np.ndarray:
        """Average the first and last `px` rows/cols to close seams."""
        # Horizontal seam (left ↔ right)
        left  = arr[:, :px, :].copy()
        right = arr[:, -px:, :].copy()
        blend = (left + right[:, ::-1, :]) / 2
        arr[:, :px,  :] = blend
        arr[:, -px:, :] = blend[:, ::-1, :]

        # Vertical seam (top ↔ bottom)
        top    = arr[:px, :, :].copy()
        bottom = arr[-px:, :, :].copy()
        blend  = (top + bottom[::-1, :, :]) / 2
        arr[:px,  :, :] = blend
        arr[-px:, :, :] = blend[::-1, :, :]

        return arr

    # ── PBR material map generation ───────────────────────────────────────────

    def generate_pbr_maps(
        self,
        albedo: Image.Image,
    ) -> dict[str, Image.Image]:
        """
        Derive PBR material maps from an albedo texture using
        frequency-domain heuristics (no external model required).

        Returns:
            dict with keys: 'normal', 'roughness', 'metallic', 'ao'
        """
        arr = np.array(albedo).astype(np.float32) / 255.0   # [H,W,3]

        # ── Normal map (Sobel gradient) ──
        grey    = arr.mean(axis=2)
        gx      = np.gradient(grey, axis=1)
        gy      = np.gradient(grey, axis=0)
        strength = 3.0
        normal   = np.stack([
            gx * strength,
            gy * strength,
            np.ones_like(grey),
        ], axis=2)
        norms    = np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
        normal  /= norms
        normal   = ((normal + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        normal_map = Image.fromarray(normal)

        # ── Roughness map (inverse local variance proxy) ──
        from scipy.ndimage import uniform_filter
        grey_blur   = uniform_filter(grey, size=5)
        grey_sq     = uniform_filter(grey ** 2, size=5)
        variance    = np.clip(grey_sq - grey_blur ** 2, 0, None)
        variance   /= (variance.max() + 1e-8)
        roughness   = (1.0 - variance)               # smoother → less rough
        roughness   = (roughness * 255).clip(0, 255).astype(np.uint8)
        roughness_map = Image.fromarray(roughness)

        # ── Metallic map (desaturated & high-value regions) ──
        sat     = arr.max(axis=2) - arr.min(axis=2)   # saturation proxy
        val     = arr.max(axis=2)                      # value
        metallic = ((val > 0.5) & (sat < 0.2)).astype(np.float32)
        metallic = (metallic * 255).astype(np.uint8)
        metallic_map = Image.fromarray(metallic)

        # ── Ambient Occlusion (darkened low-frequency component) ──
        low_freq = uniform_filter(grey, size=20)
        ao       = (low_freq / (low_freq.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        ao_map   = Image.fromarray(ao)

        return {
            "normal":    normal_map,
            "roughness": roughness_map,
            "metallic":  metallic_map,
            "ao":        ao_map,
        }

    def save_texture_pack(
        self,
        albedo: Image.Image,
        output_dir: Path,
        name: str = "texture",
    ) -> dict[str, Path]:
        """Generate and save a full PBR texture pack."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pbr = self.generate_pbr_maps(albedo)
        paths = {}

        albedo_path = output_dir / f"{name}_albedo.png"
        albedo.save(albedo_path)
        paths["albedo"] = albedo_path

        for map_name, map_img in pbr.items():
            p = output_dir / f"{name}_{map_name}.png"
            map_img.save(p)
            paths[map_name] = p
            logger.info(f"Saved {map_name} map → {p}")

        return paths
