"""
techniques/panorama.py
─────────────────────────────────────────────────────────────────────────────
360° equirectangular panorama generation using MultiDiffusion with
circular (horizontal) boundary handling.

Key ideas:
  • Canvas width ≈ 2× height for correct equirectangular ratio.
  • Tiles at the right boundary are circularly padded so the panorama
    wraps seamlessly from right back to left.
  • Vertical tiles use the same cosine blend as MultiDiffusion.
  • Optional equirectangular-to-cubemap conversion for previewing.

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

import numpy as np
import torch
from PIL import Image

from techniques.multidiffusion import MultiDiffusion, cosine_tile_weight
from config import PanoramaConfig, PANO_DIR

logger = logging.getLogger(__name__)


class PanoramaGenerator(MultiDiffusion):
    """
    Equirectangular panorama generator.

    Inherits MultiDiffusion tiled denoising and adds circular
    horizontal wrapping so the left and right edges connect seamlessly.
    """

    def generate(self, cfg: PanoramaConfig) -> Image.Image:
        """
        Generate a full 360° panorama.

        Args:
            cfg: PanoramaConfig dataclass with all parameters.
        Returns:
            PIL Image in equirectangular projection.
        """
        image = self(
            prompt              = cfg.prompt,
            negative_prompt     = cfg.negative_prompt,
            canvas_h            = cfg.canvas_height,
            canvas_w            = cfg.canvas_width,
            num_inference_steps = cfg.num_inference_steps,
            guidance_scale      = cfg.guidance_scale,
            seed                = cfg.seed,
            circular            = True,
        )
        out_path = cfg.output_dir / "panorama.png"
        image.save(out_path)
        logger.info(f"Panorama saved → {out_path}")
        return image

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
        circular:            bool  = True,
    ) -> Image.Image:
        """
        Generate panorama with optional circular horizontal wrapping.
        """
        lh = canvas_h // self.LATENT_FACTOR
        lw = canvas_w // self.LATENT_FACTOR

        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None else None
        )

        is_sdxl = hasattr(self.pipe, "base")
        cond_emb, uncond_emb, pooled_cond, pooled_uncond = self._encode_prompt(
            prompt, negative_prompt
        )
        text_embs = torch.cat([uncond_emb, cond_emb], dim=0)

        latents = torch.randn(
            1, 4, lh, lw,
            device=self.device, dtype=self.dtype, generator=generator
        )
        scheduler = (
            self.pipe.get_scheduler()
            if not is_sdxl else self.pipe.base.scheduler
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        latents   = latents * scheduler.init_noise_sigma

        positions = self._tile_positions(lh, lw)
        unet      = self.pipe.get_unet() if not is_sdxl else self.pipe.base.unet

        logger.info(
            f"PanoramaGenerator: {canvas_h}×{canvas_w} | "
            f"{len(positions)} tiles | circular={circular}"
        )

        for t in timesteps:
            noise_preds = torch.zeros_like(latents)
            weight_sums = torch.zeros(1, 1, lh, lw, device=self.device, dtype=self.dtype)

            for (y0, y1, x0, x1) in positions:
                tile, did_wrap = self._extract_tile(latents, y0, y1, x0, x1, circular)
                th, tw = tile.shape[2], tile.shape[3]
                tile_in = torch.cat([tile] * 2)
                t_in    = torch.cat([t.unsqueeze(0)] * 2)

                unet_kwargs = {}
                if is_sdxl and pooled_cond is not None:
                    added_cond_kwargs = {
                        "text_embeds": torch.cat([pooled_uncond, pooled_cond]),
                        "time_ids":    self._get_sdxl_time_ids(th, tw).expand(2, -1),
                    }
                    unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                pred   = unet(tile_in, t_in,
                              encoder_hidden_states=text_embs.expand(2, -1, -1),
                              **unet_kwargs).sample
                pu, pc = pred.chunk(2)
                guided = pu + guidance_scale * (pc - pu)

                w = cosine_tile_weight(th, tw, self.device).to(self.dtype)
                self._accumulate_tile(
                    noise_preds, weight_sums, guided, w,
                    y0, y1, x0, x1, lw, did_wrap, circular
                )

            noise_avg = noise_preds / (weight_sums + 1e-8)
            latents   = scheduler.step(noise_avg, t, latents).prev_sample

        vae    = self.pipe.get_vae() if not is_sdxl else self.pipe.base.vae
        latents = latents / 0.18215
        image  = vae.decode(latents).sample
        image  = (image / 2 + 0.5).clamp(0, 1)
        image  = image[0].cpu().permute(1, 2, 0).float().numpy()
        image  = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

    # ── Circular tile helpers ─────────────────────────────────────────────────

    def _extract_tile(
        self,
        latents: torch.Tensor,
        y0: int, y1: int, x0: int, x1: int,
        circular: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        Extract tile, handling circular wrap if the tile overflows the right edge.
        Returns (tile_tensor, did_wrap).
        """
        lw = latents.shape[3]
        if not circular or x1 <= lw:
            return latents[:, :, y0:y1, x0:x1], False

        # Wrap: stitch right portion + left portion
        right = latents[:, :, y0:y1, x0:]
        left  = latents[:, :, y0:y1, : x1 - lw]
        return torch.cat([right, left], dim=3), True

    def _accumulate_tile(
        self,
        noise_preds:  torch.Tensor,
        weight_sums:  torch.Tensor,
        guided:       torch.Tensor,
        w:            torch.Tensor,
        y0: int, y1: int, x0: int, x1: int,
        lw: int, did_wrap: bool, circular: bool,
    ):
        """Accumulate tile prediction, handling wrap-around correctly."""
        if not did_wrap:
            noise_preds[:, :, y0:y1, x0:x1] += guided * w
            weight_sums[:, :, y0:y1, x0:x1] += w
        else:
            rw = lw - x0
            noise_preds[:, :, y0:y1, x0:]       += guided[:, :, :, :rw]  * w[:, :, :, :rw]
            noise_preds[:, :, y0:y1, :x1 - lw]  += guided[:, :, :, rw:]  * w[:, :, :, rw:]
            weight_sums[:, :, y0:y1, x0:]        += w[:, :, :, :rw]
            weight_sums[:, :, y0:y1, :x1 - lw]  += w[:, :, :, rw:]


# ─── Equirectangular → Cubemap conversion ─────────────────────────────────────

def equirect_to_cubemap(
    equirect: Image.Image,
    face_size: int = 512,
) -> dict[str, Image.Image]:
    """
    Convert an equirectangular panorama to 6 cubemap faces.

    Args:
        equirect:  Equirectangular PIL image (W ≈ 2H).
        face_size: Output size for each square face.
    Returns:
        Dict with keys: 'front', 'back', 'left', 'right', 'top', 'bottom'.
    """
    eqr = np.array(equirect).astype(np.float32) / 255.0
    H, W, C = eqr.shape
    faces = {}

    face_dirs = {
        "front":  ( 0,  0,  1),
        "back":   ( 0,  0, -1),
        "left":   (-1,  0,  0),
        "right":  ( 1,  0,  0),
        "top":    ( 0,  1,  0),
        "bottom": ( 0, -1,  0),
    }

    yx  = np.linspace(-1, 1, face_size)
    yy, xx = np.meshgrid(yx, yx, indexing="ij")   # [fs, fs]

    def _sample(phi, theta):
        """Sample equirectangular at (phi, theta) angles."""
        u = (theta / (2 * math.pi) + 0.5) % 1.0
        v = phi / math.pi + 0.5
        xi = (u * (W - 1)).astype(int).clip(0, W - 1)
        yi = (v * (H - 1)).astype(int).clip(0, H - 1)
        return eqr[yi, xi]

    for face_name, (fx, fy, fz) in face_dirs.items():
        if   abs(fz) == 1: rx, ry, rz = (xx * fz,  yy,  fz * np.ones_like(xx))
        elif abs(fx) == 1: rx, ry, rz = (fx * np.ones_like(xx), yy, -xx * fx)
        else:              rx, ry, rz = (xx, fy * np.ones_like(xx), -yy * fy)

        norm  = np.sqrt(rx**2 + ry**2 + rz**2)
        rx, ry, rz = rx / norm, ry / norm, rz / norm

        theta = np.arctan2(rx, rz)
        phi   = np.arcsin(np.clip(ry, -1, 1))
        face_rgb = _sample(phi, theta)
        faces[face_name] = Image.fromarray((face_rgb * 255).clip(0, 255).astype(np.uint8))

    return faces


def save_cubemap(faces: dict[str, Image.Image], output_dir) -> None:
    """Save each cubemap face as a PNG."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, img in faces.items():
        img.save(output_dir / f"cubemap_{name}.png")
        logger.info(f"Saved cubemap face → {output_dir / f'cubemap_{name}.png'}")


def make_cubemap_cross(faces: dict[str, Image.Image], face_size: int = 512) -> Image.Image:
    """Arrange the 6 cubemap faces into a cross layout PNG."""
    order = [
        (1, 0, "top"),
        (0, 1, "left"), (1, 1, "front"), (2, 1, "right"), (3, 1, "back"),
        (1, 2, "bottom"),
    ]
    cross = Image.new("RGB", (4 * face_size, 3 * face_size))
    for col, row, name in order:
        face = faces[name].resize((face_size, face_size))
        cross.paste(face, (col * face_size, row * face_size))
    return cross
