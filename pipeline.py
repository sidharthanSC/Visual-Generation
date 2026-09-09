"""
pipeline.py — Thin wrappers around Diffusers pipelines.
Handles model loading, device placement, dtype selection, and seed management.
Supports SDXL (base + optional refiner) and Stable Diffusion 2.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional, Union

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image

from config import SDXL_MODEL_ID, SDXL_REFINER_ID, SD2_MODEL_ID

logger = logging.getLogger(__name__)


# ─── Device & dtype helpers ────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    fp16 on CUDA, bf16 on MPS, fp32 on CPU.

    Half precision on MPS is ~2.4-4x faster than fp32 on the ops that dominate a
    UNet step (measured on M5: conv 2.4x, attention 2.5x, feed-forward 4.1x) and
    halves both the download and the resident weight size — SDXL's UNet is 5.1 GB
    instead of 10.3 GB, which is what lets it fit in the MPS budget on a 16 GB Mac.

    bf16 rather than fp16 specifically because the SDXL VAE overflows in fp16 and
    decodes to all-NaN. Diffusers normally works around this by upcasting the VAE
    to fp32 for the decode, but on MPS an fp32 VAE decode falls back to a
    group_norm decomposition that tries to allocate ~20 GB and dies. bf16 carries
    fp32's exponent range at fp16's footprint, so the decode is finite and fits:
    a 1024x1024 decode peaks at 2.5 GB, and a 4096x2048 panorama at 2.6 GB.
    """
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def is_half(dtype: torch.dtype) -> bool:
    """True for the 16-bit dtypes we load fp16 checkpoint variants for."""
    return dtype in (torch.float16, torch.bfloat16)


def prepare_vae(vae, device: torch.device) -> None:
    """
    Bound VAE memory for use outside the Diffusers pipeline.

    Decoding a whole 4096x2048 canvas in one `vae.decode()` allocates far more
    than the MPS budget. `enable_tiling()` is honoured inside `AutoencoderKL.decode`
    itself, so it applies to the direct calls in techniques/ too, not just to
    pipeline calls.

    Note this deliberately does NOT pre-upcast the VAE to fp32. The SDXL pipeline
    upcasts and restores around its own decode, and on MPS it will actively cast a
    fp32 VAE back down to match fp16 latents (pipeline_stable_diffusion_xl.py:1257).
    Leaving the VAE at the pipeline dtype keeps that path correct; direct callers
    get the upcast from `decode_latents` instead.
    """
    if device.type in ("mps", "cpu"):
        vae.enable_tiling()
        vae.enable_slicing()
        # enable_tiling() alone is not enough: Diffusers only tiles when the latent
        # is strictly larger than tile_latent_min_size, which defaults to
        # sample_size/8 == 128 for SDXL. A standard 1024x1024 decode lands exactly
        # on that boundary, skips tiling, and tries to allocate ~19 GB in fp32.
        # Halving the tile makes tiling engage from 1024x1024 upward.
        vae.tile_sample_min_size = min(getattr(vae, "tile_sample_min_size", 1024), 512)
        vae.tile_latent_min_size = vae.tile_sample_min_size // 8


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode latents to a [B,3,H,W] tensor in [0,1] — the safe equivalent of a bare
    `vae.decode(latents / 0.18215).sample` for code that bypasses the pipeline.

    Fixes two things a bare call gets wrong:
      • scaling_factor is read off the VAE config. SDXL's is 0.13025, not the
        0.18215 that SD1.x/SD2 use, so the hardcoded constant scales SDXL latents
        by ~1.4x and washes out every decode.
      • an fp16 VAE overflows to NaN, so it is upcast for the decode and restored
        afterwards. On MPS the upcast target is bf16, not fp32: fp32 group_norm
        there hits a decomposition that OOMs at ~20 GB.

    Inference-only — autograd on a full-canvas decode retains every activation and
    costs ~20 GB where the no-grad decode costs 2.5 GB. Callers that need gradients
    through the decoder (LatentOptimiser) must not use this.
    """
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    orig_dtype = vae.dtype
    needs_upcast = orig_dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    if needs_upcast:
        vae.to(dtype=torch.bfloat16 if vae.device.type == "mps" else torch.float32)

    latents = latents.to(dtype=vae.dtype) / scaling_factor
    try:
        image = vae.decode(latents).sample
    finally:
        if needs_upcast:
            vae.to(dtype=orig_dtype)

    return (image / 2 + 0.5).clamp(0, 1)


def seed_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


# ─── Scheduler factory ────────────────────────────────────────────────────────

SCHEDULER_MAP = {
    "ddim":    DDIMScheduler,
    "dpm":     DPMSolverMultistepScheduler,
    "euler":   EulerDiscreteScheduler,
}

def make_scheduler(name: str = "dpm", **kwargs):
    cls = SCHEDULER_MAP.get(name, DPMSolverMultistepScheduler)
    return cls(**kwargs)


# ─── SD2 pipeline ─────────────────────────────────────────────────────────────

class SD2Pipeline:
    """Wrapper around Stable Diffusion 2."""

    def __init__(
        self,
        scheduler: str = "dpm",
        device: Optional[torch.device] = None,
        enable_xformers: bool = True,
    ):
        self.device = device or get_device()
        self.dtype  = get_dtype(self.device)
        logger.info(f"Loading SD2 on {self.device} ({self.dtype})")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD2_MODEL_ID,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        prepare_vae(self.pipe.vae, self.device)

        if enable_xformers and self.device.type == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled")
            except Exception:
                logger.warning("xformers not available, skipping")

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images_per_prompt: int = 1,
    ) -> list[Image.Image]:
        generator = seed_generator(seed, self.device)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        )
        return result.images

    def get_unet(self):
        return self.pipe.unet

    def get_vae(self):
        return self.pipe.vae

    def get_tokenizer(self):
        return self.pipe.tokenizer

    def get_text_encoder(self):
        return self.pipe.text_encoder

    def get_scheduler(self):
        return self.pipe.scheduler


# ─── SDXL pipeline ────────────────────────────────────────────────────────────

class SDXLPipeline:
    """
    Wrapper around SDXL base + optional refiner.
    Supports two-stage generation (base → refiner) per the official recipe.
    """

    def __init__(
        self,
        use_refiner: bool = True,
        scheduler: str = "dpm",
        device: Optional[torch.device] = None,
        enable_xformers: bool = True,
    ):
        self.device      = device or get_device()
        self.dtype       = get_dtype(self.device)
        self.use_refiner = use_refiner
        logger.info(f"Loading SDXL base on {self.device} ({self.dtype})")

        self.base = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if is_half(self.dtype) else None,
        ).to(self.device)

        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config
        )

        prepare_vae(self.base.vae, self.device)

        if enable_xformers and self.device.type == "cuda":
            try:
                self.base.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.refiner = None
        if use_refiner:
            logger.info("Loading SDXL refiner …")
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                SDXL_REFINER_ID,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if is_half(self.dtype) else None,
                # Reuse the base VAE and text encoder 2 rather than loading a
                # second copy — on a 16 GB machine the duplicates are the
                # difference between fitting and swapping.
                vae=self.base.vae,
                text_encoder_2=self.base.text_encoder_2,
            ).to(self.device)

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        high_noise_frac: float = 0.8,
        seed: Optional[int] = None,
        num_images_per_prompt: int = 1,
    ) -> list[Image.Image]:
        generator = seed_generator(seed, self.device)
        denoising_end = high_noise_frac if self.use_refiner else None

        # ── Stage 1: base ──
        base_output = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            denoising_end=denoising_end,
            output_type="latent" if self.use_refiner else "pil",
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        )

        if not self.use_refiner:
            return base_output.images

        # ── Stage 2: refiner ──
        refiner_output = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_output.images,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return refiner_output.images

    def get_unet(self):
        return self.base.unet

    def get_vae(self):
        return self.base.vae

    def get_tokenizers(self):
        return self.base.tokenizer, self.base.tokenizer_2

    def get_text_encoders(self):
        return self.base.text_encoder, self.base.text_encoder_2

    def get_scheduler(self):
        return self.base.scheduler

    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> tuple:
        """Return (prompt_embeds, neg_embeds, pooled_embeds, neg_pooled) for SDXL."""
        return self.base.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )


# ─── Pipeline factory ─────────────────────────────────────────────────────────

def load_pipeline(
    model: str = "sdxl",
    use_refiner: bool = True,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Union[SDXLPipeline, SD2Pipeline]:
    """
    Factory function — returns the right pipeline based on `model`.

    Args:
        model:       "sdxl" or "sd2"
        use_refiner: (SDXL only) load and use the refiner stage
        device:      override device
    """
    model = model.lower().strip()
    if model == "sdxl":
        return SDXLPipeline(use_refiner=use_refiner, device=device, **kwargs)
    elif model in ("sd2", "stable-diffusion-2"):
        return SD2Pipeline(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'sdxl' or 'sd2'.")


def free_memory():
    """Release accelerator memory after a pipeline is done."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
