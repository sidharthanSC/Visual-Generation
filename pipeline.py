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
    """fp16 on CUDA, fp32 everywhere else."""
    return torch.float16 if device.type == "cuda" else torch.float32


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
            variant="fp16" if self.dtype == torch.float16 else None,
        ).to(self.device)

        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config
        )

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
                variant="fp16" if self.dtype == torch.float16 else None,
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
    """Release GPU memory after a pipeline is done."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
