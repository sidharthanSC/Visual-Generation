"""
techniques/style_transfer.py
─────────────────────────────────────────────────────────────────────────────
LoRA-based style transfer at inference time.

Loads a trained LoRA checkpoint and blends it into the pipeline at a given
weight, enabling soft/strong style application with a single call.

Reference
---------
Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class LoRAStyleTransfer:
    """
    Apply a LoRA-fine-tuned style to any prompt at inference time.

    Args:
        pipe:           SDXLPipeline or SD2Pipeline.
        lora_path:      Path to .safetensors or .pt LoRA weights.
        lora_scale:     Blend weight ∈ [0, 1]. 1 = full style, 0 = none.
    """

    def __init__(
        self,
        pipe,
        lora_path:  Union[str, Path],
        lora_scale: float = 0.9,
    ):
        self.pipe       = pipe
        self.lora_path  = Path(lora_path)
        self.lora_scale = lora_scale
        self._loaded    = False

    def load(self):
        """Load LoRA weights into the pipeline."""
        if self._loaded:
            return
        is_sdxl = hasattr(self.pipe, "base")
        target  = self.pipe.base if is_sdxl else self.pipe.pipe

        logger.info(f"Loading LoRA from {self.lora_path} (scale={self.lora_scale})")
        target.load_lora_weights(str(self.lora_path))
        target.fuse_lora(lora_scale=self.lora_scale)
        self._loaded = True

    def unload(self):
        """Remove LoRA weights (restore original pipeline)."""
        if not self._loaded:
            return
        is_sdxl = hasattr(self.pipe, "base")
        target  = self.pipe.base if is_sdxl else self.pipe.pipe
        target.unfuse_lora()
        self._loaded = False

    def __call__(
        self,
        prompt:              str,
        negative_prompt:     str            = "",
        width:               int            = 1024,
        height:              int            = 1024,
        num_inference_steps: int            = 40,
        guidance_scale:      float          = 7.5,
        seed:                Optional[int]  = None,
        num_images:          int            = 1,
    ) -> list[Image.Image]:
        """
        Generate images with the LoRA style applied.

        The LoRA is loaded on first call and stays loaded until `.unload()`.
        """
        self.load()

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images_per_prompt=num_images,
        )
        return images

    def interpolate(
        self,
        prompt:          str,
        negative_prompt: str   = "",
        scales:          list  = [0.0, 0.3, 0.6, 1.0],
        **gen_kwargs,
    ) -> list[tuple[float, Image.Image]]:
        """
        Generate images at multiple LoRA scales for visual comparison.

        Returns:
            List of (scale, PIL Image) tuples.
        """
        is_sdxl = hasattr(self.pipe, "base")
        target  = self.pipe.base if is_sdxl else self.pipe.pipe
        results = []

        target.load_lora_weights(str(self.lora_path))

        for scale in scales:
            logger.info(f"Generating with LoRA scale={scale}")
            target.fuse_lora(lora_scale=scale)
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **gen_kwargs,
            )
            target.unfuse_lora()
            results.append((scale, images[0]))

        return results
