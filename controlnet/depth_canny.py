"""
controlnet/depth_canny.py
─────────────────────────────────────────────────────────────────────────────
ControlNet guidance wrappers for SD2 and SDXL.

Supported conditioning types:
  • Canny edge maps   (classical Canny via OpenCV)
  • Depth maps        (via MiDaS or DPT depth estimator)

ControlNet IDs used (SD2-compatible):
  • Canny:  "thibaud/controlnet-sd21-canny-diffusers"
  • Depth:  "thibaud/controlnet-sd21-depth-diffusers"
  • SDXL Canny: "diffusers/controlnet-canny-sdxl-1.0"
  • SDXL Depth: "diffusers/controlnet-depth-sdxl-1.0"

Reference
---------
Zhang et al. 2023 — "Adding Conditional Control to Text-to-Image Diffusion Models"
    https://arxiv.org/abs/2302.05543
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
)

from config import ControlNetConfig, IMG_DIR

logger = logging.getLogger(__name__)


# ─── Image preprocessors ──────────────────────────────────────────────────────

def preprocess_canny(
    image:       Union[Image.Image, np.ndarray],
    low_thresh:  int = 100,
    high_thresh: int = 200,
) -> Image.Image:
    """
    Extract Canny edge map from an input image.

    Args:
        image:       PIL or NumPy input image.
        low_thresh:  Lower hysteresis threshold.
        high_thresh: Upper hysteresis threshold.
    Returns:
        Single-channel PIL image (edges = 255, background = 0).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    grey  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(grey, low_thresh, high_thresh)
    # Stack to 3-channel for ControlNet
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def preprocess_depth(
    image:  Union[Image.Image, np.ndarray],
    device: torch.device = torch.device("cpu"),
) -> Image.Image:
    """
    Estimate depth map using DPT (Intel MiDaS v3) from transformers.

    This uses only the freely available pretrained depth estimator
    from HuggingFace transformers — no additional generative models.

    Args:
        image:  Input RGB PIL image.
        device: Compute device.
    Returns:
        Normalised depth map as a 3-channel PIL image ∈ [0, 255].
    """
    try:
        from transformers import pipeline as hf_pipeline
        estimator = hf_pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=0 if device.type == "cuda" else -1,
        )
        result = estimator(image)
        depth  = result["depth"]                  # PIL Image, greyscale
        depth  = np.array(depth).astype(np.float32)
        depth  = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth  = (depth * 255).clip(0, 255).astype(np.uint8)
        depth_rgb = np.stack([depth] * 3, axis=-1)
        return Image.fromarray(depth_rgb)
    except Exception as e:
        logger.warning(f"DPT depth failed: {e}. Returning blank depth map.")
        arr = np.zeros((*np.array(image).shape[:2], 3), dtype=np.uint8)
        return Image.fromarray(arr)


# ─── ControlNet wrapper ───────────────────────────────────────────────────────

class ControlNetWrapper:
    """
    Unified ControlNet pipeline for SD2 and SDXL.

    Args:
        cfg: ControlNetConfig dataclass.
    """

    SD2_CONTROLNET_IDS = {
        "canny": "thibaud/controlnet-sd21-canny-diffusers",
        "depth": "thibaud/controlnet-sd21-depth-diffusers",
    }
    SDXL_CONTROLNET_IDS = {
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
    }

    def __init__(self, cfg: ControlNetConfig):
        self.cfg    = cfg
        from pipeline import get_device, get_dtype
        self.device = get_device()
        self.dtype  = get_dtype(self.device)
        self.pipe   = None
        self._load()

    def _load(self):
        cfg  = self.cfg
        mode = cfg.model.lower()

        if mode == "sd2":
            cn_id = self.SD2_CONTROLNET_IDS[cfg.controlnet_type]
            from config import SD2_MODEL_ID
            base_id = SD2_MODEL_ID
        else:
            cn_id = self.SDXL_CONTROLNET_IDS[cfg.controlnet_type]
            from config import SDXL_MODEL_ID
            base_id = SDXL_MODEL_ID

        logger.info(f"Loading ControlNet ({cfg.controlnet_type}) for {mode} …")
        controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=self.dtype)

        PipelineClass = (
            StableDiffusionControlNetPipeline
            if mode == "sd2"
            else StableDiffusionXLControlNetPipeline
        )
        self.pipe = PipelineClass.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.device.type == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def __call__(
        self,
        prompt:              str,
        control_image:       Image.Image,
        negative_prompt:     str            = "",
        num_inference_steps: int            = 40,
        guidance_scale:      float          = 7.5,
        controlnet_scale:    float          = 0.8,
        seed:                Optional[int]  = None,
        width:               int            = 768,
        height:              int            = 768,
    ) -> list[Image.Image]:
        """
        Run ControlNet-guided generation.

        Args:
            control_image: Preprocessed edge/depth map (Canny or depth).
            controlnet_scale: How strongly to follow the conditioning [0,1].
        """
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None else None
        )
        result = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
            width=width,
            height=height,
        )
        return result.images

    def generate_from_sketch(
        self,
        sketch:      Image.Image,
        prompt:      str,
        output_dir:  Path = IMG_DIR,
        **kwargs,
    ) -> Image.Image:
        """
        Convenience: auto-preprocess a sketch → Canny → generate.
        """
        assert self.cfg.controlnet_type == "canny", "Use canny type for sketches."
        canny = preprocess_canny(sketch)
        imgs  = self(prompt=prompt, control_image=canny, **kwargs)
        img   = imgs[0]
        img.save(output_dir / "controlnet_from_sketch.png")
        return img

    def generate_with_depth(
        self,
        rgb_image:   Image.Image,
        prompt:      str,
        output_dir:  Path = IMG_DIR,
        **kwargs,
    ) -> Image.Image:
        """
        Convenience: estimate depth from an RGB image → generate.
        """
        assert self.cfg.controlnet_type == "depth", "Use depth type."
        depth = preprocess_depth(rgb_image, self.device)
        imgs  = self(prompt=prompt, control_image=depth, **kwargs)
        img   = imgs[0]
        img.save(output_dir / "controlnet_depth.png")
        return img
