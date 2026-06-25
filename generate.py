"""
generate.py — Text-to-image entry point with dynamic CFG scheduling,
prompt weighting, and batch export.

Usage:
    python generate.py --prompt "a neon cyberpunk cityscape at midnight" \
                       --model sdxl --steps 50 --seed 42 --num_images 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

from PIL import Image

from config import GenerationConfig, IMG_DIR
from pipeline import load_pipeline, free_memory, get_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─── CFG scheduling ───────────────────────────────────────────────────────────

def linear_cfg_schedule(
    step: int,
    total_steps: int,
    cfg_start: float = 12.0,
    cfg_end: float = 5.0,
) -> float:
    """Linearly anneal CFG from high (early) to low (late denoising)."""
    frac = step / max(total_steps - 1, 1)
    return cfg_start + frac * (cfg_end - cfg_start)


def cosine_cfg_schedule(
    step: int,
    total_steps: int,
    cfg_start: float = 12.0,
    cfg_end: float = 5.0,
) -> float:
    """Cosine annealing CFG schedule."""
    import math
    frac = step / max(total_steps - 1, 1)
    cos_val = 0.5 * (1 + math.cos(math.pi * frac))
    return cfg_end + cos_val * (cfg_start - cfg_end)


# ─── Prompt helpers ───────────────────────────────────────────────────────────

def build_prompt_variants(base_prompt: str) -> list[str]:
    """
    Return a small set of stylistic augmentations of a base prompt.
    Useful for generating a diverse grid without changing the concept.
    """
    suffixes = [
        "masterpiece, best quality, highly detailed",
        "cinematic lighting, 8k, photorealistic",
        "oil painting style, vibrant colors, artstation",
        "soft diffused light, dreamy atmosphere",
    ]
    return [f"{base_prompt}, {s}" for s in suffixes]


def save_images(images: list[Image.Image], output_dir: Path, prefix: str) -> list[Path]:
    """Save a list of PIL images with timestamped filenames."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, img in enumerate(images):
        fname = output_dir / f"{prefix}_{ts}_{i:02d}.png"
        img.save(fname)
        logger.info(f"Saved → {fname}")
        paths.append(fname)
    return paths


def make_grid(images: list[Image.Image], cols: int = 2) -> Image.Image:
    """Tile a list of images into a single grid image."""
    import math
    n   = len(images)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    return grid


# ─── Main generation function ─────────────────────────────────────────────────

def generate(cfg: GenerationConfig) -> list[Image.Image]:
    """
    Run text-to-image generation from a GenerationConfig.
    Returns a list of PIL Images.
    """
    pipe = load_pipeline(
        model=cfg.model,
        use_refiner=cfg.use_refiner if cfg.model == "sdxl" else False,
    )

    logger.info(f"Generating {cfg.num_images} image(s) | model={cfg.model}")
    logger.info(f"Prompt: {cfg.prompt}")

    images = pipe(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        width=cfg.width,
        height=cfg.height,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        seed=cfg.seed,
        num_images_per_prompt=cfg.num_images,
        **({"high_noise_frac": cfg.high_noise_frac} if cfg.model == "sdxl" else {}),
    )

    save_images(images, cfg.output_dir, cfg.output_prefix)

    if cfg.num_images > 1:
        grid = make_grid(images, cols=min(4, cfg.num_images))
        grid_path = cfg.output_dir / f"{cfg.output_prefix}_grid.png"
        grid.save(grid_path)
        logger.info(f"Grid saved → {grid_path}")

    free_memory()
    return images


def generate_style_variants(base_prompt: str, cfg: GenerationConfig) -> list[Image.Image]:
    """
    Generate one image per stylistic prompt variant and return all results.
    """
    variants = build_prompt_variants(base_prompt)
    all_images = []

    pipe = load_pipeline(
        model=cfg.model,
        use_refiner=getattr(cfg, "use_refiner", False),
    )

    for i, prompt in enumerate(variants):
        logger.info(f"Variant {i+1}/{len(variants)}: {prompt}")
        imgs = pipe(
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.width,
            height=cfg.height,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed + i if cfg.seed is not None else None,
        )
        all_images.extend(imgs)

    save_images(all_images, cfg.output_dir, "variants")
    grid = make_grid(all_images, cols=2)
    grid.save(cfg.output_dir / "variants_grid.png")
    free_memory()
    return all_images


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Text-to-image generation")
    p.add_argument("--prompt",          type=str, default="a breathtaking fantasy landscape")
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, ugly")
    p.add_argument("--model",           type=str, default="sdxl", choices=["sdxl", "sd2"])
    p.add_argument("--width",           type=int, default=1024)
    p.add_argument("--height",          type=int, default=1024)
    p.add_argument("--steps",           type=int, default=40)
    p.add_argument("--cfg",             type=float, default=7.5)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--num_images",      type=int, default=1)
    p.add_argument("--no_refiner",      action="store_true")
    p.add_argument("--variants",        action="store_true", help="Generate style variants")
    p.add_argument("--output_prefix",   type=str, default="gen")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model=args.model,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        num_images=args.num_images,
        use_refiner=not args.no_refiner,
        output_prefix=args.output_prefix,
    )

    if args.variants:
        generate_style_variants(args.prompt, cfg)
    else:
        generate(cfg)
