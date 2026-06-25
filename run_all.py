"""
run_all.py — Master demo runner for the image generative models contest.

Runs all implemented techniques sequentially, saving outputs to `outputs/`.
Each section can be toggled on/off via the RUN_* flags below.

Usage:
    python run_all.py                     # run everything
    python run_all.py --only panorama     # run only the panorama
    python run_all.py --only sds          # run only SDS
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")


# ─── Feature flags ────────────────────────────────────────────────────────────
# Set any to False to skip that experiment.

RUN_TEXT2IMG   = True
RUN_VARIANTS   = True
RUN_PANORAMA   = True
RUN_TEXTURE    = True
RUN_SDS        = True
RUN_CONTROLNET = False   # Requires ControlNet model download
RUN_REWARD     = True
RUN_UPSCALE    = True


def section(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


# ─── 1. Basic text-to-image ───────────────────────────────────────────────────

def run_text2img():
    section("TEXT-TO-IMAGE  (SDXL + refiner)")
    from generate import generate, GenerationConfig, generate_style_variants

    cfg = GenerationConfig(
        prompt          = "an ancient stone temple at dusk, god rays, cinematic, 8k",
        negative_prompt = "blurry, low quality, watermark, oversaturated",
        model           = "sdxl",
        width           = 1024,
        height          = 1024,
        num_inference_steps = 50,
        guidance_scale  = 7.5,
        seed            = 42,
        num_images      = 1,
        use_refiner     = True,
        output_prefix   = "t2i",
    )
    images = generate(cfg)
    logger.info(f"Generated {len(images)} image(s)")

    if RUN_VARIANTS:
        section("STYLE VARIANTS")
        generate_style_variants("a misty mountain valley at sunrise", cfg)


# ─── 2. Panorama ──────────────────────────────────────────────────────────────

def run_panorama():
    section("360° PANORAMA  (MultiDiffusion + circular wrapping)")
    from pipeline import load_pipeline
    from techniques.panorama import PanoramaGenerator, equirect_to_cubemap, save_cubemap, make_cubemap_cross
    from config import PanoramaConfig, PANO_DIR

    cfg  = PanoramaConfig(
        prompt          = "an alien world with twin moons, purple sky, glowing fauna, 360 panorama, equirectangular",
        negative_prompt = "seams, distortion, blurry, low quality",
        model           = "sdxl",
        canvas_width    = 4096,
        canvas_height   = 2048,
        tile_width      = 1024,
        tile_height     = 1024,
        tile_overlap    = 256,
        num_inference_steps = 40,
        guidance_scale  = 8.0,
        seed            = 10,
    )

    pipe = load_pipeline(cfg.model, use_refiner=False)
    gen  = PanoramaGenerator(
        pipe,
        tile_h=cfg.tile_height,
        tile_w=cfg.tile_width,
        overlap=cfg.tile_overlap,
    )

    pano = gen.generate(cfg)

    # Convert to cubemap
    logger.info("Converting panorama to cubemap …")
    faces = equirect_to_cubemap(pano, face_size=512)
    save_cubemap(faces, PANO_DIR / "cubemap_faces")
    cross = make_cubemap_cross(faces, face_size=512)
    cross.save(PANO_DIR / "cubemap_cross.png")
    logger.info(f"Cubemap cross saved → {PANO_DIR / 'cubemap_cross.png'}")


# ─── 3. Seamless texture ──────────────────────────────────────────────────────

def run_texture():
    section("SEAMLESS TEXTURE  (circular-padded latent + PBR maps)")
    from pipeline import load_pipeline
    from techniques.texture import SeamlessTextureGenerator
    from config import TEXTURE_DIR

    pipe = load_pipeline("sdxl", use_refiner=False)
    gen  = SeamlessTextureGenerator(pipe, texture_size=1024, overlap_pad=32)

    texture = gen.generate(
        prompt              = "weathered medieval cobblestone, mossy, wet, photorealistic, 4k seamless",
        negative_prompt     = "seams, blurry, low quality",
        num_inference_steps = 40,
        guidance_scale      = 7.5,
        seed                = 20,
    )
    texture.save(TEXTURE_DIR / "cobblestone_albedo.png")
    logger.info(f"Texture saved → {TEXTURE_DIR / 'cobblestone_albedo.png'}")

    # Save full PBR pack
    paths = gen.save_texture_pack(texture, TEXTURE_DIR, name="cobblestone")
    logger.info(f"PBR texture pack: {list(paths.keys())}")


# ─── 4. Score Distillation Sampling ──────────────────────────────────────────

def run_sds():
    section("SDS TEXTURE OPTIMISATION  (DreamFusion-style)")
    import torch.nn as nn
    from pipeline import load_pipeline, get_device
    from guidance.score_distillation import SDSLoss, run_sds_optimisation
    from config import SDSConfig, SDS_DIR

    device = get_device()
    cfg    = SDSConfig(
        prompt          = "a glowing crystal geode, vibrant colors, studio lighting",
        model           = "sdxl",
        texture_h       = 512,
        texture_w       = 512,
        sds_lr          = 1e-2,
        sds_iterations  = 200,   # short demo; increase to 500+ for quality
        guidance_scale  = 75.0,
        seed            = 1,
    )

    pipe     = load_pipeline(cfg.model, use_refiner=False)
    unet     = pipe.get_unet()
    vae      = pipe.get_vae()
    scheduler = pipe.get_scheduler()

    # Build text embeddings
    is_sdxl = hasattr(pipe, "base")
    if is_sdxl:
        cond, uncond, _, _ = pipe.encode_prompt(cfg.prompt, cfg.negative_prompt)
    else:
        from guidance.classifier_free import get_weighted_text_embeddings
        tok, tenc = pipe.get_tokenizer(), pipe.get_text_encoder()
        cond, uncond = get_weighted_text_embeddings(
            tok, tenc, cfg.prompt, cfg.negative_prompt, device
        )

    text_embs = torch.cat([uncond, cond]).to(device)

    # SDS loss
    sds = SDSLoss(
        unet            = unet,
        scheduler       = scheduler,
        vae             = vae,
        text_embeddings = text_embs,
        guidance_scale  = cfg.guidance_scale,
        t_min           = cfg.t_min,
        t_max           = cfg.t_max,
        device          = device,
    )

    # Learnable texture parameter [1, 3, H, W]
    torch.manual_seed(cfg.seed)
    texture = nn.Parameter(
        torch.randn(1, 3, cfg.texture_h, cfg.texture_w, device=device) * 0.1
    )

    # Simple pass-through renderer (texture IS the image)
    def renderer(t):
        return torch.tanh(t)     # maps R→[-1,1]

    optimised = run_sds_optimisation(
        sds_loss   = sds,
        texture    = texture,
        renderer   = renderer,
        num_iter   = cfg.sds_iterations,
        lr         = cfg.sds_lr,
        log_every  = 50,
        save_every = 100,
        output_dir = SDS_DIR,
    )
    logger.info(f"SDS optimisation complete → {SDS_DIR}")


# ─── 5. ControlNet ───────────────────────────────────────────────────────────

def run_controlnet():
    section("CONTROLNET  (Canny edges)")
    from PIL import Image as PILImage
    import numpy as np
    from controlnet.depth_canny import ControlNetWrapper, preprocess_canny
    from config import ControlNetConfig, IMG_DIR

    cfg = ControlNetConfig(
        model           = "sd2",
        controlnet_type = "canny",
        prompt          = "a futuristic sci-fi corridor, neon lights, metallic walls",
        negative_prompt = "blurry, low quality, distorted",
    )
    cn = ControlNetWrapper(cfg)

    # Create a synthetic sketch (or load your own)
    sketch = PILImage.fromarray(
        np.zeros((768, 768, 3), dtype=np.uint8)
    )
    # Draw a simple rectangle as a sketch placeholder
    import cv2
    arr = np.zeros((768, 768, 3), dtype=np.uint8)
    cv2.rectangle(arr, (100, 100), (668, 668), (255, 255, 255), 8)
    cv2.line(arr, (384, 100), (384, 668), (200, 200, 200), 4)
    sketch = PILImage.fromarray(arr)

    img = cn.generate_from_sketch(
        sketch  = sketch,
        prompt  = cfg.prompt,
        output_dir = IMG_DIR,
        seed    = 7,
    )
    logger.info(f"ControlNet output saved")


# ─── 6. Reward-guided generation ─────────────────────────────────────────────

def run_reward_guided():
    section("REWARD-GUIDED GENERATION  (CLIP aesthetic + latent optimisation)")
    from pipeline import load_pipeline, get_device
    from guidance.reward_guidance import CLIPAestheticReward, LatentOptimiser, reward_guided_generation
    from config import RewardGuidanceConfig, IMG_DIR

    device = get_device()
    cfg    = RewardGuidanceConfig(
        prompt          = "a vibrant cosmic nebula with swirling galaxies, stunning",
        model           = "sdxl",
        reward_scale    = 0.1,
        num_inference_steps = 40,
        latent_opt_steps    = 5,
        latent_opt_lr       = 0.03,
        seed                = 3,
    )

    pipe    = load_pipeline(cfg.model, use_refiner=False)
    reward  = CLIPAestheticReward(device=device)
    vae     = pipe.get_vae() if not hasattr(pipe, "base") else pipe.base.vae

    lat_opt = LatentOptimiser(
        vae         = vae,
        reward_fn   = reward.reward,
        num_steps   = cfg.latent_opt_steps,
        lr          = cfg.latent_opt_lr,
        reward_scale = cfg.reward_scale,
        device      = device,
    )

    image, rewards = reward_guided_generation(
        pipe            = pipe,
        reward_model    = reward,
        latent_opt      = lat_opt,
        prompt          = cfg.prompt,
        negative_prompt = cfg.negative_prompt,
        num_inference_steps = cfg.num_inference_steps,
        guidance_scale  = cfg.guidance_scale,
        seed            = cfg.seed,
        output_dir      = IMG_DIR,
    )
    logger.info(f"Reward trajectory: {rewards}")


# ─── 7. Upscaling ─────────────────────────────────────────────────────────────

def run_upscale():
    section("LATENT UPSCALING  (SDXL img2img)")
    from pipeline import load_pipeline
    from postprocess.upscale import LatentUpscaler
    from config import IMG_DIR
    from PIL import Image as PILImage

    # Use any previously generated image or create a test image
    existing = list(IMG_DIR.glob("t2i_*.png"))
    if not existing:
        logger.info("No t2i images found for upscaling, skipping.")
        return

    source   = PILImage.open(existing[0]).convert("RGB")
    # Downscale to 512 first to simulate a low-res input
    lr_image = source.resize((512, 512), PILImage.BICUBIC)

    pipe     = load_pipeline("sdxl", use_refiner=False)
    upscaler = LatentUpscaler(pipe, strength=0.35, scale_factor=2)

    hr_image = upscaler(
        image           = lr_image,
        prompt          = "an ancient stone temple at dusk, god rays, cinematic, 8k, sharp",
        negative_prompt = "blurry, artifacts, low quality",
        seed            = 42,
        steps           = 30,
    )
    hr_image.save(IMG_DIR / "upscaled.png")
    logger.info(f"Upscaled image saved → {IMG_DIR / 'upscaled.png'}")


# ─── Post-processing demo ─────────────────────────────────────────────────────

def run_color_grade():
    section("COLOR GRADING  (Lab transfer + tone curve + vibrance)")
    from postprocess.palette import ColorGrader, extract_palette, palette_swatch
    from config import IMG_DIR
    from PIL import Image as PILImage

    existing = list(IMG_DIR.glob("t2i_*.png"))
    if not existing:
        logger.info("No images to grade, skipping.")
        return

    image = PILImage.open(existing[0]).convert("RGB")

    graded = (
        ColorGrader(image)
        .vibrance(0.15)
        .contrast(1.08)
        .tone_curve(shadows=0.04, midtones=0.02, highlights=-0.03)
        .sharpen(1.15)
        .vignette(0.25)
        .get()
    )
    graded.save(IMG_DIR / "color_graded.png")

    # Extract and save palette
    colors = extract_palette(graded, n_colors=8)
    swatch = palette_swatch(colors)
    swatch.save(IMG_DIR / "palette_swatch.png")
    logger.info(f"Palette: {colors}")
    logger.info(f"Color graded image saved → {IMG_DIR / 'color_graded.png'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        choices=["text2img", "panorama", "texture", "sds", "controlnet", "reward", "upscale", "grade"],
        default=None,
        help="Run only one specific experiment.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0   = time.time()

    try:
        if args.only:
            dispatch = {
                "text2img":  run_text2img,
                "panorama":  run_panorama,
                "texture":   run_texture,
                "sds":       run_sds,
                "controlnet":run_controlnet,
                "reward":    run_reward_guided,
                "upscale":   run_upscale,
                "grade":     run_color_grade,
            }
            dispatch[args.only]()
        else:
            if RUN_TEXT2IMG:  run_text2img()
            if RUN_PANORAMA:  run_panorama()
            if RUN_TEXTURE:   run_texture()
            if RUN_SDS:       run_sds()
            if RUN_CONTROLNET: run_controlnet()
            if RUN_REWARD:    run_reward_guided()
            if RUN_UPSCALE:   run_upscale()
            run_color_grade()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed/60:.1f} min")
