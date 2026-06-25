# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Note: xformers is CUDA-only — skip on Mac/MPS
```

## Running experiments

```bash
# Run all techniques sequentially
python run_all.py

# Run a single technique
python run_all.py --only panorama   # or: text2img, texture, sds, controlnet, reward, upscale, grade

# Run text-to-image directly with CLI flags
python generate.py --prompt "a neon cyberpunk cityscape" --model sdxl --steps 50 --seed 42 --num_images 4

# Generate style variants (4 automatic prompt augmentations)
python generate.py --prompt "a mountain valley" --variants
```

## Architecture

All outputs land in `outputs/` subdirectories (images, panoramas, textures, sds, lora_weights, logs) — these dirs are auto-created by `config.py` on import.

**Config-first design.** Every technique has a corresponding `@dataclass` in `config.py` (e.g. `GenerationConfig`, `PanoramaConfig`, `SDSConfig`). All hyperparameters, model IDs, and paths live there — never scattered across modules.

**Pipeline layer (`pipeline.py`).** `SDXLPipeline` and `SD2Pipeline` are thin wrappers that handle device/dtype selection (fp16 on CUDA, fp32 on MPS/CPU), xformers opt-in, and the SDXL two-stage base→refiner flow. `load_pipeline(model="sdxl"|"sd2")` is the factory used everywhere. `free_memory()` runs `gc.collect()` + `cuda.empty_cache()` after each experiment.

**Module layout:**
- `guidance/` — CFG scheduling (`classifier_free.py`), SDS/VSD loss (`score_distillation.py`), CLIP-aesthetic reward guidance (`reward_guidance.py`)
- `techniques/` — panorama via MultiDiffusion with circular tiling (`panorama.py`), seamless UV texture synthesis (`texture.py`), style transfer (`style_transfer.py`)
- `controlnet/` — ControlNet wrappers for depth and canny edge conditioning (`depth_canny.py`); disabled by default (`RUN_CONTROLNET = False`) because it requires separate model downloads
- `lora/` — LoRA fine-tuning loop (`train_lora.py`)
- `postprocess/` — latent upscaling via img2img (`upscale.py`), Lab-space color grading + palette extraction (`palette.py`)

**Only two base models are used:** `stabilityai/stable-diffusion-xl-base-1.0` (+ optional refiner `stabilityai/stable-diffusion-xl-refiner-1.0`) and `Manojb/stable-diffusion-2-base`. These are fetched from HuggingFace Hub on first use.

**SDS/VSD pattern.** `SDSLoss.__call__(image)` expects a rendered image tensor `[B,3,H,W]` in `[-1,1]` with autograd connectivity. `run_sds_optimisation()` wraps it in an Adam loop and saves intermediate PNGs. `VSDLoss` extends `SDSLoss` with a monkey-patched LoRA on the frozen UNet that tracks the rendered distribution.

**Feature flags in `run_all.py`.** `RUN_*` booleans at the top of the file toggle individual experiments without modifying code. The `--only <name>` CLI arg dispatches to a single runner function.
