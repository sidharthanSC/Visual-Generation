# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# xformers is CUDA-only and intentionally absent from requirements.txt — skip on Mac/MPS
```

Models are pulled from HuggingFace Hub on first use (SDXL base is ~7 GB). There is no test suite, linter config, or CI.

## Running experiments

```bash
python run_all.py                    # every enabled experiment, in order
python run_all.py --only panorama    # text2img | panorama | texture | sds | controlnet | reward | upscale | grade

python generate.py --prompt "a neon cyberpunk cityscape" --steps 50 --seed 42 --num_images 4
python generate.py --prompt "a mountain valley" --variants   # 4 prompt-suffix augmentations + grid

python -m lora.train_lora --dataset_dir data/lora_train --steps 1000 --rank 16   # run from repo root
```

Ordering matters in `run_all.py`: `upscale` and `grade` glob `outputs/images/t2i_*.png` and silently skip if empty, so they need a prior `text2img` run. `run_color_grade()` is not behind a `RUN_*` flag — it always runs at the end of a full pass.

## Architecture

**Config-first.** Every technique has a `@dataclass` in `config.py` (`GenerationConfig`, `PanoramaConfig`, `SDSConfig`, `LoRAConfig`, `ControlNetConfig`, `RewardGuidanceConfig`, `TextureBakeConfig`). New hyperparameters go there, not into function defaults. Importing `config.py` creates the `outputs/` subdirectories as a side effect.

**Pipeline layer (`pipeline.py`).** `SDXLPipeline` and `SD2Pipeline` wrap Diffusers pipelines and own device/dtype selection, xformers opt-in (CUDA only, failures swallowed), and the SDXL two-stage base→refiner flow. `load_pipeline(model="sdxl"|"sd2")` is the factory; `free_memory()` runs `gc.collect()` + `cuda.empty_cache()` between experiments.

### The two idioms that cut across every module

**1. `hasattr(pipe, "base")` is the SDXL-vs-SD2 discriminator.** There is no shared interface for the underlying Diffusers objects — `SDXLPipeline` exposes `.base` / `.refiner`, `SD2Pipeline` exposes `.pipe`. Every module that reaches inside a pipeline (`multidiffusion.py`, `texture.py`, `style_transfer.py`, `run_all.py`) branches on this check. Both wrappers do offer `get_unet()`, `get_vae()`, `get_scheduler()`; prefer those, and reserve the `hasattr` branch for what differs (dual tokenizers/text encoders, `load_lora_weights` on the concrete pipeline).

**2. Prompt encoding returns a 4-tuple whose last two entries are SDXL-only.** `SDXLPipeline.encode_prompt()` returns `(cond, uncond, pooled_cond, pooled_uncond)`; the SD2 path calls `guidance.classifier_free.get_weighted_text_embeddings()` (which supports `(word:1.2)` weighting syntax) and pads to `(c, u, None, None)`. Callers then build `text_embs = torch.cat([uncond, cond])` for the CFG batch — **uncond first**, matching the `pred.chunk(2)` order in the denoising loops.

### Apple silicon / MPS

`get_dtype()` returns **bf16 on MPS** (fp16 on CUDA, fp32 on CPU). This is load-bearing, not a micro-optimisation:

- Half precision is 2.4–4.1x faster than fp32 on the ops that dominate a UNet step (measured on an M5), and halves the download — SDXL's UNet is 5.1 GB rather than 10.3 GB. `variant="fp16"` is requested for any 16-bit dtype, since torch casts fp16 checkpoint files to bf16 on load.
- **bf16 rather than fp16** because the SDXL VAE overflows in fp16 and decodes to all-NaN. Diffusers' usual workaround is an fp32 upcast, but on MPS an fp32 VAE decode falls into a `group_norm` decomposition that allocates ~20 GB and dies. bf16 has fp32's exponent range at fp16's footprint.
- `prepare_vae()` enables VAE tiling **and lowers `tile_sample_min_size` to 512**. `enable_tiling()` alone does nothing at 1024×1024: Diffusers only tiles when the latent is strictly larger than `sample_size/8 == 128`, so a standard decode lands exactly on the boundary and skips tiling.

With this, a 1024×1024 decode peaks at 2.5 GB and a 4096×2048 panorama at 2.6 GB. `free_memory()` calls `torch.mps.empty_cache()` as well as the CUDA equivalent.

`get_device()` is the single source of truth — do not write `torch.device("cuda" if ... else "cpu")`, which silently sends MPS machines to the CPU.

### Custom denoising loops

`techniques/`, `guidance/score_distillation.py` and `postprocess/upscale.py` do **not** call the Diffusers pipeline `__call__`. They pull `unet` / `vae` / `scheduler` out and run their own loop so they can manipulate latents mid-trajectory. Consequences when editing them:

- SDXL requires `added_cond_kwargs={"text_embeds": pooled, "time_ids": ...}` on every UNet call. Each module builds its own time-ids helper (`_sdxl_time_ids`, `_get_sdxl_time_ids`) returning `[[h, w, 0, 0, h, w]]` in *pixel* space; SD2 must not receive these kwargs. Only `multidiffusion.py`, `panorama.py` and `texture.py` do this — `score_distillation.py` and `train_lora.py` call the UNet bare despite both defaulting to `model="sdxl"`, so expect the SDXL path there to need the kwargs added before it runs.
- Decode through `pipeline.decode_latents(vae, latents)`, never a bare `vae.decode()`. It reads `scaling_factor` off the VAE config (SDXL is **0.13025**, not the 0.18215 that SD1.x/SD2 use), upcasts an fp16 VAE, and runs under `no_grad`. `LATENT_FACTOR = 8` is still hardcoded per module.
- These loops use `pipe.base` directly, so callers pass `use_refiner=False` to `load_pipeline` — the refiner is only used by `generate.py`.

**Refiner default is inconsistent by design of the CLI:** `GenerationConfig.use_refiner` defaults to `False`, but `generate.py`'s CLI defaults it to `True` (opt out with `--no_refiner`), and `load_pipeline` defaults to `True`. Be explicit.

### Module layout

- `techniques/multidiffusion.py` — the tiled-denoising base class: latent-space tile positions with overlap, cosine-weighted accumulation into a shared canvas. `panorama.py`'s `PanoramaGenerator` **subclasses** it and overrides `__call__` to add `circular=True` horizontal wrapping (`_extract_tile` stitches the right and left slices of a seam-straddling tile into one contiguous tile; `_accumulate_tile` splits the prediction back apart at the same offset), then converts equirect→cubemap. `texture.py` takes a different route to seamlessness: `F.pad(..., mode="circular")` on the latent each step, crop back before `scheduler.step`, plus FFT seam correction and derived PBR maps.
- `guidance/classifier_free.py` — `CFGScheduler` (constant/linear/cosine/warmup_cosine annealing), CFG rescale, and PAG via UNet attention hooks. `generate.py` also carries standalone `linear_cfg_schedule` / `cosine_cfg_schedule` helpers that are currently unused by the main path.
- `guidance/score_distillation.py` — `SDSLoss.__call__(image)` expects `[B,3,H,W]` in `[-1,1]` **with autograd connectivity**; it VAE-encodes, adds noise at a random `t ∈ [t_min, t_max]`, and returns a surrogate loss. `run_sds_optimisation()` wraps it in an Adam loop saving intermediate PNGs. `VSDLoss` extends it with a monkey-patched LoRA on the frozen UNet tracking the rendered distribution. SDS uses very high guidance (50–100).
- `guidance/reward_guidance.py` — CLIP (`openai/clip-vit-large-patch14`, a third model beyond the two SD checkpoints) plus `AestheticMLP`. **The MLP is randomly initialised unless `mlp_ckpt_path` is given, so reward values are meaningless out of the box**; CLIP load failures are caught and degrade to zero reward. `LatentOptimiser` does gradient ascent on latents through the VAE decoder.
- `controlnet/depth_canny.py` — disabled by default (`RUN_CONTROLNET = False`) since it needs separate downloads. The ControlNet repo IDs actually used come from the `SD2_CONTROLNET_IDS` / `SDXL_CONTROLNET_IDS` class constants on `ControlNetWrapper`, **not** from the `*_controlnet_id` fields on `ControlNetConfig` (which are unread). Defaults to `model="sd2"` because the SD2.1 ControlNets are smaller.
- `lora/train_lora.py` — self-contained trainer with hand-rolled `LoRALinear` + `inject_lora` (no PEFT). Loads its own components rather than using `pipeline.py`, always trains in fp32, and for SDXL encodes prompts with only the *first* text encoder. `techniques/style_transfer.py` consumes the output via `load_lora_weights` + `fuse_lora`.
- `postprocess/` — `upscale.py` (img2img latent upscale + `tile_upscale` for large canvases), `palette.py` (chainable `ColorGrader`, Lab-space transfer, k-means palette extraction).

**Only two base diffusion models are permitted:** `stabilityai/stable-diffusion-xl-base-1.0` (+ refiner) and `Manojb/stable-diffusion-2-base`, both pinned in `config.py`.
