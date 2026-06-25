# Image Generative Models Contest

## Project Structure

```
sdxl_contest/
├── config.py              # All hyperparameters and paths
├── pipeline.py            # Core SD/SDXL pipeline wrappers
├── generate.py            # Text-to-image generation entry point
├── guidance/
│   ├── __init__.py
│   ├── classifier_free.py # CFG scheduling
│   ├── score_distillation.py  # SDS / VSD loss
│   └── reward_guidance.py     # RLHF-style inference-time guidance
├── techniques/
│   ├── __init__.py
│   ├── panorama.py        # Equirectangular panorama generation
│   ├── texture.py         # UV mesh texture synthesis
│   ├── multidiffusion.py  # MultiDiffusion for large canvases
│   └── style_transfer.py  # LoRA-based style control
├── controlnet/
│   ├── __init__.py
│   └── depth_canny.py     # ControlNet wrappers (depth, canny)
├── lora/
│   ├── __init__.py
│   └── train_lora.py      # LoRA fine-tuning script
├── postprocess/
│   ├── __init__.py
│   ├── upscale.py         # Latent upscaling / real-esrgan style
│   └── palette.py         # Color grading & style enhancement
└── run_all.py             # Master demo runner
```

## Models Used
- **Stable Diffusion XL**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Stable Diffusion 2**: `Manojb/stable-diffusion-2-base`

## Techniques Implemented
1. Text-to-image with advanced CFG schedules
2. MultiDiffusion panorama generation
3. Score Distillation Sampling (SDS) for 3D-style texture baking
4. ControlNet (depth + canny) guidance
5. LoRA fine-tuning pipeline
6. Inference-time reward guidance
7. Tiled texture generation for 3D meshes
