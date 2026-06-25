"""
config.py — Central configuration for all contest experiments.
All hyperparameters, model IDs, and output paths live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple


# ─── Model IDs (only these two are permitted) ──────────────────────────────────
SDXL_MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
SD2_MODEL_ID   = "Manojb/stable-diffusion-2-base"

# ─── Output directories ────────────────────────────────────────────────────────
ROOT_DIR      = Path("outputs")
IMG_DIR       = ROOT_DIR / "images"
PANO_DIR      = ROOT_DIR / "panoramas"
TEXTURE_DIR   = ROOT_DIR / "textures"
SDS_DIR       = ROOT_DIR / "sds"
LORA_DIR      = ROOT_DIR / "lora_weights"
LOG_DIR       = ROOT_DIR / "logs"

for _d in [IMG_DIR, PANO_DIR, TEXTURE_DIR, SDS_DIR, LORA_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─── Generation config ─────────────────────────────────────────────────────────
@dataclass
class GenerationConfig:
    prompt: str                        = "a breathtaking landscape"
    negative_prompt: str               = (
        "blurry, low quality, ugly, deformed, watermark, text, "
        "signature, oversaturated, dull, flat"
    )
    model: str                         = "sdxl"          # "sdxl" | "sd2"
    width: int                         = 1024
    height: int                        = 1024
    num_inference_steps: int           = 20
    guidance_scale: float              = 7.5
    seed: Optional[int]                = 42
    num_images: int                    = 1
    # SDXL two-stage refinement
    use_refiner: bool                  = False
    high_noise_frac: float             = 0.8             # switch point
    # Output
    output_dir: Path                   = IMG_DIR
    output_prefix: str                 = "gen"


# ─── Panorama config ───────────────────────────────────────────────────────────
@dataclass
class PanoramaConfig:
    prompt: str                        = "a vast alien canyon at golden hour, 360 panorama"
    negative_prompt: str               = "blurry, low quality, seams, distortion"
    model: str                         = "sdxl"
    # Canvas dimensions (width should be ~2× height for equirectangular)
    canvas_width: int                  = 4096
    canvas_height: int                 = 2048
    # Tile settings for MultiDiffusion
    tile_width: int                    = 1024
    tile_height: int                   = 1024
    tile_overlap: int                  = 256
    # Sampling
    num_inference_steps: int           = 50
    guidance_scale: float              = 8.0
    seed: Optional[int]                = 0
    output_dir: Path                   = PANO_DIR


# ─── Score Distillation Sampling (SDS / VSD) config ───────────────────────────
@dataclass
class SDSConfig:
    prompt: str                        = "a golden dragon figurine, 3D render, studio lighting"
    negative_prompt: str               = "flat, 2D, painting, sketch, blurry"
    model: str                         = "sdxl"
    # Texture resolution
    texture_h: int                     = 512
    texture_w: int                     = 512
    # SDS optimisation
    sds_lr: float                      = 1e-2
    sds_iterations: int                = 500
    t_min: float                       = 0.02           # min noise level
    t_max: float                       = 0.98           # max noise level
    guidance_scale: float              = 100.0          # high CFG for SDS
    # Variational Score Distillation extras
    use_vsd: bool                      = False
    vsd_phi_lr: float                  = 1e-4
    vsd_phi_steps: int                 = 1
    seed: Optional[int]                = 1
    output_dir: Path                   = SDS_DIR


# ─── LoRA fine-tuning config ───────────────────────────────────────────────────
@dataclass
class LoRAConfig:
    model: str                         = "sdxl"
    # Data
    dataset_dir: str                   = "data/lora_train"
    instance_prompt: str               = "a photo of sks style"
    class_prompt: str                  = "a photo"
    # LoRA rank & alpha
    lora_rank: int                     = 16
    lora_alpha: int                    = 32
    target_modules: List[str]          = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2"
    ])
    # Training
    train_batch_size: int              = 1
    gradient_accumulation_steps: int   = 4
    learning_rate: float               = 1e-4
    lr_scheduler: str                  = "cosine"
    lr_warmup_steps: int               = 50
    max_train_steps: int               = 1000
    mixed_precision: str               = "fp16"          # "no" | "fp16" | "bf16"
    output_dir: Path                   = LORA_DIR
    save_every: int                    = 250


# ─── ControlNet config ─────────────────────────────────────────────────────────
@dataclass
class ControlNetConfig:
    model: str                         = "sd2"
    controlnet_type: str               = "canny"         # "canny" | "depth"
    # ControlNet model IDs
    canny_controlnet_id: str           = "thibaud/controlnet-sd21-canny-diffusers"
    depth_controlnet_id: str           = "thibaud/controlnet-sd21-depth-diffusers"
    # Sampling
    prompt: str                        = "a futuristic city, neon lights, cinematic"
    negative_prompt: str               = "blurry, low quality"
    num_inference_steps: int           = 40
    guidance_scale: float              = 7.5
    controlnet_conditioning_scale: float = 0.8
    seed: Optional[int]                = 7
    output_dir: Path                   = IMG_DIR


# ─── Reward guidance config ────────────────────────────────────────────────────
@dataclass
class RewardGuidanceConfig:
    prompt: str                        = "a vibrant oil painting of a cosmic nebula"
    negative_prompt: str               = "blurry, low quality"
    model: str                         = "sdxl"
    # CLIP aesthetic reward
    reward_scale: float                = 0.15
    num_inference_steps: int           = 50
    guidance_scale: float              = 7.5
    seed: Optional[int]                = 3
    # Latent optimisation passes
    latent_opt_steps: int              = 5
    latent_opt_lr: float               = 0.05
    output_dir: Path                   = IMG_DIR


# ─── Texture baking config ─────────────────────────────────────────────────────
@dataclass
class TextureBakeConfig:
    obj_path: str                      = "assets/mesh.obj"
    prompt: str                        = "worn stone brick texture, seamless, 4k"
    negative_prompt: str               = "blurry, seams, low quality"
    model: str                         = "sdxl"
    texture_size: int                  = 1024
    num_views: int                     = 6              # multiview bake
    camera_distance: float             = 2.5
    # SDS settings reused
    sds_lr: float                      = 5e-3
    sds_iterations: int                = 300
    guidance_scale: float              = 50.0
    seed: Optional[int]                = 5
    output_dir: Path                   = TEXTURE_DIR
