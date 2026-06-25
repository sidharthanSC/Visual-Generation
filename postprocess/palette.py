"""
postprocess/palette.py
─────────────────────────────────────────────────────────────────────────────
Color grading, palette control, and style enhancement.

All processing is purely algorithmic — no additional models required.

Features:
  • Lab-space colour transfer (Reinhard et al. 2001)
  • Cinematic LUT (3D lookup table) application
  • Vibrance, contrast, and tone curve adjustment
  • Histogram equalisation and colour palette extraction
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


# ─── Lab colour transfer ──────────────────────────────────────────────────────

def lab_color_transfer(
    source: Image.Image,
    target: Image.Image,
) -> Image.Image:
    """
    Transfer the colour palette of `target` onto `source` in Lab space.

    Reference: Reinhard et al. 2001 — "Color Transfer between Images"
    https://doi.org/10.1109/38.946629

    Args:
        source: Image whose content is kept.
        target: Image whose colour statistics are transferred.
    Returns:
        Colour-graded source image.
    """
    def _to_lab(img: Image.Image) -> np.ndarray:
        """Approximate RGB → Lab conversion."""
        rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        # sRGB → linear
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        # Linear RGB → XYZ (D65)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        xyz = rgb @ M.T
        # XYZ → Lab
        xyz /= np.array([0.95047, 1.00000, 1.08883])
        f   = np.where(xyz > 0.008856, xyz ** (1/3), 7.787 * xyz + 16/116)
        L   = 116 * f[:, :, 1] - 16
        a   = 500 * (f[:, :, 0] - f[:, :, 1])
        b   = 200 * (f[:, :, 1] - f[:, :, 2])
        return np.stack([L, a, b], axis=-1)

    def _from_lab(lab: np.ndarray) -> np.ndarray:
        """Lab → RGB."""
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        xyz = np.stack([
            np.where(fx**3 > 0.008856, fx**3, (fx - 16/116) / 7.787),
            np.where(fy**3 > 0.008856, fy**3, (fy - 16/116) / 7.787),
            np.where(fz**3 > 0.008856, fz**3, (fz - 16/116) / 7.787),
        ], axis=-1)
        xyz *= np.array([0.95047, 1.00000, 1.08883])
        # XYZ → linear RGB
        M_inv = np.linalg.inv(np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]))
        rgb = xyz @ M_inv.T
        # linear → sRGB
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.abs(rgb) ** (1/2.4) - 0.055)
        return (rgb.clip(0, 1) * 255).astype(np.uint8)

    src_lab = _to_lab(source)
    tgt_lab = _to_lab(target)

    # Per-channel mean/std matching
    result = src_lab.copy()
    for ch in range(3):
        s_mean, s_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std() + 1e-8
        t_mean, t_std = tgt_lab[:, :, ch].mean(), tgt_lab[:, :, ch].std() + 1e-8
        result[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

    return Image.fromarray(_from_lab(result))


# ─── Tone curve ───────────────────────────────────────────────────────────────

def apply_tone_curve(
    image:      Image.Image,
    shadows:    float = 0.0,    # lift shadows (+) or crush (-)
    midtones:   float = 0.0,    # brighten (+) or darken (-) mids
    highlights: float = 0.0,    # blow out (+) or pull down (-)
) -> Image.Image:
    """
    Apply an S-curve–style tone adjustment per channel.
    All values ∈ [-1, 1].
    """
    arr   = np.array(image).astype(np.float32) / 255.0

    # Build lookup table via cubic interpolation
    x = np.linspace(0, 1, 256)

    # Control points: shadows (0.1), mids (0.5), highlights (0.9)
    pts_x = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    pts_y = np.array([
        0.0,
        0.1 + shadows    * 0.1,
        0.5 + midtones   * 0.15,
        0.9 + highlights * 0.1,
        1.0,
    ]).clip(0, 1)

    lut = np.interp(x, pts_x, pts_y).clip(0, 1)
    lut_u8 = (lut * 255).astype(np.uint8)

    img_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
    result = lut_u8[img_u8]
    return Image.fromarray(result)


# ─── Color grader ─────────────────────────────────────────────────────────────

class ColorGrader:
    """
    Chainable color grading pipeline.

    Example:
        graded = (
            ColorGrader(image)
            .vibrance(0.2)
            .contrast(1.1)
            .tone_curve(shadows=0.05, highlights=-0.05)
            .sharpen(1.2)
            .get()
        )
    """

    def __init__(self, image: Image.Image):
        self.img = image.convert("RGB")

    def vibrance(self, amount: float = 0.1) -> "ColorGrader":
        """Boost saturation of less-saturated pixels (selective saturation)."""
        arr   = np.array(self.img).astype(np.float32) / 255.0
        grey  = arr.mean(axis=2, keepdims=True)
        sat   = arr.max(axis=2, keepdims=True) - arr.min(axis=2, keepdims=True)
        # Boost: pixels with low saturation get more boost
        factor = 1.0 + amount * (1.0 - sat)
        boosted = grey + factor * (arr - grey)
        self.img = Image.fromarray((boosted.clip(0, 1) * 255).astype(np.uint8))
        return self

    def contrast(self, factor: float = 1.1) -> "ColorGrader":
        """Simple contrast adjustment via PIL."""
        self.img = ImageEnhance.Contrast(self.img).enhance(factor)
        return self

    def brightness(self, factor: float = 1.05) -> "ColorGrader":
        self.img = ImageEnhance.Brightness(self.img).enhance(factor)
        return self

    def saturation(self, factor: float = 1.1) -> "ColorGrader":
        self.img = ImageEnhance.Color(self.img).enhance(factor)
        return self

    def sharpen(self, factor: float = 1.2) -> "ColorGrader":
        self.img = ImageEnhance.Sharpness(self.img).enhance(factor)
        return self

    def tone_curve(self, **kwargs) -> "ColorGrader":
        self.img = apply_tone_curve(self.img, **kwargs)
        return self

    def vignette(self, strength: float = 0.3) -> "ColorGrader":
        """Add a subtle dark vignette around the frame."""
        w, h = self.img.size
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        dist   = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        dist   = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        mask   = 1.0 - strength * dist
        mask   = mask[:, :, None]
        arr    = np.array(self.img).astype(np.float32) / 255.0
        self.img = Image.fromarray((arr * mask * 255).clip(0, 255).astype(np.uint8))
        return self

    def transfer_palette(self, target: Image.Image) -> "ColorGrader":
        self.img = lab_color_transfer(self.img, target)
        return self

    def get(self) -> Image.Image:
        return self.img

    def save(self, path: Union[str, Path]) -> "ColorGrader":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path)
        logger.info(f"Graded image → {path}")
        return self


# ─── Palette extraction ───────────────────────────────────────────────────────

def extract_palette(image: Image.Image, n_colors: int = 8) -> list[tuple[int, int, int]]:
    """
    Extract a dominant colour palette using k-means clustering.

    Args:
        image:    Input PIL image.
        n_colors: Number of palette colours to extract.
    Returns:
        List of (R, G, B) tuples.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("scikit-learn not found; returning average colour only.")
        arr  = np.array(image.convert("RGB")).reshape(-1, 3)
        mean = arr.mean(axis=0).astype(int)
        return [tuple(mean.tolist())]

    arr  = np.array(image.convert("RGB")).reshape(-1, 3).astype(np.float32)
    # Subsample for speed
    if len(arr) > 50_000:
        idx  = np.random.choice(len(arr), 50_000, replace=False)
        arr  = arr[idx]

    km     = KMeans(n_clusters=n_colors, n_init=5, random_state=0).fit(arr)
    colors = km.cluster_centers_.astype(int)
    # Sort by luminance
    lum    = 0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]
    idx    = np.argsort(lum)
    return [tuple(colors[i].tolist()) for i in idx]


def palette_swatch(colors: list[tuple[int, int, int]], size: int = 64) -> Image.Image:
    """Render a horizontal palette swatch image."""
    n   = len(colors)
    img = Image.new("RGB", (n * size, size))
    for i, c in enumerate(colors):
        img.paste(Image.new("RGB", (size, size), tuple(c)), (i * size, 0))
    return img
