"""PNG previews of the surface: 16-bit height map and a hill-shaded view."""

from __future__ import annotations

import numpy as np
from PIL import Image


def save_height_png(z_mm: np.ndarray, path: str) -> None:
    """16-bit grayscale PNG, black = lowest point, white = highest."""
    lo, hi = float(z_mm.min()), float(z_mm.max())
    norm = (z_mm - lo) / (hi - lo) if hi > lo else np.zeros_like(z_mm)
    Image.fromarray((norm * 65535).astype(np.uint16), mode="I;16").save(path)


def save_shade_png(z_mm: np.ndarray, path: str, pitch_mm: float, exaggeration: float = 4.0) -> None:
    """Lambertian hill-shade of the real geometry, slopes vertically exaggerated for readability."""
    z = np.asarray(z_mm, dtype=np.float64) * exaggeration / pitch_mm   # heights in grid units
    gy, gx = np.gradient(z)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    lo, hi = float(z_mm.min()), float(z_mm.max())
    norm = (z_mm - lo) / (hi - lo) if hi > lo else np.zeros_like(z_mm)
    img = 0.75 * shade + 0.25 * norm
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L").save(path)
