"""Step 2: binary mask -> surface heights in mm (signed-distance relief).

Every point's height depends only on its distance to the nearest ink/background
contour, in physical units:

    z = base + A + A * tanh(slope * d / A)

where ``d`` is the signed distance in mm (positive inside the dark grain, negative
outside), ``slope`` the initial slope in mm of height per mm of distance and ``A``
the saturation amplitude in mm. Contours sit at the middle of the texture band,
wide areas flatten into plateaus (ink) and valleys (background) at ``base + 2A``
and ``base``, and a line of width ``w`` only rises about ``slope * w / 2``, so thin
lines become shallow grooves rather than fragile walls. Because nothing is
normalised per image, the same local width always gives the same depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import transform


@dataclass
class ReliefParams:
    width_mm: float = 100.0     # model width; the height follows the image aspect ratio
    base_mm: float = 2.0        # flat slab under the texture
    slope: float = 0.5          # mm of height per mm of distance from the contour
    amplitude_mm: float = 1.0   # saturation amplitude; total texture band = 2 * amplitude
    invert: bool = False        # raise the background instead of the dark grain
    mesh_size: int | None = 400 # longest side of the mesh grid in vertices (None = image resolution)


@dataclass
class ReliefResult:
    z_mm: np.ndarray            # (ny, nx) surface heights in mm, row 0 = image top
    log: list[str]


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Pixel distance to the contour: positive inside ink, negative outside, ~0 on the contour."""
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    return np.where(mask, d_in - 0.5, -(d_out - 0.5)).astype(np.float32)


def resample(field: np.ndarray, mesh_size: int | None) -> np.ndarray:
    """Area-averaging downsample so the longest side is <= mesh_size vertices."""
    if mesh_size is None or max(field.shape) <= mesh_size:
        return field.astype(np.float32)
    scale = mesh_size / max(field.shape)
    shape = (max(2, round(field.shape[0] * scale)), max(2, round(field.shape[1] * scale)))
    return transform.resize(field, shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)


def compute_relief(mask: np.ndarray, p: ReliefParams | None = None) -> ReliefResult:
    p = p or ReliefParams()
    mask = mask.astype(bool)
    if p.invert:
        mask = ~mask
    log: list[str] = []
    pitch = p.width_mm / (mask.shape[1] - 1)
    a = max(p.amplitude_mm, 1e-6)
    d_mm = signed_distance(mask) * pitch
    h = a * np.tanh(p.slope * d_mm / a)
    h = resample(h, p.mesh_size)
    z = (p.base_mm + a + h).astype(np.float64)
    log.append(f"relief: {pitch:.3f} mm/px, slope {p.slope} mm/mm, amplitude {p.amplitude_mm} mm"
               + (", inverted" if p.invert else ""))
    log.append(f"surface {z.shape[1]}x{z.shape[0]} vertices, heights {z.min():.2f}..{z.max():.2f} mm "
               f"(base {p.base_mm}, texture band {p.base_mm:.2f}..{p.base_mm + 2 * a:.2f})")
    return ReliefResult(z_mm=z, log=log)
