"""Step 1: turn a wood photo into a clean binary (black/white) texture mask.

Pipeline
--------
1. grayscale
2. illumination flattening (subtract a very wide Gaussian blur) so a single
   threshold works across the whole plank, optional CLAHE for contrast
3. light denoising
4. thresholding (otsu | local | sauvola | quantile) with a fill-fraction
   safeguard so the result is never (almost) all black or all white
5. morphological cleanup (remove speckle and pinholes)

Convention: in the returned mask ``True`` means *dark grain* ("ink").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, transform
from skimage.color import rgb2gray


@dataclass
class ExtractParams:
    method: str = "otsu"          # otsu | local | sauvola | quantile
    fill: float = 0.5             # target dark fraction for 'quantile' (and the fallback)
    min_fill: float = 0.15        # safeguard bounds on the dark fraction
    max_fill: float = 0.85
    flatten_sigma: float | None = None  # None -> 5% of the shorter side; 0 disables
    clahe: bool = True
    clahe_clip: float = 0.01
    blur: float = 1.0             # denoising Gaussian sigma in px
    local_block: int | None = None      # block size for 'local'/'sauvola'; None -> auto
    local_offset: float = 0.0     # offset for 'local' (in gray units, 0..1)
    min_area: int | None = None   # min component area kept; None -> max(16, 1e-4 * H * W)
    min_radius: float = 0.0       # drop components (ink and background) whose widest point is thinner than this
    clean_radius: int = 0         # radius of a binary opening+closing; 0 disables
    invert: bool = False          # flip polarity of the result
    max_size: int | None = None   # downscale the photo so its longest side is <= this


@dataclass
class ExtractResult:
    mask: np.ndarray              # bool (H, W); True = ink
    gray: np.ndarray              # float (H, W) preprocessed grayscale, 0..1
    threshold_method: str         # method actually used (may be the fallback)
    fill_fraction: float          # fraction of True pixels in the final mask
    log: list[str]


def load_gray(path: str, max_size: int | None = None) -> np.ndarray:
    """Load an image as float grayscale in [0, 1], optionally downscaled."""
    img = Image.open(path)
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb2gray(arr).astype(np.float32)
    if max_size is not None and max(gray.shape) > max_size:
        scale = max_size / max(gray.shape)
        new_shape = (max(1, round(gray.shape[0] * scale)), max(1, round(gray.shape[1] * scale)))
        gray = transform.resize(gray, new_shape, anti_aliasing=True, preserve_range=True).astype(np.float32)
    return gray


def _rescale01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def preprocess(gray: np.ndarray, p: ExtractParams, log: list[str]) -> np.ndarray:
    """Illumination flattening, optional CLAHE, denoising. Returns float in [0, 1]."""
    g = gray.astype(np.float32)
    h, w = g.shape
    sigma = p.flatten_sigma if p.flatten_sigma is not None else 0.05 * min(h, w)
    if sigma > 0:
        background = ndi.gaussian_filter(g, sigma=sigma, mode="reflect")
        g = g - background
        log.append(f"flattened illumination with sigma={sigma:.1f}px")
    g = _rescale01(g)
    if p.clahe:
        g = exposure.equalize_adapthist(g, clip_limit=p.clahe_clip).astype(np.float32)
        log.append(f"applied CLAHE (clip={p.clahe_clip})")
    if p.blur > 0:
        g = ndi.gaussian_filter(g, sigma=p.blur, mode="reflect")
    return g


def _threshold(g: np.ndarray, p: ExtractParams, method: str, log: list[str]) -> np.ndarray:
    """Return boolean 'ink' mask (dark pixels) for the chosen method."""
    h, w = g.shape
    if method == "otsu":
        t = filters.threshold_otsu(g)
        log.append(f"otsu threshold = {t:.3f}")
        return g < t
    if method == "quantile":
        t = np.quantile(g, p.fill)
        log.append(f"quantile threshold (fill={p.fill}) = {t:.3f}")
        return g < t
    block = p.local_block if p.local_block is not None else max(15, (min(h, w) // 8) | 1)
    if block % 2 == 0:
        block += 1
    if method == "local":
        t = filters.threshold_local(g, block_size=block, method="gaussian", offset=p.local_offset)
        log.append(f"local gaussian threshold, block={block}, offset={p.local_offset}")
        return g < t
    if method == "sauvola":
        t = filters.threshold_sauvola(g, window_size=block)
        log.append(f"sauvola threshold, window={block}")
        return g < t
    raise ValueError(f"unknown threshold method: {method}")


def remove_thin_components(mask: np.ndarray, min_radius: float) -> tuple[np.ndarray, int]:
    """Remove connected components of ``mask`` that contain no disk of radius ``min_radius``.

    The criterion is the maximum of the Euclidean distance transform inside each component
    (its inscribed radius), so a long, thin line is removed even if its area is large, while a
    thick region keeps its exact shape, thin protrusions included. Returns (mask, n_removed).
    """
    labels, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask, 0
    edt = ndi.distance_transform_edt(mask)
    widest = ndi.maximum(edt, labels, index=np.arange(1, n + 1))
    thin = np.flatnonzero(np.asarray(widest) < min_radius) + 1
    if thin.size == 0:
        return mask, 0
    out = mask & ~np.isin(labels, thin)
    return out, int(thin.size)


def cleanup(mask: np.ndarray, p: ExtractParams, log: list[str]) -> np.ndarray:
    h, w = mask.shape
    min_area = p.min_area if p.min_area is not None else max(16, int(1e-4 * h * w))
    if min_area > 0:
        before = mask.copy()
        mask = morphology.remove_small_objects(mask, max_size=min_area - 1)
        mask = morphology.remove_small_holes(mask, max_size=min_area - 1)
        # also remove small specks of the *other* colour that "remove_small_objects" doesn't see
        mask = ~morphology.remove_small_objects(~mask, max_size=min_area - 1)
        mask = ~morphology.remove_small_holes(~mask, max_size=min_area - 1)
        changed = int(np.count_nonzero(before != mask))
        log.append(f"removed components/holes < {min_area}px ({changed} pixels changed)")
    if p.min_radius > 0:
        mask, n_ink = remove_thin_components(mask, p.min_radius)
        inv, n_bg = remove_thin_components(~mask, p.min_radius)
        mask = ~inv
        log.append(f"removed components thinner than radius {p.min_radius}px everywhere "
                   f"({n_ink} ink, {n_bg} background)")
    if p.clean_radius > 0:
        fp = morphology.disk(p.clean_radius)
        mask = morphology.opening(mask, fp).astype(bool)
        mask = morphology.closing(mask, fp).astype(bool)
        log.append(f"binary opening+closing with radius {p.clean_radius}")
    return mask


def extract_mask(gray: np.ndarray, p: ExtractParams | None = None) -> ExtractResult:
    """Full step-1 pipeline on a grayscale image (float, 0..1)."""
    p = p or ExtractParams()
    log: list[str] = []
    g = preprocess(gray, p, log)

    method = p.method
    mask = _threshold(g, p, method, log)
    frac = float(mask.mean())
    if not (p.min_fill <= frac <= p.max_fill):
        log.append(
            f"WARNING: {method} gave dark fraction {frac:.3f} outside "
            f"[{p.min_fill}, {p.max_fill}]; falling back to quantile fill={p.fill}"
        )
        method = "quantile"
        mask = _threshold(g, p, method, log)

    mask = cleanup(mask, p, log)
    if p.invert:
        mask = ~mask
        log.append("inverted polarity")
    frac = float(mask.mean())
    log.append(f"final ink fraction = {frac:.3f}")
    return ExtractResult(mask=mask, gray=g, threshold_method=method, fill_fraction=frac, log=log)


def extract_file(path: str, p: ExtractParams | None = None) -> ExtractResult:
    p = p or ExtractParams()
    gray = load_gray(path, p.max_size)
    return extract_mask(gray, p)


def save_mask(mask: np.ndarray, path: str) -> None:
    """Save as 8-bit PNG: ink (True) -> black, background -> white."""
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")
    img.save(path)


def load_mask(path: str) -> np.ndarray:
    """Load a black/white PNG back into a bool mask (dark pixels -> True)."""
    arr = np.asarray(Image.open(path).convert("L"))
    return arr < 128


def print_log(log: list[str], file=sys.stderr) -> None:
    for line in log:
        print("  " + line, file=file)
