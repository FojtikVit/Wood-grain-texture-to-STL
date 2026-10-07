"""Command-line interface: ``woodtex extract | mesh | run``."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .extract import ExtractParams, extract_file, load_mask, print_log, save_mask
from .mesh import heightfield_to_stl
from .preview import save_height_png, save_shade_png
from .relief import ReliefParams, compute_relief


def _add_extract_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("step 1: texture extraction")
    g.add_argument("--method", choices=["otsu", "local", "sauvola", "quantile"], default="otsu",
                   help="thresholding method (default: otsu)")
    g.add_argument("--fill", type=float, default=0.5, help="target dark fraction for 'quantile' and the fallback")
    g.add_argument("--min-fill", type=float, default=0.15, help="safeguard: minimum acceptable dark fraction")
    g.add_argument("--max-fill", type=float, default=0.85, help="safeguard: maximum acceptable dark fraction")
    g.add_argument("--flatten-sigma", type=float, default=None,
                   help="sigma (px) of the illumination-flattening blur; default 5%% of short side, 0 disables")
    g.add_argument("--no-clahe", action="store_true", help="disable CLAHE contrast enhancement")
    g.add_argument("--blur", type=float, default=1.0, help="denoising Gaussian sigma in px (default 1.0)")
    g.add_argument("--local-block", type=int, default=None, help="block size for local/sauvola (odd; default auto)")
    g.add_argument("--local-offset", type=float, default=0.0, help="offset for 'local' threshold")
    g.add_argument("--min-area", type=int, default=None,
                   help="remove components/holes smaller than this many px (default: max(16, 1e-4*H*W))")
    g.add_argument("--min-radius", type=float, default=0.0,
                   help="drop components (ink and background) that are thinner than this radius everywhere, "
                        "i.e. contain no disk of this radius, so long thin lines go too (px, 0 = off)")
    g.add_argument("--clean-radius", type=int, default=0, help="radius of a binary opening+closing (0 = off)")
    g.add_argument("--invert-mask", action="store_true", help="flip mask polarity (ink <-> background)")
    g.add_argument("--max-size", type=int, default=None, help="downscale the photo so its longest side <= this")


def _extract_params(a: argparse.Namespace) -> ExtractParams:
    return ExtractParams(
        method=a.method, fill=a.fill, min_fill=a.min_fill, max_fill=a.max_fill,
        flatten_sigma=a.flatten_sigma, clahe=not a.no_clahe, blur=a.blur,
        local_block=a.local_block, local_offset=a.local_offset, min_area=a.min_area,
        min_radius=a.min_radius, clean_radius=a.clean_radius, invert=a.invert_mask, max_size=a.max_size,
    )


def _add_relief_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("steps 2+3: relief and mesh (all sizes in mm)")
    g.add_argument("--width-mm", type=float, default=100.0, help="model width (default 100); height follows the image")
    g.add_argument("--base-mm", type=float, default=2.0, help="flat slab under the texture (default 2)")
    g.add_argument("--slope", type=float, default=0.5,
                   help="mm of height per mm of distance from a contour (default 0.5); higher = sharper lines")
    g.add_argument("--relief", type=float, default=1.0,
                   help="saturation amplitude in mm (default 1); the texture spans 2x this above the base")
    g.add_argument("--invert", action="store_true", help="raise the light background instead of the dark grain")
    g.add_argument("--mesh-size", type=int, default=400,
                   help="longest side of the mesh grid in vertices (default 400; 0 = image resolution)")


def _relief_params(a: argparse.Namespace) -> ReliefParams:
    return ReliefParams(width_mm=a.width_mm, base_mm=a.base_mm, slope=a.slope, amplitude_mm=a.relief,
                        invert=a.invert, mesh_size=(a.mesh_size or None))


def _mesh_from_mask(mask: np.ndarray, a: argparse.Namespace, stl_path: str) -> None:
    t = time.time()
    r = compute_relief(mask, _relief_params(a))
    print(f"[2/3] relief in {time.time() - t:.1f}s", file=sys.stderr)
    print_log(r.log)
    stem = os.path.splitext(stl_path)[0]
    save_height_png(r.z_mm, stem + "_height.png")
    save_shade_png(r.z_mm, stem + "_shade.png", pitch_mm=a.width_mm / (r.z_mm.shape[1] - 1))
    t = time.time()
    info = heightfield_to_stl(r.z_mm, stl_path, a.width_mm)
    sx, sy, sz = info["size_mm"]
    print(f"[3/3] wrote {stl_path}: {info['triangles']} triangles, {sx:.1f} x {sy:.1f} x {sz:.1f} mm, "
          f"{info['bytes'] / 1e6:.1f} MB in {time.time() - t:.1f}s", file=sys.stderr)
    print(f"      previews: {stem}_height.png, {stem}_shade.png", file=sys.stderr)


def cmd_extract(a: argparse.Namespace) -> None:
    t = time.time()
    r = extract_file(a.image, _extract_params(a))
    print(f"[1/3] extracted mask from {a.image} in {time.time() - t:.1f}s", file=sys.stderr)
    print_log(r.log)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    save_mask(r.mask, a.output)
    print(f"      wrote {a.output}", file=sys.stderr)


def cmd_mesh(a: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    _mesh_from_mask(load_mask(a.mask), a, a.output)


def cmd_run(a: argparse.Namespace) -> None:
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, os.path.splitext(os.path.basename(a.image))[0])
    t = time.time()
    r = extract_file(a.image, _extract_params(a))
    print(f"[1/3] extracted mask from {a.image} in {time.time() - t:.1f}s", file=sys.stderr)
    print_log(r.log)
    save_mask(r.mask, stem + "_mask.png")
    print(f"      wrote {stem}_mask.png", file=sys.stderr)
    _mesh_from_mask(r.mask, a, stem + ".stl")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="woodtex",
                                 description="Wood photo -> black/white grain mask -> relief STL.")
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="step 1 only: photo -> binary mask PNG")
    e.add_argument("image")
    e.add_argument("-o", "--output", required=True, help="output mask PNG")
    _add_extract_args(e)
    e.set_defaults(func=cmd_extract)

    m = sub.add_parser("mesh", help="steps 2+3: mask PNG -> STL (+ preview PNGs)")
    m.add_argument("mask", help="black/white mask PNG (dark = grain)")
    m.add_argument("-o", "--output", required=True, help="output STL path")
    _add_relief_args(m)
    m.set_defaults(func=cmd_mesh)

    r = sub.add_parser("run", help="all steps: photo -> mask -> STL")
    r.add_argument("image")
    r.add_argument("--out-dir", default="output", help="directory for all outputs (default: output)")
    _add_extract_args(r)
    _add_relief_args(r)
    r.set_defaults(func=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
