This is a tool for turning a picture of wood into an STL (suitable for the PrusaSlicer ColorMix feature) and it is vibecoded, including the rest of this README, but I found it pretty usable. The two things to play around with are the mask extraction parameters (I found it more helpful to aim for a sparse mask with large patches of black and white, rather than a structured texture) and the slope in the STL generation step (here I believe sparse mask -> lower slope, structured mask -> higher slope). When printed with ColorMix, I particularly like using a 2:1 layer ratio (65% vs 35% of the two extruders) for a genuine woody look.

# woodtex — wood photo → black/white grain mask → relief STL

`woodtex` turns a photograph of wood grain into a 3D-printable slab whose top surface is a relief of
the grain. Two steps, which can be run separately or together:

1. **extract** – grayscale → illumination flattening → (CLAHE) → denoise → threshold → cleanup, giving a
   clean binary mask where black = dark grain.
2. **mesh** – signed distance to the grain contours → tanh relief in millimetres → watertight binary STL
   rising from a flat rectangular base, plus a height-map PNG and a hill-shaded preview PNG.

## Install

```bash
uv sync            # numpy, scipy, scikit-image, pillow (+ pytest, trimesh for the tests)
uv run woodtex --help
uv run pytest
```

## Usage

```bash
# step 1: photo -> mask (tweak the extraction, look at the PNG, repeat)
uv run woodtex extract plank.jpg -o output/plank_mask.png --method otsu --min-radius 5

# steps 2+3: mask -> STL (100 mm wide, 2 mm base, texture band of 2 x 1 mm)
uv run woodtex mesh output/plank_mask.png -o output/plank.stl --slope 0.5 --relief 1

# or everything at once
uv run woodtex run plank.jpg --out-dir output --min-radius 5 --slope 0.5 --relief 1
```

`mesh`/`run` also write `<name>_height.png` (16-bit height map) and `<name>_shade.png` (hill-shaded
preview, slopes exaggerated 4x) next to the STL.

## Step 1 — texture extraction

| flag | default | meaning |
|---|---|---|
| `--method otsu\|local\|sauvola\|quantile` | `otsu` | global Otsu gives clean band-like regions; `local`/`sauvola` are adaptive and catch finer lines but are noisier; `quantile` keeps the darkest `--fill` fraction |
| `--flatten-sigma` | 5 % of short side | subtract a very wide blur to remove vignetting / plank-to-plank brightness changes (0 = off) |
| `--no-clahe` | off | disable the local contrast enhancement |
| `--blur` | 1.0 | denoising Gaussian sigma (px); raise to 2–3 for noisy, porous wood |
| `--min-area` | max(16, 1e-4·H·W) | remove components and holes smaller than this many px |
| `--min-radius` | 0 | remove components (ink and background gaps) that contain no disk of this radius, i.e. are narrow everywhere — long thin lines go, thick regions keep their exact shape |
| `--clean-radius` | 0 | binary opening + closing radius; smooths ragged edges and trims thin protrusions off thick regions |
| `--min-fill` / `--max-fill` | 0.15 / 0.85 | safeguard: if the method leaves a dark fraction outside this range, fall back to a quantile threshold at `--fill` so the mask is never almost all black or all white |
| `--invert-mask` | off | swap ink and background |
| `--max-size` | – | downscale big photos before processing |

The log on stderr shows the threshold used and the final dark fraction.

## Steps 2 + 3 — relief and mesh

Every point's height depends only on its distance to the nearest grain contour, in physical units:

```
z = base + A + A · tanh(slope · d / A)
```

`d` is the signed distance in mm (positive inside the dark grain), `--slope` the initial slope in mm of
height per mm of distance, `A` = `--relief` the saturation amplitude in mm. So:

* contours sit at the middle of the texture band, `base + A`;
* wide dark areas flatten into plateaus at `base + 2A`, wide light areas into valleys at `base`;
* a line of width *w* only rises about `slope · w / 2` — thin lines become shallow grooves, not walls;
* nothing is normalised per image, so the same local width always gives the same depth and two
  differently cleaned masks come out with the same look.

| flag | default | meaning |
|---|---|---|
| `--width-mm` | 100 | model width; height follows the image aspect ratio |
| `--base-mm` | 2 | flat slab under the texture |
| `--slope` | 0.5 | mm of height per mm of distance; higher = sharper, deeper lines |
| `--relief` | 1 | saturation amplitude in mm; the texture band is 2× this |
| `--invert` | off | raise the light background instead of the dark grain |
| `--mesh-size` | 400 | longest side of the mesh grid in vertices (0 = image resolution); triangles ≈ 2·W·H |

The STL is binary, watertight (top grid, four walls, fan-triangulated bottom, outward winding) and
oriented so the texture is not mirrored when viewed from above.

## Tuning tips

* Look at the mask PNG before meshing; it is the whole texture. `--min-radius 5` is a good first
  cleanup for photos around 400–700 px wide: it keeps the grain bands and drops hairlines and specks.
* `--slope` sets how sharply the surface rises away from a contour. Around 0.25 suits bold, knotty
  grain with wide bands; 0.75 suits dense, straight grain whose bands are only a few pixels wide and
  would otherwise barely leave the middle of the band.
* `--relief` sets the plateau depth; `--base-mm` only adds material underneath.
* Fine grain needs a wide print: at 100 mm width a 700 px photo has about 0.14 mm per pixel, so a
  2 px line is thinner than a nozzle.

## Layout

```
src/woodtex/extract.py   step 1: photo -> mask
src/woodtex/relief.py    step 2: mask -> surface heights (mm)
src/woodtex/mesh.py      step 3: heights -> binary STL
src/woodtex/preview.py   height-map and hill-shade PNGs
src/woodtex/cli.py       argparse CLI (extract | mesh | run)
tests/                   unit tests (synthetic masks; STL validated with trimesh)
```

## License

Public domain, see [LICENSE](LICENSE) (The Unlicense). Do whatever you like with it.
