"""Step 3: surface heights (mm) -> watertight binary STL of a slab.

The model is a rectangular block: flat bottom at z = 0, four vertical walls and
the given top surface. Rows are flipped so that +y points "up" in the source
image, i.e. the texture is not mirrored when viewed from +z.
"""

from __future__ import annotations

import numpy as np


def grid_vertices(z_mm: np.ndarray, width_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, z) with z[0] being the image *bottom* row (y = 0)."""
    ny, nx = z_mm.shape
    height_mm = width_mm * (ny - 1) / (nx - 1)
    xs = np.linspace(0.0, width_mm, nx)
    ys = np.linspace(0.0, height_mm, ny)
    return xs, ys, np.flipud(np.asarray(z_mm, dtype=np.float64))


def build_triangles(z_mm: np.ndarray, width_mm: float) -> np.ndarray:
    """(n, 3, 3) triangles with outward (counter-clockwise) winding."""
    xs, ys, z = grid_vertices(z_mm, width_mm)
    X, Y = np.meshgrid(xs, ys)
    top = np.stack([X, Y, z], axis=-1)
    bot = top.copy()
    bot[..., 2] = 0.0
    tris = []

    # top surface: two triangles per cell, CCW seen from +z
    a, b, c, d = top[:-1, :-1], top[:-1, 1:], top[1:, 1:], top[1:, :-1]
    tris.append(np.stack([a, b, c], axis=-2).reshape(-1, 3, 3))
    tris.append(np.stack([a, c, d], axis=-2).reshape(-1, 3, 3))

    # side walls: quad strips from the top perimeter down to z = 0
    def wall(tp: np.ndarray, bt: np.ndarray, flip: bool) -> np.ndarray:
        q0, q1, q2, q3 = bt[:-1], bt[1:], tp[1:], tp[:-1]
        t = np.concatenate([np.stack([q0, q1, q2], axis=1), np.stack([q0, q2, q3], axis=1)])
        return t[:, ::-1] if flip else t

    tris.append(wall(top[0, :], bot[0, :], flip=False))     # y = 0     (normal -y)
    tris.append(wall(top[-1, :], bot[-1, :], flip=True))    # y = max   (normal +y)
    tris.append(wall(top[:, 0], bot[:, 0], flip=True))      # x = 0     (normal -x)
    tris.append(wall(top[:, -1], bot[:, -1], flip=False))   # x = max   (normal +x)

    # bottom: fan from the centre to the perimeter (no T-junctions), normal -z
    perim = np.concatenate([bot[0, :], bot[1:, -1], bot[-1, -2::-1], bot[-2:0:-1, 0]])
    centre = np.array([xs.mean(), ys.mean(), 0.0])
    p0, p1 = perim, np.roll(perim, -1, axis=0)
    tris.append(np.stack([np.broadcast_to(centre, p0.shape), p1, p0], axis=1))

    return np.concatenate(tris).astype(np.float64)


def triangle_normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return n / length


def write_binary_stl(tris: np.ndarray, path: str, name: str = "woodtex") -> None:
    rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tris), dtype=rec)
    data["normal"] = triangle_normals(tris).astype(np.float32)
    data["v"] = tris.astype(np.float32)
    with open(path, "wb") as f:
        f.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(data.tobytes())


def heightfield_to_stl(z_mm: np.ndarray, path: str, width_mm: float) -> dict:
    tris = build_triangles(z_mm, width_mm)
    write_binary_stl(tris, path)
    xs, ys, z = grid_vertices(z_mm, width_mm)
    return {"triangles": int(len(tris)), "size_mm": (float(xs[-1]), float(ys[-1]), float(z.max())),
            "bytes": 84 + 50 * len(tris)}
