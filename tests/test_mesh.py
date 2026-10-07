import numpy as np
import pytest
import trimesh

from woodtex.mesh import build_triangles, heightfield_to_stl


def _load(tmp_path, z, width):
    path = tmp_path / "m.stl"
    info = heightfield_to_stl(z, str(path), width)
    mesh = trimesh.load(str(path), force="mesh")
    mesh.merge_vertices()
    return mesh, info


def test_small_mesh_watertight_and_bounds(tmp_path):
    z = 2.0 + 4.0 * np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    mesh, info = _load(tmp_path, z, 20.0)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert np.allclose(mesh.bounds[0], [0, 0, 0])
    assert np.allclose(mesh.bounds[1], [20, 30, 6])      # 3 cells of 10 mm tall, z up to 6
    assert info["triangles"] == len(mesh.faces)


def test_volume_for_flat_top(tmp_path):
    mesh, _ = _load(tmp_path, np.full((5, 7), 2.0), 60.0)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(60 * 40 * 2, rel=1e-6)


def test_normals_point_outward():
    z = 1.0 + np.random.default_rng(0).random((6, 9))
    tris = build_triangles(z, 100.0)
    m = trimesh.Trimesh(vertices=tris.reshape(-1, 3), faces=np.arange(len(tris) * 3).reshape(-1, 3), process=True)
    m.merge_vertices()
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0


def test_texture_not_mirrored():
    z = np.ones((4, 4)); z[0, 0] = 2.0            # highest point at image top-left
    v = build_triangles(z, 30.0).reshape(-1, 3)
    top = v[np.argmax(v[:, 2])]
    assert top[0] == pytest.approx(0.0) and top[1] == pytest.approx(30.0)   # -> min x, max y
