import numpy as np
import pytest

from woodtex.relief import ReliefParams, compute_relief, signed_distance


def test_signed_distance_sign_and_contour():
    m = np.zeros((20, 40), bool)
    m[:, 10:20] = True
    sd = signed_distance(m)
    assert sd[5, 15] > 0 > sd[5, 30]
    assert sd[5, 10] == pytest.approx(0.5) and sd[5, 9] == pytest.approx(-0.5)


def test_heights_in_band_and_contour_at_middle():
    m = np.zeros((40, 200), bool)
    m[:, 20:24] = True        # 4 px line
    m[:, 100:180] = True      # wide band
    p = ReliefParams(width_mm=50.0, base_mm=2.0, slope=0.5, amplitude_mm=1.0, mesh_size=None)
    z = compute_relief(m, p).z_mm
    assert z.min() >= 2.0 - 1e-9 and z.max() <= 4.0 + 1e-9
    assert z[10, 140] == pytest.approx(4.0, abs=0.01)           # wide ink saturates
    assert z[10, 60] == pytest.approx(2.0, abs=0.01)            # wide background saturates
    assert z[10, 22] == pytest.approx(3.0 + np.tanh(0.5 * 0.375), abs=0.02)  # thin line: shallow bump
    assert abs(z[10, 20] - 3.0) < 0.15                          # contour pixels near the middle


def test_same_width_same_height_regardless_of_other_content():
    a = np.zeros((60, 200), bool); a[:, 20:24] = True
    b = a.copy(); b[:, 100:190] = True
    p = ReliefParams(width_mm=50.0, mesh_size=None)
    za, zb = compute_relief(a, p).z_mm, compute_relief(b, p).z_mm
    assert np.allclose(za[:, 15:30], zb[:, 15:30])


def test_invert_and_resample():
    m = np.zeros((80, 160), bool); m[:, 40:120] = True
    p = ReliefParams(width_mm=40.0, mesh_size=80)
    z = compute_relief(m, p).z_mm
    zi = compute_relief(m, ReliefParams(width_mm=40.0, mesh_size=80, invert=True)).z_mm
    assert max(z.shape) == 80
    assert np.allclose(z + zi, 2 * (2.0 + 1.0), atol=0.05)     # mirror image around the band middle
