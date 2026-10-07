import numpy as np
import pytest

from woodtex.extract import ExtractParams, extract_mask


def synthetic_wood(h=200, w=300, seed=0):
    """Smooth brightness gradient + sinusoidal grain + noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:h, :w]
    grain = 0.5 + 0.2 * np.sin(x / 6.0 + 2 * np.sin(y / 40.0))
    gradient = 0.3 * x / w
    return np.clip(grain + gradient + 0.02 * rng.standard_normal((h, w)), 0, 1).astype(np.float32)


@pytest.mark.parametrize("method", ["otsu", "local", "sauvola", "quantile"])
def test_methods_give_balanced_mask(method):
    r = extract_mask(synthetic_wood(), ExtractParams(method=method))
    assert r.mask.dtype == bool
    assert 0.3 < r.fill_fraction < 0.7
    # illumination flattening: dark fraction similar on the left and right halves
    left, right = r.mask[:, :150].mean(), r.mask[:, 150:].mean()
    assert abs(left - right) < 0.15


def test_fallback_when_threshold_degenerates():
    # nearly uniform image: otsu would split noise arbitrarily; fill safeguard must engage or stay in range
    img = np.full((100, 100), 0.5, np.float32)
    img[:, 50:] += 0.001
    r = extract_mask(img, ExtractParams(method="otsu", flatten_sigma=0, clahe=False, min_area=0))
    assert 0.15 <= r.fill_fraction <= 0.85


def test_invert_and_min_area():
    img = synthetic_wood()
    a = extract_mask(img, ExtractParams(min_area=0)).mask
    b = extract_mask(img, ExtractParams(min_area=0, invert=True)).mask
    assert np.array_equal(a, ~b)


def test_min_radius_removes_thin_lines_but_keeps_blobs():
    from woodtex.extract import remove_thin_components, cleanup
    m = np.zeros((100, 100), bool)
    m[10:12, 5:95] = True       # 2 px wide, 90 px long line: area 180, inscribed radius ~1
    m[40:60, 40:60] = True      # 20x20 blob: area 400, inscribed radius ~10
    m[45:47, 60:90] = True      # thin protrusion attached to the blob: must stay (same component)
    out, n = remove_thin_components(m, 3.0)
    assert n == 1
    assert not out[10:12, 5:95].any()
    assert out[40:60, 40:60].all() and out[45:47, 60:90].all()
    # also applied to background gaps via cleanup(): a 2 px gap through a dark area is closed
    g = np.ones((60, 60), bool); g[30:32, :] = False
    res = cleanup(g, ExtractParams(min_area=0, min_radius=3.0), [])
    assert res.all()
    # and with min_radius=0 nothing happens
    assert np.array_equal(cleanup(g, ExtractParams(min_area=0), []), g)
