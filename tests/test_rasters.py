"""Tests for exercise 9: web raster layers (zafra.rasters). Synthetic rasters only."""

import io

import numpy as np
import pytest
import rasterio
from PIL import Image

from zafra import rasters as R


def test_to_db_and_stretch():
    db = R.to_db([1.0, 0.1, 0.01, 0.0, np.nan])
    assert db[:3] == pytest.approx([0.0, -10.0, -20.0])
    assert np.isnan(db[3:]).all()
    clipped = R.clip_stretch([-30.0, -12.0, -1.0, np.nan], "hv")
    assert clipped[:3] == pytest.approx([-25.0, -12.0, -8.0])
    assert np.isnan(clipped[3]) and clipped.dtype == np.float32


def test_boxcar_averages_linear_power_over_valid_pixels():
    lin = np.full((7, 7), 0.01)
    lin[3, 3] = 0.1  # one bright pixel
    valid = np.ones_like(lin, dtype=bool)
    out = R.speckle_filter(lin, valid, size=3)
    # mean of *linear* power in the 3x3 window, not of dB values
    assert out[3, 3] == pytest.approx((8 * 0.01 + 0.1) / 9)
    assert out[0, 0] == pytest.approx(0.01)  # edge: only valid in-image neighbours count
    # an invalid pixel neither enters its neighbours' mean nor gets a value itself
    valid[3, 3] = False
    out = R.speckle_filter(lin, valid, size=3)
    assert np.isnan(out[3, 3])
    assert out[3, 2] == pytest.approx(0.01)


def test_boxcar_ignores_nonpositive_and_nan():
    lin = np.array([[0.02, 0.0, np.nan], [0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])
    out = R.speckle_filter(lin, np.ones_like(lin, dtype=bool), size=3)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])
    assert np.nanmax(np.abs(out - 0.02)) < 1e-12


def test_change_and_season():
    d0 = np.array([-12.0, -12.0, np.nan, -14.0])
    d1 = np.array([-15.0, -11.0, -12.0, np.nan])
    d2 = np.array([-13.0, -12.5, -13.0, -16.0])
    assert R.change(d0, d1)[:2] == pytest.approx([-3.0, 1.0])
    assert np.isnan(R.change(d0, d1)[2:]).all()
    s = R.season_change([d0, d1, d2])
    assert s[0] == pytest.approx(-3.0)  # deepest drop, not the last one
    assert s[1] == pytest.approx(-0.5)
    assert np.isnan(s[2])  # first date missing
    assert s[3] == pytest.approx(-2.0)  # missing middle date ignored
    assert (s[np.isfinite(s)] <= 0).all()


def test_web_grid_is_tile_aligned():
    g = R.web_grid((-99.35, 18.50, -99.05, 18.75))
    assert g["res"] == pytest.approx(R.tile_size_m(14) / 256)
    assert g["width"] % 256 == 0 and g["height"] % 256 == 0
    left, _, _, top = g["bounds_merc"]
    assert (left + R.ORIGIN) / R.tile_size_m(14) == pytest.approx(g["tx0"])
    assert (R.ORIGIN - top) / R.tile_size_m(14) == pytest.approx(g["ty0"])
    lon0, lat0, lon1, lat1 = R.grid_bounds_lonlat(g)
    assert lon0 <= -99.35 and lat0 <= 18.50 and lon1 >= -99.05 and lat1 >= 18.75


def test_tiles_nodata_and_overview_alignment():
    g = R.web_grid((-99.35, 18.50, -99.30, 18.55))
    arr = np.full((g["height"], g["width"]), -12.0, dtype="float32")
    arr[:, :256] = np.nan  # first z14 tile column has no data
    tiles = list(R.iter_tiles(arr, g, zooms=(12, 14)))
    z14 = {(x, y) for z, x, y, _ in tiles if z == 14}
    assert all(x != g["tx0"] for x, _ in z14)  # empty tiles are skipped
    assert len(z14) == (g["width"] // 256 - 1) * (g["height"] // 256)
    for _z, _x, _y, t in tiles:
        assert t.shape == (256, 256)
        # overview zooms average the same constant: alignment padding stays NaN, not 0
        assert np.nanmax(np.abs(t - -12.0)) < 1e-5
    png = R.tile_png(tiles[0][3], "hv")
    img = Image.open(io.BytesIO(png))
    assert img.mode == "P" and img.info.get("transparency") == 0
    idx = np.array(img)
    t = tiles[0][3]
    assert (idx[np.isnan(t)] == 0).all() and (idx[np.isfinite(t)] > 0).all()


def test_block_mean_ignores_nan():
    a = np.array([[1.0, np.nan], [3.0, np.nan]], dtype="float32")
    assert R.block_mean(a, 2)[0, 0] == pytest.approx(2.0)
    assert np.isnan(R.block_mean(np.full((2, 2), np.nan), 2)[0, 0])


def test_diverging_ramp_white_at_zero():
    ramp = R.ramp_for("change")
    lo, hi = R.STRETCH["change"]
    mid = [p for p, c in ramp if c == (247, 247, 247)][0]
    assert lo + mid * (hi - lo) == pytest.approx(0.0)
    assert R.to_index([0.0], lo, hi)[0] == 1 + round(mid * 254)


def test_cog_roundtrip_keeps_db_and_nodata(tmp_path):
    g = R.web_grid((-99.35, 18.50, -99.33, 18.52))
    arr = np.full((g["height"], g["width"]), -14.5, dtype="float32")
    arr[0, 0] = np.nan
    p = R.write_cog(tmp_path / "hv.tif", arr, g, {"kind": "hv"})
    with rasterio.open(p) as src:
        assert src.crs.to_epsg() == 3857
        assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        out = src.read(1)
    assert np.isnan(out[0, 0]) and out[1, 1] == pytest.approx(-14.5)
    x, y = g["transform"] @ (10.5, 10.5)
    assert R.sample_merc(p, x, y) == pytest.approx(-14.5)
