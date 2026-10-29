"""Web raster layers (exercise 9): NISAR backscatter and change as COGs and XYZ PNG tiles.

    from zafra import rasters as R
    grid = R.web_grid(site.aoi)                       # EPSG:3857 grid aligned to z14 tiles
    hv = R.speckle_filter(hv_linear, valid, size=5)   # boxcar in linear power
    hv_web = R.reproject_to_grid(hv, transform, epsg, grid)
    R.write_cog(path, R.clip_stretch(R.to_db(hv_web), "hv"), grid)
    for z, x, y, tile in R.iter_tiles(arr, grid, zooms=(10, 14)): ...

Choices, all for display (the analysis keeps using field means, zafra.gcov):

* **Speckle.** A boxcar mean of *linear* power over `size` x `size` pixels, counting only
  valid pixels (GCOV mask and finite > 0), then dB. Averaging in linear keeps the mean power
  unbiased, as for field means (gcov.field_stats). Pixels invalid in the mask stay nodata.
* **Grid.** EPSG:3857 at the zoom-14 pixel size (9.55 m in Mercator units, ~9 m on the
  ground at 18-21 deg), with the origin on a z14 tile corner and the extent snapped outwards
  to whole z14 tiles. A z14 tile is then a plain 256 x 256 window of the COG and lower zooms
  are 2^k block means of it: no warping at tiling time. Resampling from the 10 m UTM grid
  is bilinear on the already smoothed linear power.
* **Stretch.** Values are clipped to a fixed range per layer kind (STRETCH) before writing,
  so every date of a layer shares one colour scale. Change layers are differences of the
  *unclipped* dB values, clipped afterwards.
* **Tiles.** Palette PNGs (index 0 transparent = nodata) with a fixed ramp per kind;
  overview zooms average dB values (display only).
"""

from __future__ import annotations

import io
import math
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from affine import Affine

from zafra.gcov import to_db  # noqa: F401  (re-exported: the one dB conversion)

# Web Mercator: half the world width in metres and the tile size in pixels.
ORIGIN = 20037508.342789244
TILE_PX = 256
GRID_ZOOM = 14  # COG resolution = this zoom's pixel size; also the deepest tile zoom
NODATA = float("nan")

# Fixed stretch per layer kind (dB): values are clipped to it, ramps span it.
STRETCH = {
    "hv": (-25.0, -8.0),
    "hh": (-18.0, -2.0),
    "cross": (-15.0, -3.0),  # HV - HH; cane ~ -5 .. -6 dB, bare soil / water lower
    "change": (-10.0, 3.0),
}
KIND_LABEL = {
    "hv": "HV gamma-0 (dB)",
    "hh": "HH gamma-0 (dB)",
    "cross": "HV - HH cross-ratio (dB)",
    "change": "HV change (dB)",
}

# Colour ramps as (position in [0, 1] of the stretch, RGB) stops.
GREY = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]


def diverging_ramp(lo: float, hi: float) -> list:
    """Blue (loss) - white (0 dB) - red (gain); white sits at 0 dB even if lo != -hi."""
    mid = (0.0 - lo) / (hi - lo)
    return [(0.0, (33, 102, 172)), (mid, (247, 247, 247)), (1.0, (178, 24, 43))]


def ramp_for(kind: str) -> list:
    return diverging_ramp(*STRETCH[kind]) if kind == "change" else GREY


# --------------------------------------------------------------------------------- speckle


def speckle_filter(linear, valid, size: int = 5):
    """size x size boxcar mean of linear power over valid pixels; NaN where not valid.

    Normalised convolution: sum(valid * value) / sum(valid), so invalid neighbours neither
    darken the mean nor spread nodata. A valid pixel always has itself in its window.
    """
    from scipy.ndimage import uniform_filter

    arr = np.asarray(linear, dtype="float64")
    ok = np.asarray(valid, dtype=bool) & np.isfinite(arr) & (arr > 0)
    num = uniform_filter(np.where(ok, arr, 0.0), size=size, mode="constant")
    den = uniform_filter(ok.astype("float64"), size=size, mode="constant")
    out = np.full(arr.shape, np.nan)
    out[ok] = num[ok] / den[ok]
    return out


def clip_stretch(db, kind: str):
    """Clip dB values to STRETCH[kind]; NaN stays NaN. Returns float32."""
    lo, hi = STRETCH[kind]
    return np.clip(np.asarray(db, dtype="float32"), lo, hi)


def change(db_before, db_after):
    """HV(after) - HV(before) in dB (negative = loss of volume scattering, e.g. a cut)."""
    return np.asarray(db_after, dtype="float64") - np.asarray(db_before, dtype="float64")


def season_change(stack):
    """Lowest value over the dates minus the first date (<= 0 dB: deepest drop so far).

    NaN where the first date is NaN; other dates' NaN are ignored.
    """
    stack = np.asarray(stack, dtype="float64")
    with _quiet_nan():
        low = np.nanmin(stack, axis=0)
    return low - stack[0]


@contextmanager
def _quiet_nan():
    """Silence 'All-NaN slice' / 'Mean of empty slice' warnings inside nan-reductions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# ------------------------------------------------------------------------------ web grid


def tile_size_m(z: int) -> float:
    return 2 * ORIGIN / 2**z


def lonlat_to_merc(lon, lat):
    x = np.radians(lon) * 6378137.0
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * 6378137.0
    return x, y


def merc_to_lonlat(x, y):
    lon = np.degrees(np.asarray(x) / 6378137.0)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / 6378137.0)) - np.pi / 2)
    return lon, lat


def web_grid(bbox_lonlat, zoom: int = GRID_ZOOM) -> dict:
    """EPSG:3857 grid covering a lon/lat bbox, snapped outwards to whole tiles of `zoom`.

    tx0, ty0: XYZ index of the top-left tile; width / height in pixels (multiples of 256).
    """
    lon0, lat0, lon1, lat1 = bbox_lonlat
    x0, y0 = lonlat_to_merc(lon0, lat0)
    x1, y1 = lonlat_to_merc(lon1, lat1)
    t = tile_size_m(zoom)
    tx0, tx1 = math.floor((x0 + ORIGIN) / t), math.ceil((x1 + ORIGIN) / t)
    ty0, ty1 = math.floor((ORIGIN - y1) / t), math.ceil((ORIGIN - y0) / t)  # rows go south
    res = t / TILE_PX
    left, top = -ORIGIN + tx0 * t, ORIGIN - ty0 * t
    return {
        "zoom": zoom,
        "tx0": tx0,
        "ty0": ty0,
        "width": (tx1 - tx0) * TILE_PX,
        "height": (ty1 - ty0) * TILE_PX,
        "res": res,
        "transform": Affine(res, 0.0, left, 0.0, -res, top),
        "bounds_merc": (left, top - (ty1 - ty0) * t, left + (tx1 - tx0) * t, top),
        "crs": "EPSG:3857",
    }


def grid_bounds_lonlat(grid: dict) -> list[float]:
    x0, y0, x1, y1 = grid["bounds_merc"]
    (lon0, lon1), (lat0, lat1) = merc_to_lonlat([x0, x1], [y0, y1])
    return [float(lon0), float(lat0), float(lon1), float(lat1)]


def reproject_to_grid(arr, src_transform, src_epsg: int, grid: dict, resampling="bilinear"):
    """Reproject a float array (NaN = nodata) onto the web grid. Returns float32."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    dst = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    reproject(
        source=np.asarray(arr, dtype="float32"),
        destination=dst,
        src_transform=src_transform,
        src_crs=f"EPSG:{src_epsg}",
        src_nodata=np.nan,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        dst_nodata=np.nan,
        resampling=Resampling[resampling],
    )
    return dst


def write_cog(path, arr, grid: dict, tags: dict | None = None) -> Path:
    """Float32 Cloud-Optimized GeoTIFF on the web grid, NaN nodata, deflate, overviews."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "COG",
        "width": grid["width"],
        "height": grid["height"],
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": NODATA,
        "BLOCKSIZE": TILE_PX,
        "COMPRESS": "DEFLATE",
        "PREDICTOR": 3,
        "OVERVIEW_RESAMPLING": "AVERAGE",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(arr, dtype="float32"), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def sample_merc(path, x: float, y: float) -> float:
    """Value of a single-band raster at one EPSG:3857 point (NaN outside or nodata)."""
    import rasterio

    with rasterio.open(path) as src:
        row, col = src.index(x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return float("nan")
        return float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])


# --------------------------------------------------------------------------------- tiles


def block_mean(arr, factor: int, off_x: int = 0, off_y: int = 0):
    """Mean of factor x factor blocks, ignoring NaN; the array is first padded with NaN by
    (off_y, off_x) at the top-left and up to a multiple of `factor` at the bottom-right."""
    if factor == 1 and off_x == off_y == 0:
        return np.asarray(arr, dtype="float32")
    h, w = arr.shape
    H = math.ceil((h + off_y) / factor) * factor
    W = math.ceil((w + off_x) / factor) * factor
    pad = np.full((H, W), np.nan, dtype="float32")
    pad[off_y : off_y + h, off_x : off_x + w] = arr
    blocks = pad.reshape(H // factor, factor, W // factor, factor)
    with _quiet_nan():
        return np.nanmean(blocks, axis=(1, 3)).astype("float32")


def iter_tiles(arr, grid: dict, zooms=(10, GRID_ZOOM)):
    """Yield (z, x, y, 256x256 float32) for every tile of zooms[0]..zooms[1] that has data.

    `arr` is on `grid` (web_grid at zoom grid["zoom"]); deeper zooms than the grid's are not
    supported (that would only upsample).
    """
    zg = grid["zoom"]
    z0, z1 = zooms
    if z1 > zg:
        raise ValueError(f"max zoom {z1} > grid zoom {zg}")
    for z in range(z0, z1 + 1):
        f = 2 ** (zg - z)
        # the grid's top-left tile at zoom z, and where the grid starts inside it (grid px)
        tx, ty = grid["tx0"] // f, grid["ty0"] // f
        off_x = (grid["tx0"] - tx * f) * TILE_PX
        off_y = (grid["ty0"] - ty * f) * TILE_PX
        # pad in grid pixels so blocks line up with zoom-z tile edges, then average
        small = block_mean(arr, f, off_x, off_y)
        nty, ntx = small.shape[0] // TILE_PX, small.shape[1] // TILE_PX
        for j in range(nty):
            for i in range(ntx):
                tile = small[j * TILE_PX : (j + 1) * TILE_PX, i * TILE_PX : (i + 1) * TILE_PX]
                if np.isfinite(tile).any():
                    yield z, tx + i, ty + j, tile


def palette(ramp: list) -> list[int]:
    """768-entry PNG palette: index 0 = nodata (transparent), 1..255 along the ramp."""
    pos = np.array([p for p, _ in ramp])
    rgb = np.array([c for _, c in ramp], dtype="float64")
    t = np.linspace(0, 1, 255)
    cols = np.stack([np.interp(t, pos, rgb[:, k]) for k in range(3)], axis=1)
    cols = np.vstack([[0, 0, 0], np.round(cols)]).astype(int)
    return cols.ravel().tolist()


def to_index(tile, lo: float, hi: float):
    """Map values to palette indices 1..255 over [lo, hi]; NaN -> 0."""
    t = (np.asarray(tile, dtype="float64") - lo) / (hi - lo)
    idx = 1 + np.round(np.clip(t, 0, 1) * 254)
    return np.where(np.isfinite(t), idx, 0).astype("uint8")


def tile_png(tile, kind: str) -> bytes:
    from PIL import Image

    lo, hi = STRETCH[kind]
    img = Image.fromarray(to_index(tile, lo, hi), mode="P")
    img.putpalette(palette(ramp_for(kind)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, transparency=0)
    return buf.getvalue()


def write_tiles(arr, grid: dict, kind: str, out_dir, zooms=(10, GRID_ZOOM)) -> dict:
    """Write {z}/{x}/{y}.png under out_dir; returns tile count and bytes."""
    out_dir = Path(out_dir)
    n = size = 0
    for z, x, y, tile in iter_tiles(arr, grid, zooms):
        p = out_dir / str(z) / str(x) / f"{y}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        data = tile_png(tile, kind)
        p.write_bytes(data)
        n += 1
        size += len(data)
    return {"tiles": n, "bytes": size}
