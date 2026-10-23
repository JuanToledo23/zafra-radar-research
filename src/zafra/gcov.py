"""Small helpers to read a NISAR L2 GCOV HDF5 file.

Background for developers new to SAR
------------------------------------
* **What the pixels are.** GCOV stores *covariance terms*. The "diagonal" terms (HHHH, HVHV)
  are simply backscatter *power* for one polarization: HHHH = |HH|^2, i.e. how much of the
  radar pulse sent Horizontal came back Horizontal. HVHV is the cross-pol power (sent H,
  received V), which mostly comes from volume scattering inside vegetation.

* **gamma-0.** Raw power depends on how much ground area each radar sample "sees", which
  changes with terrain slope and viewing angle. GCOV is radiometrically terrain corrected
  (RTC) and normalised as **gamma-0**: power per unit area *as seen from the radar*. That
  removes most slope/angle effects so a field can be compared with itself across dates.
  (The file also has `rtcGammaToSigmaFactor` to convert to sigma-0 if needed.)
  Read from the file: metadata/processingInformation/parameters/rtc/
  outputBackscatterNormalizationConvention == "gamma0".

* **Linear vs dB.** The file stores *linear power* (unitless ratio, typically 0.001-1 for
  land). Humans and plots prefer decibels: dB = 10 * log10(linear). In dB, -5 dB is bright,
  -25 dB is dark; a change of -3 dB means power halved.

* **Why average in linear, then convert.** Speckle makes per-pixel power follow a skewed
  (roughly exponential/gamma) distribution. The physically meaningful field value is the
  mean *power*, so we average linear values and convert that one number to dB. Averaging
  per-pixel dB values instead computes a geometric mean, which is biased low (by ~2.5 dB for
  single-look speckle, less with more looks). `field_stats` reports both so you can see it.

Everything is read lazily with h5py: we only ever slice small windows, never full arrays
(a single full HHHH layer is 33480 x 34128 float32 = ~4.6 GB).
"""

from __future__ import annotations

import h5py
import numpy as np
from affine import Affine

IDENT = "/science/LSAR/identification"
GRIDS = "/science/LSAR/GCOV/grids"
METADATA = "/science/LSAR/GCOV/metadata"


def _require(f: h5py.File, path: str):
    """Return f[path] or fail with a message naming the missing path."""
    if path not in f:
        raise KeyError(f"{f.filename}: expected HDF5 path not found: {path}")
    return f[path]


def _decode(value):
    """HDF5 strings come back as bytes (or numpy arrays of bytes); make them plain Python."""
    if isinstance(value, bytes | np.bytes_):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_decode(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def open_gcov(path) -> h5py.File:
    """Open a GCOV file read-only. Use as a context manager: `with open_gcov(p) as f:`."""
    f = h5py.File(path, "r")
    _require(f, GRIDS)
    return f


# Field name in our dict -> dataset name under /science/LSAR/identification.
# `processing_maturity` has no dataset in the file (CRID P05023, spec 1.5.0): maturity
# (PROVISIONAL) is only in the ASF/CMR collection name, so it comes back as None.
_IDENT_FIELDS = {
    "granule_id": "granuleId",
    "track": "trackNumber",
    "frame": "frameNumber",
    "orbit_direction": "orbitPassDirection",
    "look_direction": "lookDirection",
    "absolute_orbit": "absoluteOrbitNumber",
    "start_time": "zeroDopplerStartTime",
    "end_time": "zeroDopplerEndTime",
    "product_type": "productType",
    "product_version": "productVersion",
    "product_spec_version": "productSpecificationVersion",
    "crid": "compositeReleaseId",
    "processing_type": "processingType",
    "processing_maturity": "processingMaturity",
    "processing_datetime": "processingDateTime",
    "product_doi": "productDoi",
    "frequencies": "listOfFrequencies",
}


def read_identification(f: h5py.File) -> dict:
    """Read scene identity from /science/LSAR/identification.

    Fields not present in the file are returned as None and listed under "missing".
    """
    group = _require(f, IDENT)
    out, missing = {}, []
    for key, name in _IDENT_FIELDS.items():
        if name in group:
            out[key] = _decode(group[name][()])
        else:
            out[key] = None
            missing.append(f"{IDENT}/{name}")
    out["missing"] = missing
    return out


def read_grid(f: h5py.File, freq: str = "A") -> dict:
    """Describe the map grid of one frequency band (A = 40 MHz main band, B = 5 MHz side band).

    Reads only 1-D coordinate vectors and scalars, never the image layers.
    """
    gpath = f"{GRIDS}/frequency{freq}"
    g = _require(f, gpath)

    # Diagonal covariance terms are the backscatter powers: 4 letters, first pair == second.
    pols = sorted(k for k in g if len(k) == 4 and set(k) <= {"H", "V"} and k[:2] == k[2:])

    # The projection is a scalar dataset whose *value* is the EPSG code; its attributes carry
    # the full definition (epsg_code, spatial_ref WKT, utm_zone_number, ...).
    proj = _require(f, f"{gpath}/projection")
    epsg = int(proj[()])
    if "epsg_code" in proj.attrs and int(proj.attrs["epsg_code"]) != epsg:
        raise ValueError(f"{gpath}/projection value {epsg} != epsg_code attribute")

    x = _require(f, f"{gpath}/xCoordinates")[:]
    y = _require(f, f"{gpath}/yCoordinates")[:]
    # Pixel spacing from the coordinates themselves. y decreases downwards (north-up image),
    # so dy is negative.
    dx, dy = np.diff(x), np.diff(y)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise ValueError(f"{gpath}: coordinate spacing is not uniform")

    # Range bandwidth lives with the source (L1 RSLC) swath description, not in the grid.
    swath = f"{METADATA}/sourceData/swaths/frequency{freq}"
    bw = _require(f, f"{swath}/processedRangeBandwidth")[()]

    return {
        "freq": freq,
        "path": gpath,
        "polarizations": pols,
        "epsg": epsg,
        "x": x,
        "y": y,
        "shape": (len(y), len(x)),
        "dx": float(dx[0]),
        "dy": float(dy[0]),
        # Attribute on xCoordinates; "center" means coordinates are pixel centres.
        "pixel_convention": _decode(
            f[f"{gpath}/xCoordinates"].attrs.get("pixel_coordinate_convention")
        ),
        "range_bandwidth_hz": float(bw),
        "has_number_of_looks": "numberOfLooks" in g,
        "has_mask": "mask" in g,
    }


def window_for_geometry(grid: dict, geom, scale: float = 1.0):
    """Array window (row_slice, col_slice, affine) covering a geometry's bounding box.

    `geom` must already be in the grid CRS (grid["epsg"]). `scale` > 1 grows the box around
    its centre (scale=2 -> twice as wide and tall) to show surrounding context.
    The returned affine maps (col, row) inside the window to map x/y of pixel *corners*,
    which is what rasterio expects.
    """
    minx, miny, maxx, maxy = geom.bounds
    if scale != 1.0:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        hw, hh = (maxx - minx) / 2 * scale, (maxy - miny) / 2 * scale
        minx, maxx, miny, maxy = cx - hw, cx + hw, cy - hh, cy + hh

    dx, dy = grid["dx"], grid["dy"]
    # Coordinates are pixel centres; the image's outer corner is half a pixel further out.
    left = grid["x"][0] - dx / 2
    top = grid["y"][0] - dy / 2  # dy < 0, so this is above the first centre

    col0 = int(np.floor((minx - left) / dx))
    col1 = int(np.ceil((maxx - left) / dx))
    row0 = int(np.floor((maxy - top) / dy))  # dividing by negative dy: north edge -> row0
    row1 = int(np.ceil((miny - top) / dy))

    nrows, ncols = grid["shape"]
    row0, col0 = max(row0, 0), max(col0, 0)
    row1, col1 = min(row1, nrows), min(col1, ncols)
    if row0 >= row1 or col0 >= col1:
        raise ValueError("geometry does not overlap the GCOV grid")

    transform = Affine(dx, 0.0, left + col0 * dx, 0.0, dy, top + row0 * dy)
    return slice(row0, row1), slice(col0, col1), transform


def read_window(f: h5py.File, freq: str, name: str, row_slice: slice, col_slice: slice):
    """Read one layer (HHHH, HVHV, mask, numberOfLooks, ...) for a window only.

    h5py turns the slice into a partial read: only the 512x512 compressed chunks that touch
    the window are decompressed, so this is fast and memory-light.
    """
    ds = _require(f, f"{GRIDS}/frequency{freq}/{name}")
    return ds[row_slice, col_slice]


def to_db(linear):
    """10*log10 of linear power; non-positive / non-finite values become NaN."""
    arr = np.asarray(linear, dtype="float64")
    out = np.full(arr.shape, np.nan)
    ok = np.isfinite(arr) & (arr > 0)
    out[ok] = 10 * np.log10(arr[ok])
    return out


def field_stats(arr_linear, mask_inside, nlooks=None) -> dict:
    """Statistics of linear backscatter for pixels where `mask_inside` is True.

    Non-finite and <= 0 values are treated as invalid (NaN is the GCOV fill value; 0 cannot
    be converted to dB).
    """
    arr = np.asarray(arr_linear, dtype="float64")
    valid = mask_inside & np.isfinite(arr) & (arr > 0)
    vals = arr[valid]
    if vals.size == 0:
        return {"valid_px": 0}
    px_db = 10 * np.log10(vals)
    out = {
        "valid_px": int(vals.size),
        "mean_linear": float(vals.mean()),
        "mean_db": float(10 * np.log10(vals.mean())),  # the value to use
        "mean_of_px_db": float(px_db.mean()),  # biased low; shown for comparison
        "median_db": float(np.median(px_db)),
        "std_db": float(px_db.std(ddof=1)) if vals.size > 1 else 0.0,
    }
    if nlooks is not None:
        looks = np.asarray(nlooks, dtype="float64")[valid]
        out["mean_looks"] = float(np.nanmean(looks))
    return out


def mask_info(f: h5py.File, freq: str = "A") -> dict:
    """Describe the `mask` layer from its own attributes (nothing hard-coded but the logic).

    In this product version the description says: value = subswath number for valid samples
    (1..numberOfSubSwaths), 0 = invalid (at least one partially focused/invalid radar sample
    was averaged in), _FillValue (255) = outside the radar acquisition.
    """
    gpath = f"{GRIDS}/frequency{freq}"
    ds = _require(f, f"{gpath}/mask")
    fill = ds.attrs.get("_FillValue")
    return {
        "path": f"{gpath}/mask",
        "description": _decode(ds.attrs.get("description")),
        "fill_value": None if fill is None else int(fill),
        "invalid_value": 0,
        "number_of_subswaths": _decode(f[f"{gpath}/numberOfSubSwaths"][()])
        if "numberOfSubSwaths" in f[gpath]
        else None,
    }


def valid_from_mask(mask_arr, info: dict):
    """True where the GCOV mask marks a sample as valid (not 0 and not the fill value)."""
    valid = mask_arr != info["invalid_value"]
    if info["fill_value"] is not None:
        valid &= mask_arr != info["fill_value"]
    return valid


def load_field(path, epsg: int):
    """Read a field polygon file (any CRS) and return one shapely geometry in `epsg`, plus id."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path}: no CRS defined")
    geom = gdf.to_crs(epsg=epsg).geometry.union_all()
    field_id = str(gdf["id"].iloc[0]) if "id" in gdf.columns else str(path).rsplit("/", 1)[-1]
    return geom, field_id


def extract_field(f: h5py.File, field_path, freq: str = "A") -> dict:
    """Window + masks + per-polarization stats for one field. Used by scripts 02 and 03."""
    from rasterio.features import geometry_mask

    grid = read_grid(f, freq)
    geom, field_id = load_field(field_path, grid["epsg"])
    rows, cols, transform = window_for_geometry(grid, geom)
    shape = (rows.stop - rows.start, cols.stop - cols.start)

    # Polygon -> pixel mask. A pixel counts as inside if its *centre* is in the polygon
    # (all_touched=False), which avoids mixing in edge pixels from neighbouring fields.
    inside = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)

    minfo = mask_info(f, freq) if grid["has_mask"] else None
    valid = inside.copy()
    if minfo:
        valid &= valid_from_mask(read_window(f, freq, "mask", rows, cols), minfo)

    looks = None
    if grid["has_number_of_looks"]:
        looks = read_window(f, freq, "numberOfLooks", rows, cols)
    layers, stats = {}, {}
    for pol in grid["polarizations"]:
        layers[pol] = read_window(f, freq, pol, rows, cols)
        stats[pol] = field_stats(layers[pol], valid, looks)

    return {
        "field_id": field_id,
        "geom": geom,
        "grid": grid,
        "window": {"rows": [rows.start, rows.stop], "cols": [cols.start, cols.stop]},
        "transform": transform,
        "inside_px": int(inside.sum()),
        "mask_rejected_px": int(inside.sum() - valid.sum()),
        "mask_info": minfo,
        "layers": layers,
        "stats": stats,
    }
