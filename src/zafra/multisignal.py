"""Every NISAR signal per field per date, in one tidy table (input for exercise 4b).

Three series per field, all from NISAR L-band:

* **GCOV ascending** (track 19, frame 11, ~12:05 UTC = ~06:05 local): HV and HH backscatter.
* **GCOV descending** (track 113, frame 79, ~01:13 UTC = ~19:13 local *the previous day*):
  the same two polarizations seen from the other side of the orbit.
* **GUNW ascending coherence** (track 19, frame 11): one value per 12-day *pair* of dates.

Background for developers new to SAR
------------------------------------
* **Why ascending and descending are never averaged together.** On the ascending pass the
  satellite flies north and looks one way; on the descending pass it flies south and looks
  the other way, at a different incidence angle and at a different time of day (dawn with
  dew vs evening). The same field therefore has a different "normal" brightness on each
  pass, and rows/furrows or slopes facing one look direction brighten one pass only. Mixing
  them would create steps that are pure geometry. Each orbit is a separate series, compared
  only with itself; a classifier can still use both, as separate features.

* **What the cross-ratio cancels.** ratio_db = HV_db - HH_db, i.e. 10*log10(HV/HH). Anything
  that multiplies both polarizations by the same factor drops out of a ratio: residual
  calibration offsets between dates, imperfect terrain/incidence normalization, and to a
  large extent the overall brightening of wet soil after rain. What is left is the *share*
  of volume scattering (HV, from a tall canopy) relative to surface scattering (HH, from
  soil and stubble). Cutting cane removes the volume, so the ratio drops; a wet but uncut
  field brightens both channels and the ratio stays roughly put.

* **What coherence is.** An interferogram compares the *phase* of two radar images of the
  same place taken 12 days apart, pixel by pixel. Coherence (0..1) measures how consistent
  that phase difference is over a small neighbourhood: 1 means the scatterers inside each
  pixel did not move or change between the two dates (bare rock, buildings, a ploughed and
  untouched field), 0 means they were rearranged (noise-like phase).

* **Why coherence collapses when a field is cut.** Harvest (and the burn that often comes
  before it in Morelos) physically replaces the scatterers: the canopy disappears, trucks
  and cutters churn the soil, trash is left on the ground. The radar echo of the second date
  then has no fixed relationship with the first, so the pair covering the cut has very low
  coherence. Caveat: a tall, wind-moved cane canopy is *already* fairly incoherent at
  L-band over 12 days, so "low" must be judged against each field's own pairs, and the pair
  *after* the cut (stubble to stubble) is often noticeably more coherent.

Dates
-----
GCOV rows use the UTC acquisition date from the file (as timeseries.py). Coherence rows
carry both dates of the pair; in the wide table a pair is placed on its *secondary* (later)
date, the first date on which the change it measures is observable.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from zafra import gcov
from zafra import timeseries as ts

# --- GUNW layout (checked on CRID P05023 files, see docs/DATA_SOURCES.md) -----------------
# The 20 m "wrapped interferogram" grid. Coherence is computed there before any multilooking,
# so it resolves fields of ~1 ha (25 pixels); the unwrapped group has an 80 m version.
GUNW_IDENT = "/science/LSAR/identification"
GUNW_WRAPPED = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
GUNW_POL = "HH"
GUNW_COH = f"{GUNW_WRAPPED}/{GUNW_POL}/coherenceMagnitude"
# x/y pixel-centre coordinates of that grid (the HH subgroup has identical copies), the EPSG
# code and the quality mask.
GUNW_X = f"{GUNW_WRAPPED}/xCoordinates"
GUNW_Y = f"{GUNW_WRAPPED}/yCoordinates"
GUNW_PROJ = f"{GUNW_WRAPPED}/projection"
GUNW_MASK = f"{GUNW_WRAPPED}/mask"

# NISAR_L2_PR_GUNW_005_019_A_011_006_4000_SH_<ref start>_<ref end>_<sec start>_<sec end>_P05023_...
GUNW_RE = re.compile(
    r"NISAR_L2_[A-Z]{2}_GUNW_\d{3}_(?P<track>\d{3})_(?P<direction>[AD])_(?P<frame>\d{3})_\w+?_"
    r"(?P<ref>\d{8}T\d{6})_\d{8}T\d{6}_(?P<sec>\d{8}T\d{6})_\d{8}T\d{6}_(?P<crid>[A-Z]\d{5})_"
)

GCOV_COLUMNS = [
    "field_id",
    "orbit",
    "date",
    "HVHV_db",
    "HHHH_db",
    "ratio_db",
    "valid_px",
    "mean_looks",
    # QA, as in timeseries.SERIES_COLUMNS
    "inside_px",
    "clipped",
]
COH_COLUMNS = [
    "field_id",
    "ref_date",
    "sec_date",
    "coh_mean",
    "coh_median",
    # Lowest single-pixel coherence in the polygon. With hundreds of pixels it sits near 0 on
    # almost every pair, so it says little; kept because exercise 4b asks for it.
    "coh_min",
    "valid_px",
    # QA: pixels inside the polygon, and how many of them the GUNW mask flags (outside the
    # swath, invalid sample in either date, or an RSLC anomaly bit). Reported, not removed:
    # the rule asked for is "NaN excluded" and the product already NaNs unusable pixels.
    "inside_px",
    "mask_flagged_px",
    "clipped",
]
# Signals that go into the long table (QA columns stay in the wide table only).
GCOV_SIGNALS = ["HVHV_db", "HHHH_db", "ratio_db"]
COH_SIGNALS = ["coh_mean", "coh_median"]


def _as_fields(fields) -> gpd.GeoDataFrame:
    return fields if isinstance(fields, gpd.GeoDataFrame) else ts.load_fields(fields)


# --- GCOV ---------------------------------------------------------------------------------


def extract_gcov_series(scene_dir, fields, orbit_label: str) -> pd.DataFrame:
    """One row per (field, date) for every GCOV scene in `scene_dir`, labelled `orbit_label`.

    Reuses timeseries.extract_series (field mean of *linear* power -> dB), then pivots its
    one-row-per-polarization output into columns. `fields` is a GeoJSON path.
    """
    scenes = ts.list_scenes(scene_dir)
    if not scenes:
        raise FileNotFoundError(f"no GCOV scenes in {scene_dir}")
    long = ts.extract_series(scenes, fields, pols=("HVHV", "HHHH"))

    keys = ["field_id", "date"]
    db = long.pivot(index=keys, columns="pol", values="mean_db").add_suffix("_db")
    # A pixel counts only if valid in both polarizations, so take the smaller count.
    qa = long.groupby(keys).agg(
        valid_px=("valid_px", "min"),
        mean_looks=("mean_looks", "mean"),
        inside_px=("inside_px", "first"),
        clipped=("clipped", "first"),
    )
    df = db.join(qa).reset_index()
    df.columns.name = None
    df["orbit"] = orbit_label
    # Cross-ratio in dB is a difference (see module docstring for what it cancels).
    df["ratio_db"] = df["HVHV_db"] - df["HHHH_db"]
    return df.reindex(columns=GCOV_COLUMNS).sort_values(["field_id", "date"], ignore_index=True)


# --- GUNW coherence -----------------------------------------------------------------------


def list_pairs(gunw_dir) -> list[dict]:
    """GUNW .h5 files in `gunw_dir` with their reference/secondary dates, sorted by date."""
    pairs = []
    for path in sorted(Path(gunw_dir).glob("NISAR_L2_*_GUNW_*.h5")):
        m = GUNW_RE.match(path.name)
        if not m:
            raise ValueError(f"unrecognised GUNW file name: {path.name}")
        pairs.append(
            {
                "ref_date": pd.Timestamp(m["ref"]).date().isoformat(),
                "sec_date": pd.Timestamp(m["sec"]).date().isoformat(),
                "track": int(m["track"]),
                "frame": int(m["frame"]),
                "path": path,
            }
        )
    return sorted(pairs, key=lambda p: p["ref_date"])


def read_gunw_grid(f: h5py.File) -> dict:
    """The 20 m wrapped-interferogram grid as the dict gcov.window_for_geometry expects.

    Reads only the 1-D coordinate vectors and the scalar projection.
    """
    x = gcov._require(f, GUNW_X)[:]
    y = gcov._require(f, GUNW_Y)[:]
    dx, dy = np.diff(x), np.diff(y)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise ValueError(f"{f.filename}: {GUNW_WRAPPED} coordinate spacing is not uniform")
    shape = gcov._require(f, GUNW_COH).shape
    if shape != (len(y), len(x)):
        raise ValueError(f"{f.filename}: coherence shape {shape} != coordinate lengths")
    # No pixel_coordinate_convention attribute here (GCOV has one). The values are odd
    # multiples of 10 m on a 20 m grid, i.e. pixel centres with corners on round 20 m lines,
    # matching the NISAR spec, so we treat them as centres exactly as gcov.py does.
    return {
        "epsg": int(gcov._require(f, GUNW_PROJ)[()]),
        "x": x,
        "y": y,
        "shape": shape,
        "dx": float(dx[0]),
        "dy": float(dy[0]),
    }


def _mask_flagged(m: np.ndarray) -> np.ndarray:
    """True where the GUNW mask says a pixel is not clean (see its `description` attribute).

    Bits 0-7 hold a decimal number: units digit = secondary subswath, tens digit = reference
    subswath (0 = invalid sample), hundreds = water flag. Bits 8-23 are anomaly flags of the
    two RSLCs. 255 is the fill value (outside the radar swath).
    """
    m = m.astype("uint32")
    low = m & 0xFF
    return (m == 255) | (low % 10 == 0) | ((low // 10) % 10 == 0) | ((m >> 8) & 0xFFFF != 0)


def extract_coherence_series(gunw_dir, fields) -> pd.DataFrame:
    """One row per (field, pair): coherence statistics inside each polygon.

    Window reads only: for each field we slice the coherence (and mask) layer around the
    polygon's bounding box; a pixel counts if its centre is inside the polygon and its value
    is not NaN (the product's fill value).
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import box

    fields = _as_fields(fields)
    pairs = list_pairs(gunw_dir)
    if not pairs:
        raise FileNotFoundError(f"no GUNW pairs in {gunw_dir}")
    by_epsg: dict[int, gpd.GeoDataFrame] = {}
    rows = []

    for pair in tqdm(pairs, desc="GUNW pairs", unit="pair"):
        with h5py.File(pair["path"], "r") as f:
            ident = gcov._require(f, GUNW_IDENT)
            # Trust the file over the file name.
            for key, name in (("track", "trackNumber"), ("frame", "frameNumber")):
                if int(ident[name][()]) != pair[key]:
                    raise ValueError(f"{pair['path'].name}: {key} in file != file name")
            ref = gcov._decode(ident["referenceZeroDopplerStartTime"][()])[:10]
            sec = gcov._decode(ident["secondaryZeroDopplerStartTime"][()])[:10]

            grid = read_gunw_grid(f)
            if grid["epsg"] not in by_epsg:
                by_epsg[grid["epsg"]] = fields.to_crs(epsg=grid["epsg"])
            dx, dy = grid["dx"], grid["dy"]
            grid_box = box(
                grid["x"][0] - dx / 2,
                grid["y"][-1] + dy / 2,
                grid["x"][-1] + dx / 2,
                grid["y"][0] - dy / 2,
            )
            coh_ds, mask_ds = f[GUNW_COH], f[GUNW_MASK]

            for _, field in by_epsg[grid["epsg"]].iterrows():
                geom = field.geometry
                r, c, transform = gcov.window_for_geometry(grid, geom)
                shape = (r.stop - r.start, c.stop - c.start)
                inside = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)
                coh = coh_ds[r, c].astype("float64")
                flagged = _mask_flagged(mask_ds[r, c])
                ok = inside & np.isfinite(coh)
                vals = coh[ok]
                rows.append(
                    {
                        "field_id": field["id"],
                        "ref_date": ref,
                        "sec_date": sec,
                        "coh_mean": float(vals.mean()) if vals.size else np.nan,
                        "coh_median": float(np.median(vals)) if vals.size else np.nan,
                        "coh_min": float(vals.min()) if vals.size else np.nan,
                        "valid_px": int(vals.size),
                        "inside_px": int(inside.sum()),
                        "mask_flagged_px": int((inside & flagged).sum()),
                        "clipped": not geom.within(grid_box),
                    }
                )

    df = pd.DataFrame(rows).reindex(columns=COH_COLUMNS)
    df["ref_date"] = pd.to_datetime(df["ref_date"])
    df["sec_date"] = pd.to_datetime(df["sec_date"])
    return df.sort_values(["field_id", "ref_date"], ignore_index=True)


# --- feature table ------------------------------------------------------------------------


def build_feature_table(gcov_frames, coh: pd.DataFrame, out_dir=None):
    """Combine GCOV series (one frame per orbit) and coherence into long and wide tables.

    long: field_id, orbit, pair, date, signal, value. `pair` is "YYYYMMDD_YYYYMMDD" for
          coherence rows and empty for GCOV rows; coherence `date` is the secondary date.
    wide: one row per field x date x orbit with every signal and its QA columns. Coherence
          exists only for the ascending track, so its columns are NaN on descending rows,
          and a pair whose secondary date has no ascending GCOV scene gets a row of its own.

    If `out_dir` is given, writes series_long and series_wide as .parquet and .csv there.
    """
    g = pd.concat(gcov_frames, ignore_index=True)

    c = coh.rename(
        columns={"valid_px": "coh_valid_px", "inside_px": "coh_inside_px", "clipped": "coh_clipped"}
    )
    c["orbit"] = "asc"
    c["date"] = c["sec_date"]
    c["pair"] = c["ref_date"].dt.strftime("%Y%m%d") + "_" + c["sec_date"].dt.strftime("%Y%m%d")

    wide = g.merge(c.drop(columns=["sec_date"]), on=["field_id", "orbit", "date"], how="outer")
    wide = wide.sort_values(["field_id", "orbit", "date"], ignore_index=True)

    long_g = g.melt(
        id_vars=["field_id", "orbit", "date"],
        value_vars=GCOV_SIGNALS,
        var_name="signal",
        value_name="value",
    )
    long_g["pair"] = ""
    long_c = c.melt(
        id_vars=["field_id", "orbit", "pair", "date"],
        value_vars=COH_SIGNALS,
        var_name="signal",
        value_name="value",
    )
    long = pd.concat([long_g, long_c], ignore_index=True)
    long = long[["field_id", "orbit", "pair", "date", "signal", "value"]]
    long = long.sort_values(["field_id", "signal", "orbit", "date"], ignore_index=True)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in (("series_long", long), ("series_wide", wide)):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            df.to_csv(out_dir / f"{name}.csv", index=False)
    return long, wide
