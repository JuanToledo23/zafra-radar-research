"""Per-field backscatter time series across many GCOV scenes.

Why a time series works at all
------------------------------
* **Same track + frame + direction = same viewing geometry.** Every scene here is track 19,
  frame 11, ascending: the satellite flies the same orbit every 12 days and looks at each
  field from the same side and the same incidence angle. Brightness changes between dates
  are then mostly changes *on the ground* (cane cut, regrowth, wet soil), not geometry.
  The products are also on the same map grid, so a polygon hits the same pixels each date.
* **Dates are acquisition dates.** The date of each row is when the radar imaged the field
  (zeroDopplerStartTime, ~12:05 UTC = ~06:05 local in Morelos), not when it was processed.
  A harvest seen as a drop between two rows happened somewhere in that 12-day interval.
* **Average in linear, plot in dB.** Speckle makes single pixels noisy and skewed; the
  physically meaningful field value is the mean *power* (linear). We average linear power,
  then convert that one number to dB because dB turns ratios into differences: a -3 dB step
  means half the power whatever the starting level, which makes steps easy to compare
  across fields and polarizations.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from zafra import gcov

# NISAR_L2_PR_GCOV_015_019_A_011_4005_DHDH_A_20260311T120544_20260311T120619_P05023_N_F_J_001.h5
#                  cyc trk dir frm              start           end             CRID
# Only the fields we rely on are parsed; the date/track/frame are cross-checked against the
# file's own identification group when the scene is opened.
GRANULE_RE = re.compile(
    r"NISAR_L2_(?P<config>[A-Z]{2})_GCOV_(?P<cycle>\d{3})_(?P<track>\d{3})_(?P<direction>[AD])_"
    r"(?P<frame>\d{3})_\w+?_(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6})_(?P<crid>[A-Z]\d{5})_"
)

SERIES_COLUMNS = [
    "field_id",
    "date",
    "pol",
    "valid_px",
    "mean_linear",
    "mean_db",
    "median_db",
    "std_db",
    "mean_looks",
    # Extra QA columns (not used for plotting): why pixels were dropped.
    "inside_px",
    "outside_swath_px",
    "invalid_px",
    "clipped",
]


def list_scenes(nisar_dir) -> list[dict]:
    """Find GCOV .h5 files (ignores .part/.lock) and parse their names. Sorted by date."""
    scenes = []
    for path in sorted(Path(nisar_dir).glob("NISAR_L2_*_GCOV_*.h5")):
        m = GRANULE_RE.match(path.name)
        if not m:
            raise ValueError(f"unrecognised GCOV file name: {path.name}")
        start = pd.Timestamp(m["start"]).tz_localize("UTC")
        scenes.append(
            {
                "date": start.date().isoformat(),
                "start_utc": start,
                "track": int(m["track"]),
                "frame": int(m["frame"]),
                "direction": {"A": "Ascending", "D": "Descending"}[m["direction"]],
                "cycle": int(m["cycle"]),
                "crid": m["crid"],
                "path": path,
            }
        )
    return sorted(scenes, key=lambda s: s["start_utc"])


def load_fields(fields_geojson) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(fields_geojson)
    if gdf.crs is None:
        raise ValueError(f"{fields_geojson}: no CRS defined")
    if "id" not in gdf.columns or gdf["id"].duplicated().any():
        raise ValueError(f"{fields_geojson}: every feature needs a unique 'id' property")
    return gdf


def extract_series(scenes, fields_geojson, pols=("HVHV", "HHHH"), freq="A") -> pd.DataFrame:
    """One row per (field, date, polarization).

    Each scene is opened once and every field is read from it before moving on: opening a
    7 GB HDF5 file and reading its coordinate vectors is the slow part, the small field
    windows are cheap.
    """
    fields = load_fields(fields_geojson)
    by_epsg: dict[int, gpd.GeoDataFrame] = {}  # reproject once per grid CRS, not per scene
    rows = []

    for scene in tqdm(scenes, desc="scenes", unit="scene"):
        with gcov.open_gcov(scene["path"]) as f:
            ident = gcov.read_identification(f)
            # Trust the file over the file name.
            for key in ("track", "frame"):
                if ident[key] != scene[key]:
                    raise ValueError(f"{scene['path'].name}: {key} in file != file name")
            date = ident["start_time"][:10]

            grid = gcov.read_grid(f, freq)
            missing = set(pols) - set(grid["polarizations"])
            if missing:
                raise KeyError(f"{scene['path'].name}: polarizations not in file: {missing}")
            minfo = gcov.mask_info(f, freq) if grid["has_mask"] else None
            if grid["epsg"] not in by_epsg:
                by_epsg[grid["epsg"]] = fields.to_crs(epsg=grid["epsg"])

            for _, field in by_epsg[grid["epsg"]].iterrows():
                res = gcov.extract_geometry(f, grid, field.geometry, minfo, pols)
                for pol in pols:
                    rows.append(
                        {
                            "field_id": field["id"],
                            "date": date,
                            "pol": pol,
                            **res["stats"][pol],
                            "inside_px": res["inside_px"],
                            "outside_swath_px": res["outside_swath_px"],
                            "invalid_px": res["invalid_px"],
                            "clipped": res["clipped"],
                        }
                    )

    df = pd.DataFrame(rows).reindex(columns=SERIES_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["field_id", "pol", "date"]).reset_index(drop=True)


def cache_paths(out_dir, n_fields: int, n_scenes: int) -> tuple[Path, Path]:
    stem = Path(out_dir) / f"series_{n_fields}fields_{n_scenes}scenes"
    return stem.with_suffix(".parquet"), stem.with_suffix(".csv")


def load_or_build(scenes, fields_geojson, out_dir, force=False, **kw) -> tuple[pd.DataFrame, bool]:
    """Return (series, from_cache). Rebuilds if --force or the cache does not match the inputs.

    The cache name only encodes counts, so we also check that the cached field ids and dates
    are exactly the current ones; otherwise editing a polygon list of the same length would
    silently reuse stale numbers.
    """
    fields = load_fields(fields_geojson)
    parquet, csv = cache_paths(out_dir, len(fields), len(scenes))
    if parquet.exists() and not force:
        df = pd.read_parquet(parquet)
        same_fields = set(df["field_id"]) == set(fields["id"])
        same_dates = set(df["date"].dt.date.astype(str)) == {s["date"] for s in scenes}
        if same_fields and same_dates:
            return df, True
        print(f"cache {parquet.name} does not match current fields/scenes; rebuilding")

    df = extract_series(scenes, fields_geojson, **kw)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False)
    return df, False
