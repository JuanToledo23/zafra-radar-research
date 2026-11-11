"""Sentinel-1 GRD (C-band) field-mean series from Earth Engine (exercise 9 follow-up).

    from zafra import optical, sentinel1 as S1
    optical.init_ee()
    df = S1.field_series(geom_lonlat, "2025-11-01", "2026-06-01")   # VH and VV, both passes

Same pixel rules as script 09's p36 check (which pulled VH only, 2024-25):

* COPERNICUS/S1_GRD stores sigma-0 in dB; each pixel is converted to linear power, the
  field mean is taken in linear power (unweighted: pixel centre inside the polygon, on the
  image's own 10 m grid) and converted back to dB, as for NISAR (gcov.field_stats);
* two slices of the same pass and track can both cover the field on one date: the one with
  the most pixels is kept; dates where the field is only partly covered (< 90 % of the
  track's usual pixel count) are dropped.

Tracks are kept apart (`rel_orbit`): over Morelos the descending pass has two tracks (143
and 41) with different incidence angles, so they are separate series, like NISAR orbits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # Earth Engine is the `ee` extra; clean() works without it
    import ee
except ImportError:
    ee = None

COLLECTION = "COPERNICUS/S1_GRD"
POLS = ("VH", "VV")
PASSES = {"ASCENDING": "asc", "DESCENDING": "desc"}


def collection(region: ee.Geometry, start: str, end: str, orbit_pass: str) -> ee.ImageCollection:
    """IW dual-pol (VV + VH) GRD scenes over `region`, start .. end (exclusive), one pass."""
    return (
        ee.ImageCollection(COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )


def to_linear(img: ee.Image) -> ee.Image:
    """dB -> linear power for the VH and VV bands, renamed vh / vv."""
    return ee.Image(10).pow(img.select(list(POLS)).divide(10)).rename([p.lower() for p in POLS])


def _stats(region: ee.Geometry):
    def stats(img):
        s = to_linear(img).reduceRegion(
            reducer=ee.Reducer.mean()
            .unweighted()
            .combine(ee.Reducer.count().unweighted(), sharedInputs=True),
            geometry=region,
            crs=img.select("VH").projection(),
            scale=10,
            maxPixels=1e8,
        )
        return ee.Feature(
            None,
            {
                "image_id": img.get("system:index"),
                "time_ms": img.get("system:time_start"),
                "rel_orbit": img.get("relativeOrbitNumber_start"),
                "platform": img.get("platform_number"),
                "vh_linear": s.get("vh_mean"),
                "vv_linear": s.get("vv_mean"),
                "valid_px": s.get("vh_count"),
            },
        )

    return stats


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one slice per (track, date) and drop partly covered dates; add dB columns."""
    if df.empty:
        return df
    df = df.dropna(subset=["vh_linear", "vv_linear"]).copy()
    df["date"] = pd.to_datetime(df["time_ms"], unit="ms").dt.normalize()
    df["rel_orbit"] = df["rel_orbit"].astype(int)
    df = df.sort_values(["rel_orbit", "date", "valid_px"], ascending=[True, True, False])
    df = df.drop_duplicates(["rel_orbit", "date"])
    usual = df.groupby("rel_orbit")["valid_px"].transform("max")
    df = df[df["valid_px"] >= 0.9 * usual]
    for p in ("vh", "vv"):
        df[f"{p}_db"] = 10 * np.log10(df[f"{p}_linear"])
    return df


def field_series(geom, start: str, end: str) -> pd.DataFrame:
    """VH and VV field means per acquisition, both passes, every track (EE must be initialised).

    `geom` is a shapely polygon in lon/lat. Columns: date, orbit, rel_orbit, platform,
    vh_db, vv_db, vh_linear, vv_linear, valid_px, image_id.
    """
    if ee is None:
        raise ImportError('field_series needs Earth Engine: pip install -e ".[ee]"')
    region = ee.Geometry(geom.__geo_interface__)
    frames = []
    for orbit_pass, orbit in PASSES.items():
        fc = ee.FeatureCollection(collection(region, start, end, orbit_pass).map(_stats(region)))
        df = pd.DataFrame([f["properties"] for f in fc.getInfo()["features"]])
        frames.append(clean(df).assign(orbit=orbit) if not df.empty else df)
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    cols = ["date", "orbit", "rel_orbit", "platform", "vh_db", "vv_db", "vh_linear", "vv_linear",
            "valid_px", "image_id"]  # fmt: skip
    return df[cols].sort_values(["orbit", "rel_orbit", "date"]).reset_index(drop=True)
