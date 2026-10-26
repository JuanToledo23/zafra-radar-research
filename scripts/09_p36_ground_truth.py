"""Ground-truth check for p36 (La Marometa, Tlaquiltenango) and a harvest-front test.

    python scripts/09_p36_ground_truth.py [--max-field-cloud 10] [--from-cache] [--no-write]

The grower of p36 reports the field was burned and cut the same day, 2025-03-16 (zafra
2024/25). NISAR starts 2025-11-23, so that date is checked with other sensors:

  1. Sentinel-1 C-band VH (Earth Engine COPERNICUS/S1_GRD, 2024-10-01..2025-06-30):
     descending pass (as asked) and, as a secondary series, the ascending pass. The largest
     consecutive drop and the exercise-4 rule / ratio-test detectors (src/zafra/detect/),
     run unchanged on the S1 series. Descending has no scene between 2025-02-07 and
     2025-04-02 over p36, so it can only bracket the cut to ~54 days; ascending (track 78)
     has 2025-03-11 and 2025-03-23.
  2. Sentinel-2 NDVI around 2025-03-16 (exercise 6: clearest scene <= 20 d before / after,
     confirmed / refuted / inconclusive) and the NDVI step date.
  3. VIIRS fires (FIRMS) 2025-03-05..03-25 inside the polygon and within 500 m.
  4. Zafra 2025/26 on p36: NISAR HV asc/desc and HH coherence, plus the NDVI window
     (exercise 7 rule). The NDVI window is written to p36's harvest_window_start/end in
     data/fields/fields_v2.geojson (skip with --no-write), as for the other 37 fields.
  5. Harvest front: the mill cuts by region, so cane fields within 700 m of p36 should have
     been cut in the same weeks. There are no surveyed polygons there, so candidates are
     SNIC segments of a monthly NDVI stack (Earth Engine), kept if they pass exercise 7's
     cane tests (NDVI plateau, <= 1 collapse per zafra, max >= 0.75). For each: 2025 NDVI
     window and S1 largest VH drop. Writes data/fields/p36_neighbours.geojson.

Outputs in outputs/front/: p36_2025_series.png, p36_front_2025.png and CSV caches of every
Earth Engine pull (reused with --from-cache).
"""

import argparse
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import io

import ee
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zafra import fires as F
from zafra import multisignal as ms
from zafra import optical as O
from zafra import timeseries as ts
from zafra.detect import GcovPixelProvider, detect_ratio, detect_rule
from zafra.detect import ratio_test as RT
from zafra.detect import rule as RULE
from zafra.paths import FIELDS_DIR, FIRMS_DIR, NISAR_DIR, OUTPUTS_DIR, ROOT


def load_script(name: str):
    """Import scripts/<name>.py (file names start with a digit, so no plain import)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X6 = load_script("06_optical_validation")  # ndvi_step, verdict (exercise 6)
X7 = load_script("07_label_fields")  # clear_scenes, label_field, harvest_window (exercise 7)

OUT_DIR = OUTPUTS_DIR / "front"
FIELDS = FIELDS_DIR / "fields_v2.geojson"
NEIGHBOURS_GEOJSON = FIELDS_DIR / "p36_neighbours.geojson"
FIELD_ID = "p36"
UTM = F.METRIC_CRS

GT = pd.Timestamp("2025-03-16")  # grower: burned and cut that day
TOL_DAYS = 7
S1_START, S1_END = "2024-10-01", "2025-07-01"  # end exclusive -> through 2025-06-30
FIRMS_START, FIRMS_END = "2025-03-05", "2025-03-26"  # local dates, end exclusive
FIRMS_BUFFER_M = 500
SEASON_2425 = ("2024-10-01", "2025-07-01")  # NDVI step search for the 2024/25 cut
# S1 "largest drop" only counts drops whose later scene is in the Nov-May zafra: from June
# the first rains swing C-band VH by 1-2 dB (wet soil, then drying), as much as a cut.
ZAFRA_2425 = ("2024-11-01", "2025-06-01")
OPT_DAYS = 20  # exercise 6: clearest scene within 20 d before / after

# NISAR 2025/26, same folders as scripts/08_multisignal.py
DESC_DIR = NISAR_DIR / "GCOV_DESCENDING_t113f79"
GUNW_DIR = NISAR_DIR / "GUNW_ASCENDING_t19f11"

# Harvest front
NEIGH_RADIUS_M = 700
SEG_MARGIN_M = 300  # segment a slightly larger area so edge segments are not truncated
SNIC_SIZE_PX = 15  # seed spacing, 150 m: a ~2-5 ha segment, the size of a typical field here
SNIC_COMPACTNESS = 0.5
MIN_SEG_HA, MAX_SEG_HA = 1.0, 40.0
MIN_MAIN_PART = 0.8  # main polygon must hold >= 80 % of a segment's area
MAX_OVERLAP_P36 = 0.1  # drop segments with > 10 % of their area inside p36
MIN_NEIGH_SPACING_M = 150  # two chosen segments' centroids at least this far apart
MAX_NEIGHBOURS = 10
S1_COLLECTION = "COPERNICUS/S1_GRD"


# --------------------------------------------------------------------------- Sentinel-1


def s1_collection(region: ee.Geometry, orbit_pass: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(region)
        .filterDate(S1_START, S1_END)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )


def s1_linear(img: ee.Image) -> ee.Image:
    """COPERNICUS/S1_GRD stores sigma-0 in dB; the field mean must be taken in linear power
    (as for NISAR, see ratio_test.py), so convert per pixel first."""
    return ee.Image(10).pow(img.select("VH").divide(10)).rename("vh")


def s1_vh_series(geom, orbit_pass: str) -> pd.DataFrame:
    """Field-mean VH (linear and dB) per acquisition date, pixel centre inside the polygon.

    Two slices of the same pass can both cover the field; keep the one with most pixels,
    and drop dates where the field is only partly covered (< 90 % of the usual count).
    """
    region = ee.Geometry(geom.__geo_interface__)

    def stats(img):
        proj = img.select("VH").projection()
        s = s1_linear(img).reduceRegion(
            reducer=ee.Reducer.mean()
            .unweighted()
            .combine(ee.Reducer.count().unweighted(), sharedInputs=True),
            geometry=region,
            crs=proj,
            scale=10,
            maxPixels=1e8,
        )
        return ee.Feature(
            None,
            {
                "image_id": img.get("system:index"),
                "time_ms": img.get("system:time_start"),
                "rel_orbit": img.get("relativeOrbitNumber_start"),
                "mean_linear": s.get("vh_mean"),
                "valid_px": s.get("vh_count"),
            },
        )

    fc = ee.FeatureCollection(s1_collection(region, orbit_pass).map(stats))
    df = pd.DataFrame([f["properties"] for f in fc.getInfo()["features"]])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["time_ms"], unit="ms").dt.normalize()
    df = df.dropna(subset=["mean_linear"])
    df = df.sort_values(["date", "valid_px"], ascending=[True, False]).drop_duplicates("date")
    df = df[df["valid_px"] >= 0.9 * df["valid_px"].max()]
    df["mean_db"] = 10 * np.log10(df["mean_linear"])
    df["orbit"] = {"DESCENDING": "desc", "ASCENDING": "asc"}[orbit_pass]
    cols = ["date", "orbit", "rel_orbit", "mean_db", "mean_linear", "valid_px", "image_id"]
    return df[cols].reset_index(drop=True)


def s1_pixels(image_id: str, geom) -> np.ndarray:
    """Linear VH of every pixel of the field in one S1 image (for the ratio test's ENL)."""
    region = ee.Geometry(geom.__geo_interface__)
    img = ee.Image(f"{S1_COLLECTION}/{image_id}")
    vals = (
        s1_linear(img)
        .reduceRegion(
            ee.Reducer.toList(),
            region,
            crs=img.select("VH").projection(),
            scale=10,
            maxPixels=1e8,
        )
        .get("vh")
        .getInfo()
    )
    return np.asarray(vals, dtype="float64")


def largest_drop(s: pd.DataFrame, col: str = "mean_db", after_in=None):
    """(date_before, date_after, step dB) of the largest consecutive decrease.

    after_in = (start, end) limits it to drops whose later scene falls in [start, end).
    """
    s = s.sort_values("date").reset_index(drop=True)
    d = s[col].diff()
    if after_in is not None:
        d = d.where(s["date"].between(*map(pd.Timestamp, after_in), inclusive="left"))
    if d.notna().sum() == 0:
        return pd.NaT, pd.NaT, np.nan
    i = int(d.idxmin())
    return s.loc[i - 1, "date"], s.loc[i, "date"], float(d[i])


def run_detectors(series: pd.DataFrame, fid: str, pixels) -> tuple[pd.Series, pd.Series]:
    """Exercise-4 rule and ratio test on one field's series.

    The detectors select rows with pol == "HVHV" (NISAR's cross-pol). For Sentinel-1 the
    cross-pol is VH; the physics they rely on (volume scattering lost at the cut) is the
    same, so the S1 VH series is passed in under that label. Thresholds are unchanged
    (they were set for L-band; C-band saturates earlier on cane, so its drop can be smaller).
    """
    df = series.assign(field_id=fid, pol=RT.POL)
    rule = detect_rule(df).iloc[0]
    det = detect_ratio(df, lambda _fid, date: pixels(date)).iloc[0]
    return rule, det


# --------------------------------------------------------------------------- agreement


def window_offset(a, b, t=GT) -> float:
    """Days from t to the window [a, b] (0 if t is inside it)."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if a <= t <= b:
        return 0.0
    return float(min(abs((t - a).days), abs((t - b).days)))


def agrees(a, b) -> str:
    off = window_offset(a, b)
    if np.isnan(off):
        return "n/a"
    width = (pd.Timestamp(b) - pd.Timestamp(a)).days
    txt = "yes" if off <= TOL_DAYS else f"no ({off:.0f} d off)"
    if width > 30:
        txt += f", {width}-d window: weak"
    return txt


def fmt_win(a, b) -> str:
    return "—" if pd.isna(a) or pd.isna(b) else f"{a:%Y-%m-%d} → {b:%Y-%m-%d}"


# --------------------------------------------------------------------------- NISAR 2025/26


def nisar_p36(p36_path) -> dict:
    """HV asc/desc series, rule/ratio detection, largest drops, and coherence for p36."""
    out = {}
    for orbit, d in (("asc", NISAR_DIR), ("desc", DESC_DIR)):
        series = ts.extract_series(ts.list_scenes(d), p36_path, pols=("HVHV", "HHHH"))
        hv = series[series["pol"] == "HVHV"].sort_values("date")
        provider = GcovPixelProvider(d, p36_path)
        out[orbit] = {
            "hv": hv,
            "drop": largest_drop(hv),
            "rule": detect_rule(series).iloc[0],
            "ratio": detect_ratio(series, provider).iloc[0],
        }
    coh = ms.extract_coherence_series(GUNW_DIR, ts.load_fields(p36_path))
    out["coh"] = coh.sort_values("ref_date").reset_index(drop=True)
    return out


def write_window(path, fid: str, start, end) -> None:
    """Set harvest_window_start/end of one feature in place, leaving every other line (and
    its formatting) untouched."""
    lines = path.read_text().split("\n")
    key = f'"id": "{fid}"'
    hits = [i for i, line in enumerate(lines) if key in line]
    if len(hits) != 1:
        raise ValueError(f"expected one line with {key} in {path}, found {len(hits)}")
    i = hits[0]
    for name, val in (("harvest_window_start", start), ("harvest_window_end", end)):
        new = "null" if pd.isna(val) else f'"{val:%Y-%m-%d}"'
        lines[i], n = re.subn(rf'"{name}": (null|"[0-9-]+")', f'"{name}": {new}', lines[i])
        if n != 1:
            raise ValueError(f"{name} not found on the {fid} line")
    text = "\n".join(lines)
    n_features = len(json.loads(text)["features"])  # still valid JSON before writing
    path.write_text(text)
    print(f"wrote {fid} harvest window {fmt_win(start, end)} to {path.relative_to(ROOT)}")
    print(f"  ({n_features} features, file re-parsed OK)")


# --------------------------------------------------------------------------- harvest front


def month_median(coll: ee.ImageCollection, fill: ee.Image) -> ee.Image:
    """Median NDVI of one month; cloudy pixels, or a month with no scene, take `fill`."""
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.median().unmask(fill), fill))


def segment_neighbourhood(p36_geom) -> gpd.GeoDataFrame:
    """SNIC segments of a monthly NDVI stack (Oct 2024 - Jun 2025) around p36.

    Fields differ from their neighbours mostly in *when* they were cut and how fast they
    regrew, so a month-by-month NDVI stack separates them better than a single image.
    """
    field_m = gpd.GeoSeries([p36_geom], crs="EPSG:4326").to_crs(UTM).iloc[0]
    area = gpd.GeoSeries([field_m.buffer(NEIGH_RADIUS_M + SEG_MARGIN_M)], crs=UTM)
    region = ee.Geometry(area.to_crs("EPSG:4326").iloc[0].__geo_interface__)

    def ndvi(img):
        clear = img.select("SCL").remap(list(O.SCL_KEEP), [1] * len(O.SCL_KEEP), 0).eq(1)
        return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(clear)

    coll = (
        ee.ImageCollection(O.S2_COLLECTION)
        .filterBounds(region)
        .filterDate(*SEASON_2425)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", X7.MAX_TILE_CLOUD))
        .map(ndvi)
    )
    fill = coll.median()
    months = pd.date_range(SEASON_2425[0], SEASON_2425[1], freq="MS")[:-1]
    stack = ee.Image.cat(
        [
            month_median(
                coll.filterDate(f"{m:%Y-%m-%d}", f"{m + pd.offsets.MonthBegin():%Y-%m-%d}"), fill
            ).rename(f"m{m:%Y%m}")
            for m in months
        ]
    )
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=stack,
        compactness=SNIC_COMPACTNESS,
        connectivity=8,
        neighborhoodSize=4 * SNIC_SIZE_PX,
        seeds=ee.Algorithms.Image.Segmentation.seedGrid(SNIC_SIZE_PX),
    )
    # SNIC depends on pixel size, so fix it to the 10 m UTM grid instead of the map scale.
    clusters = snic.select("clusters").reproject(crs=UTM, scale=10).clip(region)
    vec = clusters.reduceToVectors(
        geometry=region,
        crs=UTM,
        scale=10,
        geometryType="polygon",
        eightConnected=False,
        labelProperty="cluster",
        maxPixels=1e9,
    ).map(lambda f: f.transform("EPSG:4326", 1))
    gdf = gpd.GeoDataFrame.from_features(vec.getInfo()["features"], crs="EPSG:4326")
    # One cluster id can come back as several pieces; keep it as one multi-part segment.
    gdf = gdf.dissolve("cluster").reset_index()

    # Pixel polygons traced with 4-connectivity split at diagonal joins into many slivers.
    # Close 5 m gaps (half a pixel), keep the main part, and drop segments whose main part
    # is < MIN_MAIN_PART of the whole (no single field shape). This also gives Earth Engine
    # simple polygons; it rejects some multi-part ones when clipping S1 scenes.
    seg_m = gdf.to_crs(UTM)
    total = seg_m.area
    main = (
        seg_m.buffer(5)
        .buffer(-5)
        .apply(lambda g: max(getattr(g, "geoms", [g]), key=lambda x: x.area))
    )
    main = main.simplify(5).buffer(0)
    gdf = gdf[(main.area / total >= MIN_MAIN_PART).to_numpy()].copy()
    seg_m = gpd.GeoSeries(main[main.area / total >= MIN_MAIN_PART], crs=UTM)
    gdf["geometry"] = seg_m.to_crs("EPSG:4326").to_numpy()
    gdf["area_ha"] = seg_m.area / 1e4
    gdf["overlap_p36"] = seg_m.intersection(field_m).area / seg_m.area
    gdf["dist_m"] = seg_m.centroid.distance(field_m)
    keep = (
        gdf["area_ha"].between(MIN_SEG_HA, MAX_SEG_HA)
        & (gdf["overlap_p36"] <= MAX_OVERLAP_P36)
        & (gdf["dist_m"] <= NEIGH_RADIUS_M)
    )
    gdf = gdf[keep].copy()
    gdf["seg_id"] = [f"s{c}" for c in gdf["cluster"]]
    return gdf.sort_values("dist_m").reset_index(drop=True)


def neighbour_series(segs: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NDVI (exercise-7 period) and S1 VH (descending + ascending) for every segment."""

    def one(row):
        print(f"{row.seg_id}: querying S2 + S1 ...", file=sys.stderr)
        nd = O.s2_ndvi_series(row.geometry, X7.START, X7.END, max_cloud=X7.MAX_TILE_CLOUD)
        s1 = pd.concat(
            [s1_vh_series(row.geometry, p) for p in ("DESCENDING", "ASCENDING")],
            ignore_index=True,
        )
        return nd.assign(seg_id=row.seg_id), s1.assign(seg_id=row.seg_id)

    with ThreadPoolExecutor(max_workers=6) as ex:
        parts = list(ex.map(one, segs.itertuples()))
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


def pick_neighbours(segs, ndvi, s1, max_field_cloud) -> pd.DataFrame:
    """Cane segments (exercise-7 tests + max >= 0.75), nearest first, spread out."""
    rows = []
    for _, sg in segs.iterrows():
        clear = X7.clear_scenes(ndvi[ndvi["seg_id"] == sg["seg_id"]], max_field_cloud)
        lab = X7.label_field(clear)
        ws, we = X7.harvest_window(clear, "2024-25")
        r = {"seg_id": sg["seg_id"], **lab, "n_clear": len(clear), "s2_start": ws, "s2_end": we}
        for orbit in ("desc", "asc"):
            o = s1[(s1["seg_id"] == sg["seg_id"]) & (s1["orbit"] == orbit)]
            a, b, step = largest_drop(o, after_in=ZAFRA_2425)
            r |= {f"s1_{orbit}_before": a, f"s1_{orbit}_after": b, f"s1_{orbit}_step_db": step}
            r[f"vh_{orbit}_range_db"] = o["mean_db"].max() - o["mean_db"].min()
        rows.append(r)
    res = segs.merge(pd.DataFrame(rows), on="seg_id")
    res["cane"] = (res["label"] == "cana") & (res["ndvi_max"] >= X7.PEAK)

    chosen = []
    cents = res.to_crs(UTM).centroid
    for i in res[res["cane"]].sort_values("dist_m").index:
        if all(cents[i].distance(cents[j]) >= MIN_NEIGH_SPACING_M for j in chosen):
            chosen.append(i)
        if len(chosen) == MAX_NEIGHBOURS:
            break
    res["chosen"] = res.index.isin(chosen)
    # Readable ids for the chosen ones (n01 = nearest); SNIC cluster ids are arbitrary ints.
    res["snic_id"] = res["seg_id"]
    for k, i in enumerate(chosen, start=1):
        res.loc[i, "seg_id"] = f"n{k:02d}"
    return res


# --------------------------------------------------------------------------- plots


def plot_series(s1, ndvi, fires_near, max_field_cloud, out) -> None:
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    style = {
        "desc": ("S1 VH descending (t143)", "tab:orange"),
        "asc": ("S1 VH ascending (t78)", "tab:blue"),
    }
    for orbit, (label, color) in style.items():
        o = s1[s1["orbit"] == orbit].sort_values("date")
        ax.plot(o["date"], o["mean_db"], marker="o", ms=4, color=color, label=label)
    ax.set_ylabel("VH sigma-0 (dB), field mean")

    s = ndvi[(ndvi["date"] >= S1_START) & (ndvi["date"] < S1_END)]
    clear = (s["cloud_pct"] <= max_field_cloud) & s["ndvi_mean"].notna()
    axn.plot(
        s.loc[clear, "date"],
        s.loc[clear, "ndvi_mean"],
        "-o",
        ms=4,
        color="tab:green",
        label=f"S2 NDVI, clear (≤ {max_field_cloud:g} % masked)",
    )
    axn.plot(
        s.loc[~clear, "date"],
        s.loc[~clear, "ndvi_mean"],
        "o",
        ms=4,
        mfc="none",
        mec="0.6",
        label="S2 NDVI, cloudy",
    )
    axn.axhline(X6.HIGH, color="0.5", ls="--", lw=0.8)
    axn.axhline(X6.LOW, color="0.5", ls=":", lw=0.8)
    axn.set_ylim(-0.05, 1)
    axn.set_ylabel("NDVI, field mean")

    for a in (ax, axn):
        a.axvspan(
            GT - pd.Timedelta(days=TOL_DAYS),
            GT + pd.Timedelta(days=TOL_DAYS),
            color="k",
            alpha=0.07,
            lw=0,
        )
        a.axvline(GT, color="k", lw=1.5)
        for _, f in fires_near.iterrows():
            t = f["acq_utc"].tz_convert("UTC").tz_localize(None)
            inside = f["dist_edge_m"] == 0
            a.axvline(t, color="tab:red", lw=1.2 if inside else 0.8, alpha=0.9 if inside else 0.4)
        a.grid(alpha=0.3)
    handles = [
        Line2D(
            [], [], color="k", lw=1.5, label=f"ground truth {GT:%Y-%m-%d} (± {TOL_DAYS} d shaded)"
        ),
        Line2D(
            [], [], color="tab:red", lw=1.2, alpha=0.9, label="VIIRS fire, pixel centre inside p36"
        ),
        Line2D(
            [], [], color="tab:red", lw=0.8, alpha=0.4, label=f"VIIRS fire ≤ {FIRMS_BUFFER_M} m"
        ),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles, fontsize=8, loc="lower left")
    axn.legend(fontsize=8, loc="lower left")
    axn.xaxis.set_major_locator(mdates.MonthLocator())
    axn.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axn.set_xlim(pd.Timestamp(S1_START), pd.Timestamp(S1_END))
    axn.set_xlabel("date (UTC)")
    fig.suptitle(f"{FIELD_ID} La Marometa, zafra 2024/25: Sentinel-1 VH, Sentinel-2 NDVI, VIIRS")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def rgb_background(bounds):
    """True-colour S2 median of 2025-03-20..04-10 (just after the cut), or None."""
    try:
        region = ee.Geometry.Rectangle(list(bounds), "EPSG:4326", False)
        img = (
            ee.ImageCollection(O.S2_COLLECTION)
            .filterBounds(region)
            .filterDate("2025-03-20", "2025-04-10")
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .visualize(bands=["B4", "B3", "B2"], min=0, max=3000, gamma=1.3)
        )
        url = img.getThumbURL(
            {"region": region, "dimensions": 800, "format": "png", "crs": "EPSG:4326"}
        )
        return plt.imread(io.BytesIO(requests.get(url, timeout=60).content), format="png")
    except Exception as e:  # noqa: BLE001 (background only; the map still renders)
        print(f"(no RGB background: {e})")
        return None


def plot_front(p36, res, out) -> None:
    sel = res[res["chosen"]].copy()
    sel["mid"] = sel["s2_start"] + (sel["s2_end"] - sel["s2_start"]) / 2
    pad = 0.002
    xmin, ymin, xmax, ymax = pd.concat([sel.geometry, p36.geometry]).total_bounds
    bounds = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

    fig, ax = plt.subplots(figsize=(10, 8))
    bg = rgb_background(bounds)
    if bg is not None:
        ax.imshow(bg, extent=(bounds[0], bounds[2], bounds[1], bounds[3]), alpha=0.6)
    rest = res[~res["chosen"]]
    rest.boundary.plot(ax=ax, color="0.4", lw=0.5, ls=":")

    t = sel["mid"].dropna()
    lo = min(t.min(), GT - pd.Timedelta(days=30)) if len(t) else GT - pd.Timedelta(days=30)
    hi = max(t.max(), GT + pd.Timedelta(days=30)) if len(t) else GT + pd.Timedelta(days=30)
    norm = matplotlib.colors.Normalize(mdates.date2num(lo), mdates.date2num(hi))
    cmap = matplotlib.colormaps["viridis"]
    for _, r in sel.iterrows():
        color = "0.8" if pd.isna(r["mid"]) else cmap(norm(mdates.date2num(r["mid"])))
        within = window_offset(r["s2_start"], r["s2_end"]) <= TOL_DAYS
        gpd.GeoSeries([r.geometry]).plot(
            ax=ax, color=color, edgecolor="k", lw=1.6 if within else 0.6, alpha=0.85
        )
        c = r.geometry.representative_point()
        ax.annotate(
            r["seg_id"],
            (c.x, c.y),
            fontsize=7,
            ha="center",
            color="w",
            path_effects=[pe.withStroke(linewidth=2, foreground="k")],
        )
    gt_color = cmap(norm(mdates.date2num(GT)))
    p36.plot(ax=ax, color=gt_color, edgecolor="tab:red", lw=2.2, alpha=0.85)
    c = p36.geometry.iloc[0].representative_point()
    ax.annotate(
        f"{FIELD_ID}\n{GT:%m-%d} (grower)",
        (c.x, c.y),
        ha="center",
        fontsize=8,
        color="w",
        path_effects=[pe.withStroke(linewidth=2, foreground="k")],
    )
    gpd.GeoSeries([p36.to_crs(UTM).geometry.iloc[0].buffer(NEIGH_RADIUS_M)], crs=UTM).to_crs(
        "EPSG:4326"
    ).boundary.plot(ax=ax, color="w", ls="--", lw=1)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("2025 harvest: midpoint of the S2 NDVI window")
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    cb.ax.axhline(mdates.date2num(GT), color="tab:red", lw=2)
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="tab:red", lw=2, label=f"{FIELD_ID} (ground truth)"),
            Patch(
                facecolor="none",
                edgecolor="k",
                lw=1.6,
                label=f"neighbour, window within ± {TOL_DAYS} d",
            ),
            Patch(facecolor="none", edgecolor="k", lw=0.6, label="neighbour, window further off"),
            Line2D([], [], color="0.4", ls=":", label="other segments (not cane / not chosen)"),
            Line2D([], [], color="w", ls="--", label=f"{NEIGH_RADIUS_M} m from {FIELD_ID}"),
        ],
        fontsize=8,
        loc="lower left",
        facecolor="0.85",
    )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1 / np.cos(np.deg2rad((bounds[1] + bounds[3]) / 2)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        "Harvest front 2025 around La Marometa: S2 harvest date of SNIC cane segments\n"
        "(background: S2 true colour, median 2025-03-20..04-10)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def cached_csv(path, build, from_cache, dates=("date",)):
    if from_cache and path.exists():
        return pd.read_csv(path, parse_dates=list(dates))
    df = build()
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--max-field-cloud", type=float, default=10)
    ap.add_argument("--from-cache", action="store_true", help="reuse outputs/front/*.csv")
    ap.add_argument("--no-write", action="store_true", help="do not touch fields_v2.geojson")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    O.init_ee()
    could_not = []

    fields = ts.load_fields(FIELDS)
    p36 = fields[fields["id"] == FIELD_ID].reset_index(drop=True)
    geom = p36.geometry.iloc[0]
    p36_path = OUT_DIR / "p36.geojson"  # single-field file for the NISAR readers
    p36[["id", "geometry"]].to_file(p36_path, driver="GeoJSON")
    print(
        f"{FIELD_ID}: {p36['area_ha'].iloc[0]:.2f} ha, ground truth {GT:%Y-%m-%d} "
        f"(burned and cut the same day, per grower; zafra 2024/25)"
    )

    # ---- (a) Sentinel-1 VH -------------------------------------------------------------
    s1 = cached_csv(
        OUT_DIR / "p36_s1_vh.csv",
        lambda: pd.concat(
            [s1_vh_series(geom, p) for p in ("DESCENDING", "ASCENDING")], ignore_index=True
        ),
        args.from_cache,
    )
    s1_rows = {}
    print(f"\n## (a) Sentinel-1 VH, {S1_START}..2025-06-30, field mean of linear sigma-0")
    for orbit in ("desc", "asc"):
        o = s1[s1["orbit"] == orbit].sort_values("date").reset_index(drop=True)
        ids = dict(zip(o["date"], o["image_id"], strict=True))
        a, b, step = largest_drop(o, after_in=ZAFRA_2425)
        rule, det = run_detectors(
            o, FIELD_ID, lambda d, ids=ids: s1_pixels(ids[pd.Timestamp(d)], geom)
        )
        s1_rows[orbit] = (a, b, step, rule, det)
        gaps = o["date"].diff().dt.days
        k = int(gaps.idxmax())
        print(
            f"\n### {orbit} (rel. orbit {o['rel_orbit'].iloc[0]}): {len(o)} dates, "
            f"{int(o['valid_px'].median())} px; longest gap {int(gaps[k])} d "
            f"({o.loc[k - 1, 'date']:%Y-%m-%d} → {o.loc[k, 'date']:%Y-%m-%d})"
        )
        print(
            "dates/dB: "
            + ", ".join(f"{d:%m-%d} {v:.1f}" for d, v in zip(o["date"], o["mean_db"], strict=True))
        )
        print(
            f"- largest consecutive drop (later scene in "
            f"{ZAFRA_2425[0]}..05-31): {fmt_win(a, b)} ({step:+.2f} dB)"
        )
        print(
            f"- rule (baseline {rule['baseline_db']:.2f} dB, "
            f"−{RULE.DROP_DB:g}/−{RULE.SUSTAIN_DB:g} dB): "
            + (fmt_win(rule["date_before"], rule["date_detected"]) if rule["detected"] else "none")
        )
        print(
            f"- ratio test (ENL {det['enl']:.2f}, baseline {det['baseline_db']:.2f} ± "
            f"{det['baseline_se_db']:.2f} dB on {det['baseline_dates']}): "
            + (
                f"{fmt_win(det['date_before'], det['date_detected'])}, drop "
                f"{det['drop_db']:+.2f} dB, p_prev {det['p_prev']:.1e}, {det['confidence']}"
                if det["detected"]
                else "none"
            )
        )
        if gaps.max() > 30 and o.loc[k - 1, "date"] < GT < o.loc[k, "date"]:
            could_not.append(
                f"S1 {orbit}: no scene {o.loc[k - 1, 'date']:%Y-%m-%d}→"
                f"{o.loc[k, 'date']:%Y-%m-%d}; cannot date the cut closer than that"
            )

    # ---- (b) Sentinel-2 NDVI -------------------------------------------------------------
    ndvi = cached_csv(
        OUT_DIR / "p36_ndvi.csv",
        lambda: O.s2_ndvi_series(geom, X7.START, X7.END, max_cloud=X7.MAX_TILE_CLOUD),
        args.from_cache,
    )
    # The cut date itself is left out of both sides: S2 looks at ~11:00 local, VIIRS saw the
    # fire at 13:43, so the 03-16 scene may show the field before, during or after the burn.
    before, after = O.clearest_before_after(
        ndvi,
        GT - pd.Timedelta(days=1),
        GT + pd.Timedelta(days=1),
        days=OPT_DAYS - 1,
        max_field_cloud=args.max_field_cloud,
    )
    v, why = X6.verdict(before, after)
    s2425 = ndvi[(ndvi["date"] >= SEASON_2425[0]) & (ndvi["date"] < SEASON_2425[1])]
    step_a, step_b = X6.ndvi_step(s2425, args.max_field_cloud)
    clear = X7.clear_scenes(ndvi, args.max_field_cloud)
    w7_a, w7_b = X7.harvest_window(clear, "2024-25")
    on_gt = clear[clear["date"] == GT]
    near = clear[
        (clear["date"] >= GT - pd.Timedelta(days=45))
        & (clear["date"] <= GT + pd.Timedelta(days=45))
    ]
    print(f"\n## (b) Sentinel-2 NDVI (clear = ≤ {args.max_field_cloud:g} % of field masked)")
    print(
        "clear scenes ±45 d: "
        + ", ".join(
            f"{d:%m-%d} {x:.2f}" for d, x in zip(near["date"], near["ndvi_mean"], strict=True)
        )
    )
    for name, pk in (("before", before), ("after", after)):
        print(
            f"- clearest {name} (≤ {OPT_DAYS} d): "
            + (
                "none"
                if pk is None
                else f"{pk['date']:%Y-%m-%d} NDVI {pk['ndvi_mean']:.2f} "
                f"({pk['cloud_pct']:.0f} % masked)"
            )
        )
    if not on_gt.empty:
        print(
            f"- scene on {GT:%Y-%m-%d} itself (~11:00 local): NDVI {on_gt['ndvi_mean'].iloc[0]:.2f}"
        )
    print(f"- verdict for {GT:%Y-%m-%d} (exercise 6 thresholds): {v} ({why})")
    print(
        f"- NDVI window 2024/25 (exercise 7 rule, ≥ {X7.HARVEST_HIGH} → ≤ {X7.LOW}, the rule "
        f"used for fields_v2 and the neighbours): {fmt_win(w7_a, w7_b)}"
    )
    print(
        f"- NDVI step 2024/25 (exercise 6: last ≥ {X6.HIGH} → first "
        f"≤ {X6.LOW}): {fmt_win(step_a, step_b)}"
    )

    # ---- (c) FIRMS -----------------------------------------------------------------------
    fires = F.load_firms(FIRMS_DIR)
    lo, hi = (pd.Timestamp(d, tz=F.LOCAL_TZ) for d in (FIRMS_START, FIRMS_END))
    win = fires[(fires["acq_local"] >= lo) & (fires["acq_local"] < hi)]
    near_f = F.fires_near_field(win, geom, buffer_m=FIRMS_BUFFER_M)
    season_f = F.fires_near_field(
        fires[
            (fires["acq_local"] >= pd.Timestamp("2024-11-01", tz=F.LOCAL_TZ))
            & (fires["acq_local"] < pd.Timestamp("2025-06-01", tz=F.LOCAL_TZ))
        ],
        geom,
        buffer_m=FIRMS_BUFFER_M,
    )
    print(
        f"\n## (c) VIIRS fires "
        f"{FIRMS_START}..{pd.Timestamp(FIRMS_END) - pd.Timedelta(days=1):%Y-%m-%d} "
        f"(local), within {FIRMS_BUFFER_M} m of {FIELD_ID}"
    )
    if near_f.empty:
        print("(none)")
    else:
        print("| local date | local time | satellite | conf | FRP MW | dist to edge m | inside |")
        print("|---|---|---|---|---|---|---|")
        for _, f in near_f.iterrows():
            print(
                f"| {f['acq_local']:%Y-%m-%d} | {f['acq_local']:%H:%M} | {f['satellite']} | "
                f"{f['confidence']} | {f['frp']:.2f} | {f['dist_edge_m']:.0f} | "
                f"{'YES' if f['dist_edge_m'] == 0 else ''} |"
            )
    print(
        f"whole zafra 2024/25 (Nov-May) within {FIRMS_BUFFER_M} m: {len(season_f)} detections on "
        + (", ".join(sorted({f"{t:%m-%d}" for t in season_f["acq_local"]})) or "no dates")
    )
    inside = near_f[near_f["dist_edge_m"] == 0]

    # ---- (d) summary ---------------------------------------------------------------------
    def firms_win(df):
        if df.empty:
            return pd.NaT, pd.NaT
        d = df["acq_local"].dt.tz_localize(None).dt.normalize()
        return d.min(), d.max()

    table = []
    for orbit, name in (("desc", "S1 VH descending"), ("asc", "S1 VH ascending (secondary)")):
        a, b, step, rule, det = s1_rows[orbit]
        table += [
            (f"{name}: largest drop ({step:+.1f} dB)", a, b),
            (f"{name}: rule (ex. 4)", rule["date_before"], rule["date_detected"]),
            (f"{name}: ratio test (ex. 4)", det["date_before"], det["date_detected"]),
        ]
    table += [
        (f"S2 NDVI step (ex. 6, {X6.HIGH}→{X6.LOW})", step_a, step_b),
        (f"S2 NDVI window (ex. 7, {X7.HARVEST_HIGH}→{X7.LOW})", w7_a, w7_b),
        (
            f"S2 NDVI before/after {GT:%m-%d}: {v}",
            None if before is None else before["date"],
            None if after is None else after["date"],
        ),
        ("FIRMS inside polygon", *firms_win(inside)),
        (f"FIRMS ≤ {FIRMS_BUFFER_M} m", *firms_win(near_f)),
    ]
    print(f"\n## (d) Summary: does each source agree with {GT:%Y-%m-%d} (± {TOL_DAYS} d)?")
    print(
        "A window agrees if it overlaps the ±7-day interval; a "
        "FIRMS 'window' is first→last detection.\n"
    )
    print("| source | window | agrees |")
    print("|---|---|---|")
    for name, a, b in table:
        if name.startswith("S2 NDVI before/after") and v != "confirmed":
            ok = f"no ({v})" if v == "refuted" else v
        else:
            ok = agrees(a, b)
        print(f"| {name} | {fmt_win(a, b)} | {ok} |")

    # ---- 2. zafra 2025/26 on p36 ------------------------------------------------------------
    print("\n## 2. Zafra 2025/26 on p36 (NISAR + NDVI)")
    n = nisar_p36(p36_path)
    for orbit in ("asc", "desc"):
        r = n[orbit]
        a, b, step = r["drop"]
        print(
            f"- NISAR HV {orbit}: largest drop {fmt_win(a, b)} ({step:+.2f} dB); rule "
            + (
                fmt_win(r["rule"]["date_before"], r["rule"]["date_detected"])
                if r["rule"]["detected"]
                else "none"
            )
            + "; ratio test "
            + (
                f"{fmt_win(r['ratio']['date_before'], r['ratio']['date_detected'])} "
                f"({r['ratio']['drop_db']:+.1f} dB, {r['ratio']['confidence']})"
                if r["ratio"]["detected"]
                else "none"
            )
        )
        print(
            "    "
            + ", ".join(
                f"{d:%m-%d} {x:.1f}"
                for d, x in zip(r["hv"]["date"], r["hv"]["mean_db"], strict=True)
            )
        )
    coh = n["coh"]
    print(
        "- HH coherence (12-d asc pairs): "
        + ", ".join(
            f"{a:%m-%d}/{b:%m-%d} {c:.2f}"
            for a, b, c in zip(coh["ref_date"], coh["sec_date"], coh["coh_mean"], strict=True)
        )
    )
    i = coh["coh_mean"].idxmin()
    print(
        f"  lowest pair {coh.loc[i, 'ref_date']:%Y-%m-%d}/{coh.loc[i, 'sec_date']:%Y-%m-%d} "
        f"({coh.loc[i, 'coh_mean']:.2f})"
    )
    ws, we = X7.harvest_window(clear, "2025-26")
    f26 = F.fires_near_field(
        fires[
            (fires["acq_local"] >= pd.Timestamp("2025-11-01", tz=F.LOCAL_TZ))
            & (fires["acq_local"] < pd.Timestamp("2026-06-01", tz=F.LOCAL_TZ))
        ],
        geom,
        buffer_m=FIRMS_BUFFER_M,
    )
    print(
        f"- VIIRS 2025/26 within {FIRMS_BUFFER_M} m: "
        + (
            ", ".join(
                f"{t:%Y-%m-%d %H:%M} ({e:.0f} m)"
                for t, e in zip(f26["acq_local"], f26["dist_edge_m"], strict=True)
            )
            or "none"
        )
    )
    print(
        f"- S2 NDVI window 2025/26 (exercise 7 rule, ≥ "
        f"{X7.HARVEST_HIGH} → ≤ {X7.LOW}): {fmt_win(ws, we)}"
    )
    if pd.notna(ws):
        march = "yes" if (ws.month == 3 or we.month == 3) else "no"
        print(
            f"  in March? {march} (2024/25 grower date {GT:%m-%d}; offset of 2026 window to "
            f"{GT.replace(year=2026):%Y-%m-%d}: "
            f"{window_offset(ws, we, GT.replace(year=2026)):.0f} d)"
        )
        if not args.no_write:
            write_window(FIELDS, FIELD_ID, ws, we)
    else:
        could_not.append("p36 2025/26: no NDVI step found, harvest window left null")

    # ---- 3. harvest front ---------------------------------------------------------------------
    print(f"\n## 3. Harvest front 2024/25: cane segments within {NEIGH_RADIUS_M} m of {FIELD_ID}")
    seg_path = OUT_DIR / "segments.geojson"
    if args.from_cache and seg_path.exists():
        segs = gpd.read_file(seg_path)
    else:
        segs = segment_neighbourhood(geom)
        segs.to_file(seg_path, driver="GeoJSON")
    print(
        f"{len(segs)} SNIC segments {MIN_SEG_HA:g}-{MAX_SEG_HA:g} "
        f"ha, centroid ≤ {NEIGH_RADIUS_M} m from {FIELD_ID}"
    )
    nd_path, s1_path = OUT_DIR / "neigh_ndvi.csv", OUT_DIR / "neigh_s1.csv"
    if args.from_cache and nd_path.exists() and s1_path.exists():
        nd_n = pd.read_csv(nd_path, parse_dates=["date"])
        s1_n = pd.read_csv(s1_path, parse_dates=["date"])
    else:
        nd_n, s1_n = neighbour_series(segs)
        nd_n.to_csv(nd_path, index=False, date_format="%Y-%m-%d")
        s1_n.to_csv(s1_path, index=False, date_format="%Y-%m-%d")
    res = pick_neighbours(segs, nd_n, s1_n, args.max_field_cloud)
    print(
        f"cane (exercise-7 tests, max ≥ {X7.PEAK}): {int(res['cane'].sum())}/{len(res)}; "
        f"chosen {int(res['chosen'].sum())} (nearest first, "
        f"centroids ≥ {MIN_NEIGH_SPACING_M} m apart)"
    )
    sel = res[res["chosen"]].copy()
    sel["s2_off_d"] = [
        window_offset(a, b) for a, b in zip(sel["s2_start"], sel["s2_end"], strict=True)
    ]
    for o in ("asc", "desc"):
        sel[f"s1_{o}_off_d"] = [
            window_offset(a, b)
            for a, b in zip(sel[f"s1_{o}_before"], sel[f"s1_{o}_after"], strict=True)
        ]
    print(
        "\n| segment | dist m | ha | NDVI max | S2 window 2025 | "
        "off d | S1 asc largest drop | off d "
        "| S1 desc largest drop | off d |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in sel.iterrows():
        print(
            f"| {r['seg_id']} | {r['dist_m']:.0f} | {r['area_ha']:.1f} | {r['ndvi_max']:.2f} | "
            f"{fmt_win(r['s2_start'], r['s2_end'])} | {r['s2_off_d']:.0f} | "
            f"{fmt_win(r['s1_asc_before'], r['s1_asc_after'])} ({r['s1_asc_step_db']:+.1f} dB) | "
            f"{r['s1_asc_off_d']:.0f} | {fmt_win(r['s1_desc_before'], r['s1_desc_after'])} "
            f"({r['s1_desc_step_db']:+.1f} dB) | {r['s1_desc_off_d']:.0f} |"
        )
    k = len(sel)
    for col, name in (
        ("s2_off_d", "S2 NDVI window"),
        ("s1_asc_off_d", "S1 asc largest drop"),
        ("s1_desc_off_d", "S1 desc largest drop"),
    ):
        print(
            f"- {name} within ± {TOL_DAYS} d of {GT:%Y-%m-%d}: "
            f"{int((sel[col] <= TOL_DAYS).sum())}/{k}"
        )
    widths = (sel["s2_end"] - sel["s2_start"]).dt.days
    print(f"- S2 window widths: median {widths.median():.0f} d, max {widths.max():.0f} d")
    if k < 6:
        could_not.append(f"harvest front: only {k} cane segments found (asked for 6-10)")

    out = sel[
        [
            "seg_id",
            "snic_id",
            "dist_m",
            "area_ha",
            "ndvi_max",
            "label",
            "reason",
            "n_clear",
            "s2_start",
            "s2_end",
            "s2_off_d",
            "s1_asc_before",
            "s1_asc_after",
            "s1_asc_step_db",
            "s1_asc_off_d",
            "s1_desc_before",
            "s1_desc_after",
            "s1_desc_step_db",
            "s1_desc_off_d",
            "geometry",
        ]
    ].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"seg_id": "id"}).round(3)
    out.insert(1, "source", "SNIC segment (Earth Engine), not a surveyed field")
    gpd.GeoDataFrame(out, crs="EPSG:4326").to_file(NEIGHBOURS_GEOJSON, driver="GeoJSON")

    # ---- plots --------------------------------------------------------------------------------
    plot_series(s1, ndvi, season_f, args.max_field_cloud, OUT_DIR / "p36_2025_series.png")
    plot_front(p36, res, OUT_DIR / "p36_front_2025.png")

    print("\n## Could not verify\n")
    print("\n".join(f"- {c}" for c in could_not) or "- (none)")
    print("\n## Files\n")
    for path in (
        NEIGHBOURS_GEOJSON,
        OUT_DIR / "p36_2025_series.png",
        OUT_DIR / "p36_front_2025.png",
    ):
        print(f"- {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
