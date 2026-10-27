"""Sentinel-2 NDVI per field, as an independent (optical) check on the radar harvest dates.

Why a second sensor
-------------------
The radar detector (exercise 4) sees harvest as a drop in L-band HV backscatter: microwaves
at ~24 cm wavelength bounce around inside the cane canopy (volume scattering), and cutting
the cane removes that volume. Sentinel-2 measures something physically different: sunlight
reflected by the surface in visible and near-infrared bands. The two share no hardware, no
physics and no processing chain, so their errors are (mostly) unrelated. Radar can be fooled
by soil moisture after rain or by lodged cane; optical is blind under clouds and smoke. If
both say "the cane went away between these dates", that is much stronger evidence than
either one alone.

What NDVI is
------------
NDVI = (NIR - Red) / (NIR + Red), with Sentinel-2 B8 (842 nm, near-infrared) and B4
(665 nm, red). Green leaves absorb red light for photosynthesis and strongly reflect
near-infrared (leaf cell structure scatters it). So:
  * dense green cane:      NIR >> Red  -> NDVI ~0.7-0.9
  * bare soil / stubble:   NIR ~ Red   -> NDVI ~0.1-0.3
  * burned trash, ash:     low NIR     -> NDVI can go near 0
NDVI is a ratio, so the 1e-4 reflectance scale factor of the L2A product cancels out.

Why cane NDVI saturates
-----------------------
Once the canopy covers the ground completely (leaf area index above ~3, which mature cane
exceeds by far), adding more leaves barely changes red or NIR at the top of the canopy:
NDVI flattens around 0.8-0.9. So NDVI cannot tell a 10-month-old field from a 14-month-old
one, and it cannot see the stalks (what is harvested) at all, only the leaves on top. That
is fine for us: we only ask a yes/no question, "standing canopy or not", and the jump from
~0.8 to ~0.2 at harvest is large. It also means a field can look "standing" in NDVI while
the cane inside is already drying for harvest (NDVI only drops gently as leaves dry).

Why SCL masking, per pixel
--------------------------
Clouds are bright in red and NIR alike, giving NDVI ~0; cloud *shadows* darken everything
and give noisy NDVI. Either would look like a fake harvest. The L2A product ships a Scene
Classification Layer (SCL, 20 m, produced by ESA's Sen2Cor) labelling every pixel as cloud,
shadow, vegetation, soil, etc. We keep only pixels labelled vegetation, bare soil, water or
unclassified, and drop the rest *pixel by pixel*. The scene-wide CLOUDY_PIXEL_PERCENTAGE
metadata is only a coarse pre-filter: a tile 30% cloudy can still be clear over our 2-7 ha
field, and a 2% cloudy tile can have its one cloud right on top of it. So we also compute
the cloud percentage *inside the field* and use that to judge whether a date is usable.
"""

from __future__ import annotations

import ee
import google.oauth2.credentials
import numpy as np
import pandas as pd

from zafra import config

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# SCL classes (Sen2Cor). Keep only "clean surface" classes; everything else is masked.
#  0 no data            1 saturated/defective   2 dark area / topographic shadow
#  3 cloud shadow       4 vegetation            5 bare soil (not vegetated)
#  6 water              7 unclassified          8 cloud, medium probability
#  9 cloud, high prob. 10 thin cirrus          11 snow / ice
SCL_KEEP = (4, 5, 6, 7)

_initialized = False


def init_ee() -> None:
    """Initialise Earth Engine with the project in .env (EE_PROJECT). Safe to call twice.

    Credentials come from `earthengine authenticate` (~/.config/earthengine/credentials).
    We build them explicitly instead of letting ee.Initialize() look them up, because its
    lookup also opens gcloud's application_default_credentials.json and crashes if that
    path exists but is not a readable file. Nothing secret is printed.
    """
    global _initialized
    if _initialized:
        return
    project = config.ee_project()
    try:
        args = ee.oauth.get_credentials_arguments()
    except OSError:
        args = {}
    if args.get("refresh_token"):
        creds = google.oauth2.credentials.Credentials(None, **args).with_quota_project(project)
        ee.Initialize(credentials=creds, project=project)
    else:
        ee.Initialize(project=project)  # fall back to EE's own lookup (e.g. gcloud ADC)
    _initialized = True


def _to_ee_geometry(geom) -> ee.Geometry:
    """Accept a shapely geometry (lon/lat, EPSG:4326) or an ee.Geometry."""
    if isinstance(geom, ee.Geometry):
        return geom
    return ee.Geometry(geom.__geo_interface__)


def _field_stats(img: ee.Image, region: ee.Geometry) -> ee.Feature:
    """NDVI mean over clear pixels + clear/total pixel counts for one image and one field."""
    scl = img.select("SCL")
    clear = scl.remap(list(SCL_KEEP), [1] * len(SCL_KEEP), 0).eq(1)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(clear)
    # `total` counts every pixel of the field that has data (SCL 0 = no data, off-swath).
    total = scl.neq(0).selfMask().rename("total")
    # Reduce on the 10 m grid of the red band. SCL is 20 m; Earth Engine resamples it by
    # nearest neighbour, so each 20 m class label covers four 10 m pixels.
    proj = img.select("B4").projection()
    # Unweighted reducers: a pixel either counts fully (centre inside the polygon) or not at
    # all. The default weighted mean would also take in edge pixels that merely touch the
    # field, with fractional weights, so clear and total counts would not be comparable.
    stats = ndvi.addBands(total).reduceRegion(
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
            "tile_cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"),
            "ndvi_mean": stats.get("ndvi_mean"),
            "n_px": stats.get("ndvi_count"),
            "n_total": stats.get("total_count"),
        },
    )


def s2_ndvi_series(field_geom, start: str, end: str, max_cloud: float = 30) -> pd.DataFrame:
    """Per-date mean NDVI of one field from Sentinel-2 L2A, with SCL cloud/shadow masking.

    field_geom: shapely polygon in lon/lat (or an ee.Geometry).
    start, end: "YYYY-MM-DD"; end is exclusive (Earth Engine filterDate convention).
    max_cloud:  scene-level pre-filter on CLOUDY_PIXEL_PERCENTAGE (whole 110 km tile).

    Returns one row per acquisition date, sorted, with columns:
      date        acquisition date (UTC; S2 passes Morelos ~17:00 UTC = ~11:00 local)
      ndvi_mean   mean NDVI over the field's clear 10 m pixels (NaN if none are clear)
      cloud_pct   % of the field's pixels masked by SCL (cloud, shadow, cirrus, ...)
      n_px        number of clear pixels averaged
    plus image_id (for thumbnails), tile_cloud_pct and n_total (field pixels with data).

    A pixel counts as "in the field" if its centre falls inside the polygon (Earth Engine's
    reduceRegion rule), the same rule used for the NISAR pixels in exercise 3.
    """
    init_ee()
    region = _to_ee_geometry(field_geom)
    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )
    fc = ee.FeatureCollection(coll.map(lambda img: _field_stats(img, region)))
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    cols = ["date", "ndvi_mean", "cloud_pct", "n_px", "n_total", "tile_cloud_pct", "image_id"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time_ms"], unit="ms").dt.normalize()
    df["n_px"] = df["n_px"].fillna(0).astype(int)
    df["n_total"] = df["n_total"].fillna(0).astype(int)
    df = df[df["n_total"] > 0]  # field outside this granule's swath
    df["cloud_pct"] = 100 * (1 - df["n_px"] / df["n_total"])
    df["ndvi_mean"] = pd.to_numeric(df["ndvi_mean"], errors="coerce")
    # The same date can appear twice (two granules / processing baselines of the same pass).
    # Keep the copy that sees the most clear pixels.
    df = (
        df.sort_values(["date", "n_px"], ascending=[True, False])
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    return df[cols]


def s2_bare_share(field_geom, image_id: str, low: float = 0.35) -> float:
    """Share of a field's clear 10 m pixels with NDVI <= `low` in one S2 L2A image.

    Same pixel rule as s2_ndvi_series (SCL classes SCL_KEEP, centre inside the polygon,
    unweighted). A cane cut clears the whole field at once, so the scene after it is mostly
    bare; a pasture browning out or a partial cut leaves a mix. NaN if no pixel is clear.
    """
    init_ee()
    region = _to_ee_geometry(field_geom)
    img = ee.Image(f"{S2_COLLECTION}/{image_id}")
    clear = img.select("SCL").remap(list(SCL_KEEP), [1] * len(SCL_KEEP), 0).eq(1)
    bare = img.normalizedDifference(["B8", "B4"]).lte(low).rename("bare").updateMask(clear)
    stats = bare.reduceRegion(
        reducer=ee.Reducer.mean().unweighted(),
        geometry=region,
        crs=img.select("B4").projection(),
        scale=10,
        maxPixels=1e8,
    ).getInfo()
    v = stats.get("bare")
    return float("nan") if v is None else float(v)


def clearest_before_after(
    series: pd.DataFrame,
    window_start,
    window_end,
    days: int = 20,
    max_field_cloud: float = 10,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Clearest scene within `days` before window_start and within `days` after window_end.

    window_start/end are the radar scenes bracketing the harvest (last "standing" and first
    "cut" acquisition). Optical scenes *between* them are skipped on purpose: the cut
    happened at an unknown moment in there, so such a scene could legitimately show either
    state and would not test the window.

    "Clearest" = lowest in-field cloud_pct; ties go to the scene closest to the window
    (less time for the field to change for other reasons). Scenes with in-field cloud_pct
    above `max_field_cloud` do not count as clear. Returns (before, after); either is None
    if no clear scene exists in that interval.
    """
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    ok = series[(series["cloud_pct"] <= max_field_cloud) & series["ndvi_mean"].notna()]

    def pick(sub: pd.DataFrame, anchor: pd.Timestamp) -> pd.Series | None:
        if sub.empty:
            return None
        sub = sub.assign(_gap=(sub["date"] - anchor).abs())
        return sub.sort_values(["cloud_pct", "_gap"]).iloc[0].drop("_gap")

    before = ok[(ok["date"] >= ws - pd.Timedelta(days=days)) & (ok["date"] <= ws)]
    after = ok[(ok["date"] >= we) & (ok["date"] <= we + pd.Timedelta(days=days))]
    return pick(before, ws), pick(after, we)


def rgb_thumb_url(image_id: str, field_geom, pad_m: float = 250, size: int = 400) -> str:
    """URL of a true-colour PNG (B4,B3,B2) of one S2 image around a field, outline in yellow.

    The URL is a temporary signed link to Earth Engine's thumbnail service; it carries no
    credentials of ours, but we still do not print it.
    """
    init_ee()
    region = _to_ee_geometry(field_geom)
    img = ee.Image(f"{S2_COLLECTION}/{image_id}")
    # Reflectance 0-0.3 (0-3000 in L2A units) stretched to 0-255; gamma > 1 brightens the
    # dark vegetation so cane and bare soil are easy to tell apart by eye.
    rgb = img.visualize(bands=["B4", "B3", "B2"], min=0, max=3000, gamma=1.3)
    outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(region)]), 1, 2)
    rgb = rgb.blend(outline.visualize(palette=["ffff00"]))
    return rgb.getThumbURL(
        {"region": region.buffer(pad_m).bounds(), "dimensions": size, "format": "png"}
    )


# --- area-wide images for field segmentation (exercise 4d) ----------------------------------

UTM_EPSG = 32614
WORLDCOVER = "ESA/WorldCover/v200"
# getDownloadURL refuses requests above ~48 MB; 1000 x 1000 px x 5 float32 bands is 20 MB.
TILE_PX = 1000
FILL = -9999.0  # masked pixels in transit; NaN in the written GeoTIFF
# NDVI stack codes (int16): NDVI x 10000 where clear; these two where not.
STACK_MASKED = -20000  # covered by the scene, but SCL says cloud / shadow / cirrus / ...
STACK_NODATA = -32768  # not covered by any granule of that date


def aoi_grid(bbox_lonlat, scale: float = 10.0) -> dict:
    """The UTM 14N pixel grid (EPSG:32614) covering a lon/lat bbox, snapped to `scale` m.

    Every export uses this grid (crs_transform + dimensions), so tiles and products line up
    pixel for pixel. Returns epsg, transform (affine tuple), width, height.
    """
    from pyproj import Transformer

    lon0, lat0, lon1, lat1 = bbox_lonlat
    tr = Transformer.from_crs(4326, UTM_EPSG, always_xy=True)
    xs, ys = tr.transform([lon0, lon0, lon1, lon1], [lat0, lat1, lat0, lat1])
    x0 = np.floor(min(xs) / scale) * scale
    y1 = np.ceil(max(ys) / scale) * scale
    w = int(np.ceil((max(xs) - x0) / scale))
    h = int(np.ceil((y1 - min(ys)) / scale))
    return {
        "epsg": UTM_EPSG,
        "transform": (scale, 0.0, x0, 0.0, -scale, y1),
        "width": w,
        "height": h,
    }


def _clear_mask(img: ee.Image) -> ee.Image:
    return img.select("SCL").remap(list(SCL_KEEP), [1] * len(SCL_KEEP), 0).eq(1).rename("clear")


def _region(grid: dict) -> ee.Geometry:
    s, _, x0, _, _, y1 = grid["transform"]
    x1, y0 = x0 + s * grid["width"], y1 - s * grid["height"]
    return ee.Geometry.Rectangle([x0, y0, x1, y1], f"EPSG:{grid['epsg']}", False)


def scene_clear_fractions(grid: dict, start: str, end: str, max_cloud: float = 40):
    """Per acquisition date: share of the AOI that is covered and SCL-clear.

    Granules of the same date (adjacent MGRS tiles of one pass) are mosaicked first. The
    share is computed at 100 m, which is plenty for ranking dates.
    """
    init_ee()
    region = _region(grid)
    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )
    days = coll.aggregate_array("system:time_start").getInfo()
    days = sorted({pd.Timestamp(t, unit="ms").normalize() for t in days})
    rows = []
    for d in days:
        mos = coll.filterDate(
            d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        clear = _clear_mask(mos.mosaic()).unmask(0)
        frac = clear.reduceRegion(ee.Reducer.mean(), region, scale=100, crs=f"EPSG:{grid['epsg']}")
        ids = mos.aggregate_array("system:index")
        rows.append({"date": d, "clear_frac": frac.get("clear"), "image_ids": ids})
    info = ee.List([ee.Dictionary(r | {"date": r["date"].isoformat()}) for r in rows]).getInfo()
    df = pd.DataFrame(info)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("clear_frac", ascending=False, ignore_index=True)


def scene_image(date, grid: dict) -> ee.Image:
    """One date's mosaic: B4, B3, B2, B8 (reflectance 0-1) and NDVI, SCL-masked per pixel."""
    d = pd.Timestamp(date)
    mos = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(_region(grid))
        .filterDate(d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        .mosaic()
    )
    refl = mos.select(["B4", "B3", "B2", "B8"]).divide(10000)
    ndvi = mos.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return refl.addBands(ndvi).updateMask(_clear_mask(mos)).toFloat()


def temporal_composite(grid: dict, start: str, end: str, max_cloud: float = 60) -> ee.Image:
    """Per-pixel NDVI median, standard deviation and minimum over every SCL-clear look.

    Neighbouring cane fields are cut on different dates, so their NDVI histories differ even
    when they look alike on any one day: std and min separate them. Also returns n_clear.
    """
    init_ee()
    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(_region(grid))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(
            lambda im: (
                im.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(_clear_mask(im))
            )
        )
    )
    red = (
        ee.Reducer.median()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    out = coll.reduce(red)
    return out.select(
        ["ndvi_median", "ndvi_stdDev", "ndvi_min", "ndvi_count"],
        ["ndvi_median", "ndvi_std", "ndvi_min", "n_clear"],
    ).toFloat()


def worldcover_image() -> ee.Image:
    """ESA WorldCover 2021 (v200) land-cover class, 10 m (50 = built-up, 80 = water)."""
    init_ee()
    return ee.ImageCollection(WORLDCOVER).first().select("Map").toFloat()


def download_tiled(
    img: ee.Image, grid: dict, out_path, tile_px: int = TILE_PX, workers: int = 4, dtype="float32"
):
    """Download `img` on `grid` as one GeoTIFF, via getDownloadURL in tile_px x tile_px tiles.

    Each tile is requested with an explicit crs_transform + dimensions, so tiles abut exactly
    and are pasted into one array (no resampling). float32: masked pixels become NaN.
    int16: the image must carry its own no-data codes (ndvi_stack_image). Band names are
    written as band descriptions.
    """
    import io
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import rasterio
    import requests
    from affine import Affine

    init_ee()
    bands = img.bandNames().getInfo()
    s, _, x0, _, _, y1 = grid["transform"]
    W, H = grid["width"], grid["height"]
    is_float = np.dtype(dtype).kind == "f"
    out = np.full((len(bands), H, W), np.nan if is_float else STACK_NODATA, dtype=dtype)
    jobs = [(r, c) for r in range(0, H, tile_px) for c in range(0, W, tile_px)]

    def fetch(rc):
        r, c = rc
        h, w = min(tile_px, H - r), min(tile_px, W - c)
        url = img.unmask(FILL, sameFootprint=False).getDownloadURL(
            {
                "crs": f"EPSG:{grid['epsg']}",
                "crs_transform": [s, 0, x0 + c * s, 0, -s, y1 - r * s],
                "dimensions": f"{w}x{h}",
                "format": "GEO_TIFF",
            }
        )
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        data = resp.content
        if data[:2] == b"PK":  # some endpoints still zip the GeoTIFF
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                data = z.read(z.namelist()[0])
        with rasterio.MemoryFile(data) as mf, mf.open() as ds:
            arr = ds.read().astype(dtype)
        if is_float:
            arr[arr == FILL] = np.nan  # NaN cannot travel in the request JSON, so a sentinel does
        return r, c, arr

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, c, arr in ex.map(fetch, jobs):
            out[:, r : r + arr.shape[1], c : c + arr.shape[2]] = arr

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": len(bands),
        "width": W,
        "height": H,
        "crs": f"EPSG:{grid['epsg']}",
        "transform": Affine(*grid["transform"]),
        "nodata": np.nan if is_float else STACK_NODATA,
        "compress": "deflate",
        "predictor": 3 if is_float else 2,
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out)
        for i, b in enumerate(bands, 1):
            dst.set_band_description(i, b)
    return out_path, len(jobs)


def ndvi_stack_image(grid: dict, start: str, end: str, max_cloud: float = 50):
    """One int16 band per acquisition date: NDVI x 10000 on SCL-clear pixels, STACK_MASKED
    on covered but masked pixels, STACK_NODATA off-swath.

    Same scene pre-filter (tile CLOUDY_PIXEL_PERCENTAGE <= max_cloud) and pixel mask as
    s2_ndvi_series, but for every pixel of the AOI at once, so per-field series can be
    computed locally for any number of polygons. Granules of one date are mosaicked.
    Returns (image, dates, tile_cloud) where band i is dates[i] ("d_YYYYMMDD").
    """
    init_ee()
    region = _region(grid)
    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )
    info = coll.aggregate_array("system:time_start").getInfo()
    clouds = coll.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    days = pd.Series(clouds, index=[pd.Timestamp(t, unit="ms").normalize() for t in info])
    tile_cloud = days.groupby(level=0).mean().sort_index()
    bands = []
    for d in tile_cloud.index:
        mos = coll.filterDate(
            d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        m = mos.mosaic()
        ndvi = m.normalizedDifference(["B8", "B4"]).multiply(10000).round()
        covered = m.select("SCL").neq(0)
        code = (
            ee.Image(STACK_NODATA)
            .where(covered, STACK_MASKED)
            .where(covered.And(_clear_mask(m)), ndvi)
            .unmask(STACK_NODATA)
        )
        bands.append(code.toInt16().rename(f"d_{d:%Y%m%d}"))
    return (
        ee.ImageCollection(bands).toBands().rename([f"d_{d:%Y%m%d}" for d in tile_cloud.index]),
        list(tile_cloud.index),
        tile_cloud,
    )
