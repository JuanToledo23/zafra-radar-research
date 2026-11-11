"""Per-polygon statistics for thousands of polygons at once (exercise 4d).

The per-field extractors (timeseries.extract_series, multisignal.extract_coherence_series,
features.enl_table, optical.s2_ndvi_series) loop over fields: one window read or one Earth
Engine request per polygon. That is fine for 38 fields and far too slow for ~10^4 segments.
Here each scene is read **once** over the whole area of interest, the polygons are burnt
into a label raster on that scene's grid, and every statistic is a `np.bincount` over the
labels. The pixel rules are the same as the per-field code:

* a pixel belongs to a polygon if its **centre** is inside (rasterize, all_touched=False);
* GCOV: valid where the `mask` layer is neither 0 nor the fill value and the value is
  finite and > 0; field mean in **linear power**, then dB; valid_px is the smaller of the
  HH and HV counts; ENL = mean^2 / var (ddof=1) of the linear HV pixels;
* GUNW coherence: pixels with a finite value; mean and minimum per polygon;
* S2 NDVI: mean over SCL-clear pixels; cloud_pct = 100 x (1 - clear / covered).

Where two polygons overlap, a pixel goes to the one drawn last; segments do not overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GCOV_POLS = ("HVHV", "HHHH")


def rasterize_ids(geoms, transform, shape) -> np.ndarray:
    """Label raster: 1..n for geoms[0..n-1] (pixel centre inside), 0 elsewhere."""
    from rasterio.features import rasterize

    shapes = ((g, i + 1) for i, g in enumerate(geoms))
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")


def zonal_sum(labels: np.ndarray, values: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    """Sum of `values` over valid pixels for labels 1..n (index 0 of the result = label 1)."""
    lab = labels[valid]
    return np.bincount(lab, weights=values[valid].astype("float64"), minlength=n + 1)[1:]


def zonal_count(labels: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(labels[valid], minlength=n + 1)[1:]


def zonal_min(labels: np.ndarray, values: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n + 1, np.inf)
    np.minimum.at(out, labels[valid], values[valid].astype("float64"))
    out[np.isinf(out)] = np.nan
    return out[1:]


def _aoi_window(grid: dict, bounds):
    """(rows, cols, transform) of a product grid around metric bounds (minx, miny, maxx, maxy)."""
    from shapely.geometry import box

    from zafra import gcov

    return gcov.window_for_geometry(grid, box(*bounds))


# --- GCOV ---------------------------------------------------------------------------------


def _on_grid(gdf_metric, bounds, epsg: int):
    """Polygons and AOI bounds in the product grid's CRS. A frame whose grid is in another
    UTM zone than the site's metric CRS (two orbits over a zone boundary) gets the polygons
    reprojected; on the site's own grid they are used as they are."""
    if gdf_metric.crs.to_epsg() == epsg:
        return gdf_metric, bounds
    g = gdf_metric.to_crs(epsg)
    minx, miny, maxx, maxy = g.total_bounds
    return g, (minx - 50, miny - 50, maxx + 50, maxy + 50)


def gcov_scene_stats(path, gdf_metric, ids, bounds) -> pd.DataFrame:
    """One row per polygon for one GCOV scene: HVHV_db, HHHH_db, ratio_db, valid_px,
    mean_looks, inside_px, enl. `gdf_metric` is reprojected to the grid CRS if needed."""
    from zafra import gcov

    with gcov.open_gcov(path) as f:
        ident = gcov.read_identification(f)
        grid = gcov.read_grid(f, "A")
        gdf_metric, bounds = _on_grid(gdf_metric, bounds, grid["epsg"])
        rows, cols, tr = _aoi_window(grid, bounds)
        shape = (rows.stop - rows.start, cols.stop - cols.start)
        lab = rasterize_ids(gdf_metric.geometry, tr, shape)
        n = len(ids)
        minfo = gcov.mask_info(f, "A")
        ok = gcov.valid_from_mask(gcov.read_window(f, "A", "mask", rows, cols), minfo)
        inside = lab > 0
        looks = gcov.read_window(f, "A", "numberOfLooks", rows, cols).astype("float64")
        out = {"field_id": ids, "date": pd.Timestamp(ident["start_time"][:10])}
        counts = {}
        for pol in GCOV_POLS:
            a = gcov.read_window(f, "A", pol, rows, cols).astype("float64")
            v = inside & ok & np.isfinite(a) & (a > 0)
            c = zonal_count(lab, v, n)
            s = zonal_sum(lab, a, v, n)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = s / c
                out[f"{pol}_db"] = 10 * np.log10(mean)
            counts[pol] = c
            if pol == "HVHV":
                s2 = zonal_sum(lab, a * a, v, n)
                with np.errstate(divide="ignore", invalid="ignore"):
                    var = (s2 - c * mean**2) / (c - 1)
                    out["enl"] = np.where(c >= 2, mean**2 / var, np.nan)
                    out["mean_looks"] = zonal_sum(lab, looks, v, n) / c
        out["valid_px"] = np.minimum(counts["HVHV"], counts["HHHH"])
        out["inside_px"] = zonal_count(lab, inside, n)
    df = pd.DataFrame(out)
    df["ratio_db"] = df["HVHV_db"] - df["HHHH_db"]
    return df


# --- GUNW coherence -----------------------------------------------------------------------


def gunw_pair_stats(path, gdf_metric, ids, bounds) -> pd.DataFrame:
    """One row per polygon for one GUNW pair: ref_date, sec_date, coh_mean, coh_min,
    valid_px (finite coherence, pixel centre inside)."""
    import h5py

    from zafra import gcov
    from zafra import multisignal as ms

    with h5py.File(path, "r") as f:
        ident = gcov._require(f, ms.GUNW_IDENT)
        ref = gcov._decode(ident["referenceZeroDopplerStartTime"][()])[:10]
        sec = gcov._decode(ident["secondaryZeroDopplerStartTime"][()])[:10]
        grid = ms.read_gunw_grid(f)
        gdf_metric, bounds = _on_grid(gdf_metric, bounds, grid["epsg"])
        rows, cols, tr = _aoi_window(grid, bounds)
        shape = (rows.stop - rows.start, cols.stop - cols.start)
        lab = rasterize_ids(gdf_metric.geometry, tr, shape)
        coh = f[ms.GUNW_COH][rows, cols].astype("float64")
    n = len(ids)
    v = (lab > 0) & np.isfinite(coh)
    c = zonal_count(lab, v, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = zonal_sum(lab, coh, v, n) / c
    return pd.DataFrame(
        {
            "field_id": ids,
            "ref_date": pd.Timestamp(ref),
            "sec_date": pd.Timestamp(sec),
            "coh_mean": mean,
            "coh_min": zonal_min(lab, coh, v, n),
            "valid_px": c,
        }
    )


def radar_timeseries(
    gdf, asc_dir, desc_dir, gunw_dir, cache_dir, id_col="id", log=print, crs="EPSG:32614"
):
    """(gcov, coherence) tables for every polygon of `gdf`, one pass per scene.

    gcov: field_id, orbit, date, HVHV_db, HHHH_db, ratio_db, valid_px, mean_looks, enl,
    inside_px. coherence: field_id, ref_date, sec_date, coh_mean, coh_min, valid_px.
    Each scene's result is cached in `cache_dir` (parquet named after the granule), keyed
    by a hash of the polygons, so an interrupted run resumes and a changed set rebuilds.
    `crs` is the site's metric CRS (default Morelos UTM 14N); each product's grid may differ.
    """
    import hashlib
    import time
    from pathlib import Path

    from zafra import multisignal as ms
    from zafra import timeseries as ts

    g = gdf.to_crs(crs)
    ids = g[id_col].to_numpy()
    key = hashlib.sha1(
        b"".join(g.geometry.to_wkb()) + "|".join(map(str, ids)).encode()
    ).hexdigest()[:12]
    cache_dir = Path(cache_dir) / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = g.total_bounds
    bounds = (minx - 50, miny - 50, maxx + 50, maxy + 50)

    def cached(path, build):
        if path.exists():
            return pd.read_parquet(path)
        df = build()
        df.to_parquet(path, index=False)
        return df

    frames = []
    for orbit, d in (("asc", asc_dir), ("desc", desc_dir)):
        for s in ts.list_scenes(d):
            t = time.perf_counter()
            df = cached(
                cache_dir / f"{s['path'].stem}.parquet",
                lambda s=s: gcov_scene_stats(s["path"], g, ids, bounds),
            )
            frames.append(df.assign(orbit=orbit))
            log(f"GCOV {orbit} {s['date']}: {time.perf_counter() - t:.1f} s")
    coh = []
    for p in ms.list_pairs(gunw_dir):
        t = time.perf_counter()
        coh.append(
            cached(
                cache_dir / f"{p['path'].stem}.parquet",
                lambda p=p: gunw_pair_stats(p["path"], g, ids, bounds),
            )
        )
        log(f"GUNW {p['ref_date']}/{p['sec_date']}: {time.perf_counter() - t:.1f} s")
    gc = pd.concat(frames, ignore_index=True)
    return gc, pd.concat(coh, ignore_index=True)


def drop_excluded(gc, coh, site):
    """Remove the site's excluded acquisitions (Site.nisar_exclude) from radar_timeseries
    output: GCOV rows of the orbit on that date, and every GUNW pair with it as reference
    or secondary date. Returns (gc, coh, dropped) with dropped = row counts per table."""
    keep = pd.Series(True, index=gc.index)
    for orbit, role in (("asc", "gcov_asc"), ("desc", "gcov_desc")):
        dates = site.excluded_dates(role)
        if dates:
            day = pd.to_datetime(gc["date"]).dt.date.astype(str)
            keep &= ~((gc["orbit"] == orbit) & day.isin(dates))
    dates = site.excluded_dates("gunw_asc")
    keep_c = pd.Series(True, index=coh.index)
    for col in ("ref_date", "sec_date"):
        keep_c &= ~pd.to_datetime(coh[col]).dt.date.astype(str).isin(dates)
    dropped = {"gcov_rows": int((~keep).sum()), "coherence_rows": int((~keep_c).sum())}
    return gc[keep].reset_index(drop=True), coh[keep_c].reset_index(drop=True), dropped


# --- Sentinel-2 NDVI stack ----------------------------------------------------------------


def ndvi_series_from_stack(gdf, stack_tif, id_col="id") -> pd.DataFrame:
    """Per-polygon, per-date NDVI from the int16 stack of optical.ndvi_stack_image, in the
    layout of optical.s2_ndvi_series (+ `id`): date, ndvi_mean, cloud_pct, n_px, n_total,
    tile_cloud_pct (NaN), image_id (= "d_YYYYMMDD", the band name). Bare shares come
    from bare_shares_from_stack, for the (id, image_id) pairs build_table asks for.
    """
    import rasterio

    from zafra import optical

    with rasterio.open(stack_tif) as ds:
        g = gdf.to_crs(ds.crs)
        lab = rasterize_ids(g.geometry, ds.transform, (ds.height, ds.width))
        names = list(ds.descriptions)
        n = len(g)
        rows = []
        for b, name in enumerate(names, 1):
            a = ds.read(b)
            covered = a != optical.STACK_NODATA
            clear = covered & (a != optical.STACK_MASKED)
            c = zonal_count(lab, clear, n)
            tot = zonal_count(lab, covered, n)
            s = zonal_sum(lab, a, clear, n) / 10000
            with np.errstate(invalid="ignore", divide="ignore"):
                rows.append(
                    pd.DataFrame(
                        {
                            "id": g[id_col].to_numpy(),
                            "date": pd.Timestamp(name[2:]),
                            "ndvi_mean": s / c,
                            "n_px": c,
                            "n_total": tot,
                            "image_id": name,
                        }
                    )
                )
    df = pd.concat(rows, ignore_index=True)
    df = df[df["n_total"] > 0].copy()
    df["cloud_pct"] = 100 * (1 - df["n_px"] / df["n_total"])
    df["tile_cloud_pct"] = np.nan
    cols = ["date", "ndvi_mean", "cloud_pct", "n_px", "n_total", "tile_cloud_pct", "image_id"]
    return df[[*cols, "id"]].sort_values(["id", "date"], ignore_index=True)


def bare_shares_from_stack(gdf, stack_tif, requests, low: float = 0.35, id_col="id") -> dict:
    """{(id, image_id): share of the polygon's clear pixels with NDVI <= low} for each
    requested (id, image_id), as optical.s2_bare_share does with one EE call per field."""
    import rasterio

    from zafra import optical

    req = pd.DataFrame(requests, columns=["id", "image_id"])
    if req.empty:
        return {}
    with rasterio.open(stack_tif) as ds:
        g = gdf.to_crs(ds.crs)
        lab = rasterize_ids(g.geometry, ds.transform, (ds.height, ds.width))
        pos = {fid: i for i, fid in enumerate(g[id_col])}
        names = list(ds.descriptions)
        n = len(g)
        out = {}
        for name, r in req.groupby("image_id"):
            a = ds.read(names.index(name) + 1)
            clear = (a != optical.STACK_NODATA) & (a != optical.STACK_MASKED)
            c = zonal_count(lab, clear, n)
            bare = zonal_count(lab, clear & (a <= low * 10000), n)
            for fid in r["id"]:
                i = pos[fid]
                out[(fid, name)] = bare[i] / c[i] if c[i] else np.nan
    return out
