"""Automatic field delineation over the supply area (exercise 4d).

Two ways to cut a Sentinel-2 image into candidate field polygons:

* **SAM** (Segment Anything, Meta AI, ViT-B, Apache 2.0) through samgeo. SAM is a pretrained
  foundation model for promptable segmentation; in "automatic" mode it is prompted with a
  regular grid of points and returns one mask per object it finds. It was trained on
  natural photos, not satellite images: here it is used as-is, with no fine-tuning. The
  image is cut into overlapping tiles (samgeo batch mode), each tile is upsampled 2x so a
  2-3 ha field (15-20 px at 10 m) is ~30-40 px for SAM, and samgeo writes one binary raster:
  every mask's pixels, minus each mask's 1-px eroded border. The borders are what separate
  touching fields; connected components of that raster are the segments, and each is grown
  back by 1 px (`expand_labels`) to undo the erosion.
* **Classical**: Felzenszwalb graph segmentation of the crop-year NDVI stack (median, std, min).
  Neighbouring cane fields are cut on different dates, so their NDVI histories differ even
  where a single image shows two identical green blocks.

Both go through the same cleaning (clean_segments) and are scored against hand-traced
polygons with match_fields (IoU of the best-overlapping segment, over/under-segmentation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_CRS = "EPSG:32614"
MIN_HA, MAX_HA = 0.5, 60.0
SIMPLIFY_M = 5.0
WORLDCOVER_DROP = {50: "built-up", 80: "permanent water"}
OVER_SHARE = 0.20  # a segment covering > 20 % of the field counts as one of its pieces
UNDER_RATIO = 1.5  # best segment > 1.5 x the field area -> under-segmented

# SAM automatic mask generator. Defaults are tuned for photos (32 points per side, IoU 0.88,
# stability 0.95); fields at 10 m are small and low-contrast, so: a denser point grid and
# looser quality thresholds. min_mask_region_area removes specks and holes (pixels of the
# 2x-upsampled tile, i.e. 50 px = 12.5 px at 10 m = 0.125 ha).
SAM_KWARGS = {
    "points_per_side": 48,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 50,
}
SAM_TILE = 384  # core of each tile (px at 10 m); samgeo adds SAM_BOUND on every side
SAM_BOUND = 64
SAM_RESIZE = 1024  # (384 + 2*64) = 512 px read, upsampled to SAM's native 1024


# Felzenszwalb on the NDVI stack. Chosen from 3 settings (scale 50 / 100 / 200) by their IoU on
# the 38 reference fields (results/4d), so its score there is optimistic.
FZ_SCALE, FZ_SIGMA, FZ_MIN_PX = 100.0, 0.5, 30


# --- inputs for SAM -----------------------------------------------------------------------


def stretch_uint8(arr, lo: float, hi: float) -> np.ndarray:
    """Linear stretch of `arr` from [lo, hi] to 1..255 (0 is kept for NaN / no data)."""
    a = np.asarray(arr, dtype="float64")
    out = 1 + np.clip((a - lo) / (hi - lo), 0, 1) * 254
    return np.where(np.isfinite(a), out, 0).astype("uint8")


def write_sam_input(src_tif, bands, out_tif, pct=(2, 98)) -> dict:
    """3-band uint8 GeoTIFF for SAM from three bands (1-based) of `src_tif`.

    One stretch for the whole AOI (2-98 % of each band), so every tile sees the same colours.
    Returns {band: (lo, hi)}.
    """
    import rasterio

    with rasterio.open(src_tif) as ds:
        prof = ds.profile
        data = ds.read(list(bands))
        names = [ds.descriptions[b - 1] for b in bands]
    out, limits = [], {}
    for name, a in zip(names, data, strict=True):
        lo, hi = np.nanpercentile(a, pct)
        limits[name] = (float(lo), float(hi))
        out.append(stretch_uint8(a, lo, hi))
    prof.update(count=3, dtype="uint8", nodata=0, predictor=2)
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(np.stack(out))
    return limits


def run_sam(input_tif, out_tif, checkpoint_dir, device: str = "cpu", sam_kwargs=None) -> int:
    """samgeo automatic mask generation, tiled; writes the binary mask raster. Returns the
    number of tiles."""
    from samgeo import SamGeo
    from samgeo.common import calculate_sample_grid

    sam = SamGeo(
        model_type="vit_b",
        automatic=True,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        sam_kwargs=sam_kwargs or SAM_KWARGS,
    )
    sam.generate(
        str(input_tif),
        str(out_tif),
        batch=True,
        batch_sample_size=(SAM_TILE, SAM_TILE),
        bound=SAM_BOUND,
        sample_resize=(SAM_RESIZE, SAM_RESIZE),
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
    )
    import rasterio

    with rasterio.open(input_tif) as ds:
        return len(calculate_sample_grid(ds.height, ds.width, SAM_TILE, SAM_TILE, SAM_BOUND))


def mask_to_labels(mask: np.ndarray, grow_px: int = 1) -> np.ndarray:
    """Connected components (4-connectivity) of a binary mask, each grown by `grow_px` into
    the background so the eroded mask borders are given back."""
    from skimage.measure import label
    from skimage.segmentation import expand_labels

    lab = label(np.asarray(mask) > 0, connectivity=1)
    return expand_labels(lab, distance=grow_px) if grow_px else lab


# --- classical segmentation -------------------------------------------------------------------


def _norm(a, pct=(2, 98)):
    lo, hi = np.nanpercentile(a, pct)
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)


def classic_labels(
    median, std, minimum, scale: float = FZ_SCALE, sigma: float = FZ_SIGMA, min_size=FZ_MIN_PX
) -> np.ndarray:
    """Felzenszwalb graph segmentation of the NDVI stack (median, std, min as 3 channels).

    Each layer is stretched to 0-1 (2-98 %), so the three weigh alike. Felzenszwalb merges
    neighbouring pixels while the colour difference across their boundary is small relative to
    the variation inside each region; `scale` sets the preferred region size, `min_size` (px)
    the smallest region. Every pixel ends in one segment (labels from 1).

    A watershed of the Sobel edge strength (seeds = low-edge interiors) was tried first and
    under-segmented 27-33 of the 38 reference fields: the seeds leak across weak boundaries.
    """
    from skimage.segmentation import felzenszwalb

    img = np.dstack([_norm(x) for x in (median, std, minimum)])
    return felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size) + 1


# --- labels -> polygons -> cleaning -------------------------------------------------------------


def labels_to_polygons(labels: np.ndarray, transform, crs=METRIC_CRS):
    """One polygon per label value (> 0), pixel-edge outlines, in `crs`."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    lab = np.asarray(labels).astype("int32")
    recs = [
        (int(v), shape(g))
        for g, v in shapes(lab, mask=lab > 0, transform=transform, connectivity=4)
    ]
    gdf = gpd.GeoDataFrame({"label": [v for v, _ in recs]}, geometry=[g for _, g in recs], crs=crs)
    # A label can come out as several pieces (touching only at corners): one row per label.
    return gdf.dissolve("label", as_index=False)


def _polygonal(geom):
    """Keep the polygonal part of a geometry (make_valid can return collections)."""
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    if isinstance(geom, Polygon | MultiPolygon):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon | MultiPolygon)]
        if polys:
            from shapely.ops import unary_union

            return unary_union(polys)
    return None


def fix_geometries(gdf):
    """make_valid every geometry, keep polygonal parts, drop empties."""
    from shapely import make_valid

    g = gdf.copy()
    g["geometry"] = [_polygonal(make_valid(x)) if x is not None else None for x in g.geometry]
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g


def worldcover_majority(gdf, worldcover_tif) -> pd.Series:
    """Most frequent ESA WorldCover class inside each polygon (pixel centres)."""
    import rasterio
    from rasterstats import zonal_stats

    with rasterio.open(worldcover_tif) as ds:
        arr = ds.read(1)
        nod = ds.nodata
        tr = ds.transform
        crs = ds.crs
    arr = np.where(np.isfinite(arr), arr, 0).astype("int16")
    zs = zonal_stats(
        gdf.to_crs(crs).geometry, arr, affine=tr, nodata=0 if nod is None else 0, stats="majority"
    )
    return pd.Series([z["majority"] for z in zs], index=gdf.index)


def clean_segments(
    gdf,
    min_ha: float = MIN_HA,
    max_ha: float = MAX_HA,
    simplify_m: float = SIMPLIFY_M,
    worldcover_tif=None,
) -> tuple:
    """Fix, simplify, filter by area and land cover. Returns (clean gdf, counts dict).

    Order: make valid -> simplify (topology-preserving, `simplify_m`) -> make valid again
    (simplifying can create self-intersections) -> keep min_ha <= area <= max_ha -> drop
    polygons whose WorldCover majority class is built-up (50) or water (80), if a WorldCover
    raster is given. Areas in hectares in the metric CRS. Adds area_ha (and wc_class).
    """
    counts = {"raw": len(gdf)}
    g = fix_geometries(gdf.to_crs(METRIC_CRS))
    g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
    g = fix_geometries(g)
    counts["after_fix"] = len(g)
    g["area_ha"] = g.geometry.area / 1e4
    counts["too_small"] = int((g["area_ha"] < min_ha).sum())
    counts["too_large"] = int((g["area_ha"] > max_ha).sum())
    g = g[(g["area_ha"] >= min_ha) & (g["area_ha"] <= max_ha)]
    if worldcover_tif is not None and len(g):
        g["wc_class"] = worldcover_majority(g, worldcover_tif)
        bad = g["wc_class"].isin(list(WORLDCOVER_DROP))
        counts["worldcover_dropped"] = int(bad.sum())
        g = g[~bad]
    counts["kept"] = len(g)
    return g.reset_index(drop=True), counts


# --- evaluation against hand-traced polygons -------------------------------------------------


def match_fields(ref, seg, ref_id: str = "id") -> pd.DataFrame:
    """For every reference polygon: the segment with the largest intersection and its IoU,
    plus over- and under-segmentation flags.

    over_segmented:  at least two segments each cover > OVER_SHARE of the field's area.
    under_segmented: the best segment is > UNDER_RATIO x the field's area.
    A field no segment touches gets IoU 0 and best_segment None ("missed").
    """
    ref = ref.to_crs(METRIC_CRS).reset_index(drop=True)
    seg = seg.to_crs(METRIC_CRS).reset_index(drop=True)
    sidx = seg.sindex
    rows = []
    for _, r in ref.iterrows():
        g = r.geometry
        a = g.area
        cand = seg.iloc[sidx.query(g, predicate="intersects")]
        inter = cand.geometry.intersection(g).area if len(cand) else pd.Series(dtype=float)
        row = {
            "field_id": r[ref_id],
            "field_ha": a / 1e4,
            "best_segment": None,
            "segment_ha": np.nan,
            "iou": 0.0,
            "cover_share": 0.0,
            "n_pieces_gt20": 0,
            "over_segmented": False,
            "under_segmented": False,
        }
        if len(inter) and inter.max() > 0:
            k = inter.idxmax()
            s = seg.geometry.loc[k]
            i = inter.loc[k]
            row |= {
                "best_segment": seg.index[k] if "seg_id" not in seg else seg.loc[k, "seg_id"],
                "segment_ha": s.area / 1e4,
                "iou": i / (a + s.area - i),
                "cover_share": i / a,
            }
            pieces = int((inter / a > OVER_SHARE).sum())
            row["n_pieces_gt20"] = pieces
            row["over_segmented"] = pieces >= 2
            row["under_segmented"] = s.area > UNDER_RATIO * a
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_matches(m: pd.DataFrame) -> dict:
    """Median IoU, share with IoU >= 0.5, counts of over / under / missed."""
    return {
        "n_fields": len(m),
        "median_iou": float(m["iou"].median()),
        "mean_iou": float(m["iou"].mean()),
        "share_iou_ge_0_5": float((m["iou"] >= 0.5).mean()),
        "n_iou_ge_0_5": int((m["iou"] >= 0.5).sum()),
        "over_segmented": int(m["over_segmented"].sum()),
        "under_segmented": int(m["under_segmented"].sum()),
        "missed": int(m["best_segment"].isna().sum()),
    }


# --- merging fragments (exercise 4e) ------------------------------------------------------------

MERGE_MAX_DAYS = 6  # half the 12-day revisit: two fragments of one cut share a date to ~this
MIN_SHARED_EDGE_M = 10.0  # one 10 m pixel edge: corner contacts do not make neighbours


def adjacent_pairs(gdf, min_edge_m: float = MIN_SHARED_EDGE_M, tol_m: float = 1.0):
    """(i, j) positional pairs, i < j, of polygons sharing >= min_edge_m of boundary.

    gdf in a metric CRS. A boundary counts as shared where it lies within `tol_m` of the
    other polygon (simplified neighbours no longer share exact vertices).
    """
    geoms = gdf.geometry.reset_index(drop=True)
    left, right = geoms.sindex.query(geoms.buffer(tol_m), predicate="intersects")
    keep = left < right
    pairs = []
    for i, j in zip(left[keep], right[keep], strict=True):
        if geoms[i].boundary.intersection(geoms[j].buffer(tol_m)).length >= min_edge_m:
            pairs.append((int(i), int(j)))
    return pairs


def merge_groups(dates, statuses, pairs, max_days: int = MERGE_MAX_DAYS) -> np.ndarray:
    """Group ids after merging adjacent polygons with compatible harvest dates.

    Neighbours (i, j) are compatible if both have a date and they differ by <= max_days, or
    both have no date and the same status. Pairs are joined closest dates first (union-find),
    and a join is refused if the merged group's dates would span more than max_days, so a
    chain of small steps cannot drift (A-B 5 d and B-C 5 d do not give A-C 10 d).
    """
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    st = pd.Series(statuses).reset_index(drop=True)
    n = len(d)
    parent = list(range(n))
    lo, hi = d.copy(), d.copy()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for i, j in pairs:
        if pd.notna(d[i]) and pd.notna(d[j]):
            gap = abs((d[i] - d[j]).days)
            if gap <= max_days:
                edges.append((gap, i, j))
        elif pd.isna(d[i]) and pd.isna(d[j]) and st[i] == st[j]:
            edges.append((0, i, j))
    for _, i, j in sorted(edges):
        a, b = find(i), find(j)
        if a == b:
            continue
        if pd.notna(lo[a]):
            new_lo, new_hi = min(lo[a], lo[b]), max(hi[a], hi[b])
            if (new_hi - new_lo).days > max_days:
                continue
            lo[a], hi[a] = new_lo, new_hi
        parent[b] = a
    return np.array([find(i) for i in range(n)])


def dissolve_groups(gdf, groups, date_col: str, weighted=(), first=()):
    """One polygon per group: geometry union, area_ha summed, `date_col` and `weighted`
    columns area-weighted means, `first` columns taken from the largest member, plus
    n_merged and member_ids (comma-separated `seg_id`)."""
    g = gdf.assign(_grp=groups, _a=gdf.geometry.area)
    rows = []
    for _, m in g.groupby("_grp", sort=False):
        w = m["_a"].to_numpy()
        big = m.loc[m["_a"].idxmax()]
        row = {c: big[c] for c in first}
        dd = pd.to_datetime(m[date_col])
        if dd.notna().all():
            ref = dd.min()
            off = ((dd - ref).dt.total_seconds() * w).sum() / w.sum()
            row[date_col] = (ref + pd.Timedelta(seconds=off)).normalize()
        else:
            row[date_col] = pd.NaT
        for c in weighted:
            v = m[c].to_numpy(dtype="float64")
            ok = np.isfinite(v)
            row[c] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        row["area_ha"] = float(w.sum() / 1e4)
        row["n_merged"] = len(m)
        row["member_ids"] = ",".join(m["seg_id"].astype(str))
        row["geometry"] = m.geometry.union_all() if len(m) > 1 else m.geometry.iloc[0]
        rows.append(row)
    import geopandas as gpd

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
