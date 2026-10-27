"""Exercise 5, step 4: validation of the transfer to a second site, without ground truth.

    python scripts/16_transfer_validation.py [--site brazil] [--seed 5]

Inputs (from scripts 12 and 14 run with the same --site): the per-segment harvest table
(paths.harvest_geojson), the Sentinel-2 NDVI stack (paths.ndvi_stack), results weekly.csv
and area_summary.json, the NISAR acquisition dates.

1. Independent NDVI dates on a sample. 40 gated cane segments with a harvest date and 20
   gated cane segments without one, drawn at random (seeded). For each, the per-segment
   NDVI series from the local S2 stack and the script-07 harvest window: the first clear
   scene (<= 10 % of the segment masked) with NDVI <= 0.35 after one >= 0.5, inside the
   season; window = (previous clear scene, that scene]. Error = detector date - window
   midpoint. The table has the exercise-4b columns (median |error|, mean error, within
   +-6 / +-12 days, missed, false alarms) with the definitions spelled out in the report.
2. Weekly harvests vs VIIRS fires within 200 m of gated cane (weekly.csv): Pearson r in
   the same week and with fires one week later. Computed but not meaningful on a
   mechanized green-harvest site (no burning).
3. Distribution of the detected dates (p10, median, p90; count and area by month), with the
   share inside the site's season window.
   Date uncertainty: dated segments per date_uncertainty_days (half the acquisition gap).
4. Gate vs MapBiomas Collection 10 (2024, class 20 = sugar cane), majority class per
   segment on the S2 grid. An independent land-cover map, not ground truth. Both gates
   (the site's delivered gate_variant and the other one), plus variant B re-fitted on the
   MapBiomas labels ("with local labels"): out-of-fold P(cane) from 5-fold CV grouped by
   3 km spatial blocks of segment centroids. That row is a reference for what local labels
   buy; it is never the delivered gate.
5. Largest single HV drop (hv_drop_db) at harvest: median over dated cane segments, against
   the same figure on the Morelos area map and the Morelos labelled-cane median (-3.5 dB,
   38 fields_v2 polygons, exercise 4c).

Writes <results_dir>/validation_sample.csv, validation.json and validation.md.
"""

import argparse
import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from zafra import plots, zonal
from zafra import timeseries as ts
from zafra.detect import cane_gate as G
from zafra.site import add_site_argument, load_site

N_DATED, N_UNDATED = 40, 20
HARVEST_HIGH, LOW = 0.5, 0.35  # script 07: a scene >= 0.5, then the first one <= 0.35
MAX_FIELD_CLOUD = 10  # % of the segment masked for a scene to count as clear (script 07)
TOL = (6, 12)
CV_FOLDS, CV_BLOCK_M = 5, 3000  # local-label refit of gate B: folds, spatial block size
MAPBIOMAS = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection10/"
    "mapbiomas_brazil_collection10_integration_v2"
)
MAPBIOMAS_BAND = "classification_2024"
MB_CANE = 20
MORELOS_LABELLED_HV_DROP_DB = -3.5  # docs/VALIDATION.md, exercise 4c limitations
MB_NAMES = {  # the classes that occur in the AOI (MapBiomas legend, collection 10)
    3: "forest formation",
    4: "savanna formation",
    9: "forest plantation",
    11: "wetland",
    12: "grassland",
    15: "pasture",
    20: "sugar cane",
    21: "mosaic of uses",
    24: "urban area",
    25: "other non-vegetated",
    33: "river, lake",
    39: "soybean",
    41: "other temporary crops",
    46: "coffee",
    47: "citrus",
    48: "other perennial crops",
}


def ndvi_window(clear: pd.DataFrame, season) -> tuple:
    """Script-07 harvest window inside `season` (start, end exclusive): (prev, low] dates."""
    a, b = season
    armed, prev = False, None
    for r in clear.itertuples():
        if r.ndvi_mean >= HARVEST_HIGH:
            armed = True
        elif r.ndvi_mean <= LOW and armed:
            if a <= r.date < b:
                return prev, r.date
            armed = False
        prev = r.date
    return None, None


def mapbiomas_majority(seg, site) -> pd.Series:
    """MapBiomas 2024 majority class per segment, from a raster on the site's S2 grid
    (downloaded once into s2_dir)."""
    import rasterio

    from zafra import optical, segments

    path = site.path("s2_dir") / "mapbiomas_2024.tif"
    if not path.exists():
        import ee

        optical.init_ee()
        grid = optical.aoi_grid(site.aoi, epsg=site.epsg)
        # float32 like WorldCover: gaps become NaN, i.e. class 0 (no data) in the majority vote
        img = ee.Image(MAPBIOMAS).select(MAPBIOMAS_BAND).rename("mapbiomas").toFloat()
        optical.download_tiled(img, grid, path)
    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg() == site.epsg
    return segments.worldcover_majority(seg, path)


def local_refit(g: gpd.GeoDataFrame, site) -> np.ndarray:
    """Out-of-fold P(cane) of gate variant B re-fitted on g["mb_cane"] (MapBiomas majority
    labels), CV_FOLDS folds grouped by CV_BLOCK_M spatial blocks; NaN without optical data."""
    from sklearn.model_selection import GroupKFold

    cols = G.VARIANTS[G.OPTICAL_VARIANT]
    feats = (
        pd.read_parquet(site.path("cane_features"))
        .set_index("field_id")
        .reindex(g["seg_id"])
        .reset_index()
    )
    ok = np.flatnonzero(feats["has_optical"].fillna(False).astype(bool).to_numpy())
    c = g.to_crs(site.metric_crs).geometry.centroid
    block = (
        (c.x // CV_BLOCK_M).astype(int).astype(str)
        + "_"
        + (c.y // CV_BLOCK_M).astype(int).astype(str)
    )
    y = g["mb_cane"].to_numpy().astype(int)
    p = np.full(len(g), np.nan)
    for tr, te in GroupKFold(CV_FOLDS).split(ok, groups=block.to_numpy()[ok]):
        model = G.fit(feats.iloc[ok[tr]], y[ok[tr]], cols)
        p[ok[te]] = G.predict(model, feats.iloc[ok[te]], cols)
    return p


def error_table(s: pd.DataFrame, first_datable, method: str) -> tuple[pd.DataFrame, dict]:
    """4b-style summary over the sample, plus the categories behind each count."""
    dated = s[s["dated"]]
    und = s[~s["dated"]]
    has_w = dated["ndvi_end"].notna()
    e = dated.loc[has_w, "err_days"]
    a = e.abs()
    fa_none = int((~has_w).sum())
    und_w = und["ndvi_end"].notna()
    datable = und_w & (und["ndvi_end"] >= first_datable)
    cats = {
        "dated, NDVI window found": int(has_w.sum()),
        "dated, no NDVI harvest in the season (false alarm)": fa_none,
        "undated, NDVI harvest datable by radar (missed)": int(datable.sum()),
        "undated, NDVI harvest before the first datable date": int((und_w & ~datable).sum()),
        "undated, no NDVI harvest in the season": int((~und_w).sum()),
    }
    n = len(dated)
    row = {
        "method": method,
        "median |error| (d)": a.median(),
        "mean error (d)": e.mean(),
        f"within ±{TOL[0]} d": f"{int((a <= TOL[0]).sum())}/{n}",
        f"within ±{TOL[1]} d": f"{int((a <= TOL[1]).sum())}/{n}",
        "missed": f"{int(datable.sum())}/{len(und)}",
        "false alarms": f"{fa_none}/{n}",
    }
    return pd.DataFrame([row]), cats


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_site_argument(ap)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    site = load_site(args.site)
    out_dir = site.path("results_dir")
    t0 = time.perf_counter()

    g = gpd.read_file(site.path("harvest_geojson"))
    cane = g[g["cane_pred"].fillna(False).astype(bool)].copy()
    cane["dated"] = cane["harvest_date"].notna()
    rng = np.random.default_rng(args.seed)
    pick = []
    for flag, n in ((True, N_DATED), (False, N_UNDATED)):
        pool = cane.index[cane["dated"] == flag].to_numpy()
        pick += list(rng.choice(pool, size=min(n, len(pool)), replace=False))
    s = cane.loc[pick].copy()
    s["id"] = s["seg_id"]

    # 1. NDVI windows from the local stack
    ndvi = zonal.ndvi_series_from_stack(s, site.path("ndvi_stack"))
    clear = ndvi[(ndvi["cloud_pct"] <= MAX_FIELD_CLOUD) & ndvi["ndvi_mean"].notna()]
    wins = {fid: ndvi_window(c.sort_values("date"), site.season) for fid, c in clear.groupby("id")}
    s["n_clear"] = s["id"].map(clear.groupby("id").size()).fillna(0).astype(int)
    s["ndvi_start"] = s["id"].map(lambda f: wins.get(f, (None, None))[0])
    s["ndvi_end"] = s["id"].map(lambda f: wins.get(f, (None, None))[1])
    for c in ("ndvi_start", "ndvi_end", "harvest_date"):
        s[c] = pd.to_datetime(s[c])
    s["ndvi_mid"] = s["ndvi_start"] + (s["ndvi_end"] - s["ndvi_start"]) / 2
    s["err_days"] = (s["harvest_date"] - s["ndvi_mid"]).dt.total_seconds() / 86400

    # first date the 3-look baseline rule can emit: 4th acquisition of the earlier orbit
    acq = {
        o: sorted(pd.Timestamp(x["date"]) for x in ts.list_scenes(site.nisar[r].dir))
        for o, r in (("asc", "gcov_asc"), ("desc", "gcov_desc"))
    }
    first_datable = min(d[3] for d in acq.values() if len(d) > 3)
    first_radar = min(d[0] for d in acq.values())
    tab, cats = error_table(
        s, first_datable, f"gate {site.gate_variant} + 4b xgboost, 3-look baseline"
    )

    # 2. weekly harvests vs fires near cane
    weekly = pd.read_csv(out_dir / "weekly.csv", index_col=0, parse_dates=True)
    corr = {
        col: plots.week_correlation(weekly["harvests"], weekly[col], max_lag=1)
        .set_index("fires_weeks_after")["r"]
        .round(3)
        .to_dict()
        for col in ("fires_cane", "fires_all")
    }

    # 3. date distribution of every dated cane segment
    hd = pd.to_datetime(cane.loc[cane["dated"], "harvest_date"])
    dist = {f"p{int(q * 100)}": f"{hd.quantile(q):%Y-%m-%d}" for q in (0.1, 0.5, 0.9)}
    in_season = float(((hd >= site.season[0]) & (hd < site.season[1])).mean())
    order = np.argsort(hd.to_numpy())
    w = cane.loc[cane["dated"], "area_ha"].to_numpy()[order]
    cum = np.cumsum(w) / w.sum()
    dist_ha = {
        f"p{int(q * 100)}": f"{hd.iloc[order[np.searchsorted(cum, q)]]:%Y-%m-%d}"
        for q in (0.1, 0.5, 0.9)
    }
    month = pd.DataFrame(
        {
            "segments": hd.dt.to_period("M").value_counts().sort_index(),
            "ha": cane.loc[cane["dated"]].groupby(hd.dt.to_period("M"))["area_ha"].sum().round(0),
        }
    ).reset_index(names="month")
    dc = cane.loc[cane["dated"]]
    unc = (
        dc.groupby("date_uncertainty_days")
        .agg(
            segments=("seg_id", "size"),
            ha=("area_ha", "sum"),
            first_date=("harvest_date", "min"),
            last_date=("harvest_date", "max"),
        )
        .round({"ha": 0})
        .reset_index()
        .rename(columns={"date_uncertainty_days": "± days"})
    )
    for c in ("first_date", "last_date"):
        unc[c] = pd.to_datetime(unc[c]).dt.strftime("%Y-%m-%d")

    # 4. gate vs MapBiomas
    g["mb_class"] = mapbiomas_majority(g, site).to_numpy()
    g["mb_cane"] = g["mb_class"] == MB_CANE
    is_cane = g["cane_pred"].fillna(False).astype(bool)
    g["p_cane_local"] = local_refit(g, site)

    def agree(pred, name):
        tp = int((pred & g["mb_cane"]).sum())
        fp = int((pred & ~g["mb_cane"]).sum())
        fn = int((~pred & g["mb_cane"]).sum())
        tn = int((~pred & ~g["mb_cane"]).sum())
        ha = g["area_ha"]
        ha_g, ha_mb = float(ha[pred].sum()), float(ha[g["mb_cane"]].sum())
        ha_both = float(ha[pred & g["mb_cane"]].sum())
        return {
            "gate": name,
            "ha gated": round(ha_g),
            "% of AOI": f"{ha_g / ha.sum():.1%}",
            "ha MB cane": round(ha_mb),
            "ha both": round(ha_both),
            "precision (ha)": f"{ha_both / max(ha_g, 1e-9):.1%}",
            "recall (ha)": f"{ha_both / max(ha_mb, 1e-9):.1%}",
            "precision (segments)": f"{tp / max(tp + fp, 1):.1%}",
            "recall (segments)": f"{tp / max(tp + fn, 1):.1%}",
            "agree (segments)": f"{(tp + tn) / len(g):.1%}",
        }

    gates = [
        (G.RADAR_VARIANT, "A radar", "p_cane_radar"),
        (G.OPTICAL_VARIANT, "B optical", "p_cane_optical"),
    ]
    gates.sort(key=lambda x: x[0] != site.gate_variant)  # the delivered gate first
    rows = [
        agree(
            g[col].fillna(0) >= G.THRESHOLD,
            f"{name} ({'delivered' if v == site.gate_variant else 'not delivered'})",
        )
        for v, name, col in gates
    ]
    rows.append(
        agree(
            g["p_cane_local"].fillna(0) >= G.THRESHOLD,
            f"B optical, with local labels (MapBiomas, {CV_FOLDS}-fold CV by "
            f"{CV_BLOCK_M // 1000} km blocks; not delivered)",
        )
    )
    mb_tab = pd.DataFrame(rows)
    gated_mb = (
        g.loc[is_cane, "mb_class"]
        .map(lambda c: MB_NAMES.get(c, str(c)))
        .value_counts()
        .rename_axis("MapBiomas class of gated cane")
        .reset_index(name="segments")
    )
    by_conf = (
        g[is_cane]
        .groupby("gate_confidence")["mb_cane"]
        .agg(["size", "mean"])
        .rename(columns={"size": "segments", "mean": "MB cane share"})
        .reset_index()
    )

    # 5. HV drop at harvest, this site vs Morelos (same column, same pipeline)
    def hv_drop(gdf):
        c = gdf[gdf["cane_pred"].fillna(False).astype(bool)]
        dated = c["harvest_date"].notna()
        return {
            "median_dated_db": round(float(c.loc[dated, "hv_drop_db"].median()), 2),
            "median_all_cane_db": round(float(c["hv_drop_db"].median()), 2),
            "dated_segments": int(dated.sum()),
        }

    morelos = load_site("morelos", register=False)
    hv = {
        site.name: hv_drop(g),
        "morelos_area_map": hv_drop(gpd.read_file(morelos.path("harvest_geojson"))),
        "morelos_labelled_cane_median_db": MORELOS_LABELLED_HV_DROP_DB,
    }

    s_mb = g.set_index("seg_id")["mb_class"]
    s["mb_class"] = s["seg_id"].map(s_mb)

    cols = [
        "seg_id",
        "area_ha",
        "p_cane",
        "p_cane_radar",
        "gate_confidence",
        "status",
        "confidence",
        "harvest_date",
        "harvest_after",
        "n_clear",
        "ndvi_start",
        "ndvi_end",
        "err_days",
        "mb_class",
    ]
    s[cols].sort_values(["harvest_date", "seg_id"]).to_csv(
        out_dir / "validation_sample.csv", index=False, date_format="%Y-%m-%d", float_format="%.3f"
    )
    res = {
        "seed": args.seed,
        "first_radar_acquisition": f"{first_radar:%Y-%m-%d}",
        "first_datable_harvest_after": f"{first_datable:%Y-%m-%d}",
        "acquisitions": {o: [f"{d:%Y-%m-%d}" for d in v] for o, v in acq.items()},
        "error_table": tab.to_dict("records"),
        "sample_categories": cats,
        "week_corr": corr,
        "date_distribution_segments": dist,
        "date_distribution_area": dist_ha,
        "dated_share_in_season": round(in_season, 3),
        "dated_segments": int(len(hd)),
        "by_month": month.astype({"month": str}).to_dict("records"),
        "date_uncertainty": unc.to_dict("records"),
        "mapbiomas": mb_tab.to_dict("records"),
        "mapbiomas_of_gated": gated_mb.to_dict("records"),
        "mapbiomas_by_gate_confidence": by_conf.to_dict("records"),
        "hv_drop": hv,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "validation.json").write_text(json.dumps(res, indent=2, default=str))
    lines = [
        f"# Exercise 5: transfer validation, site {site.name}",
        "",
        "Generated by `scripts/16_transfer_validation.py`; see `transfer.md` for the reading.",
        "",
        f"## Detector dates vs independent NDVI windows (sample {N_DATED} + {N_UNDATED}, "
        f"seed {args.seed})",
        "",
        md_table(tab),
        "",
        *[f"* {k}: {v}" for k, v in cats.items()],
        f"* first radar acquisition {first_radar:%Y-%m-%d}; first date the 3-look baseline "
        f"rule can emit (harvest_after): {first_datable:%Y-%m-%d}",
        "",
        "## Weekly harvests vs VIIRS fires (not meaningful here: green mechanized harvest)",
        "",
        md_table(
            pd.DataFrame(
                [
                    {"fires": k, **{f"r, fires {w:+d} wk": v[w] for w in (0, 1)}}
                    for k, v in corr.items()
                ]
            ),
            "{:.2f}",
        ),
        "",
        "## Detected harvest dates",
        "",
        f"Segments: p10 {dist['p10']}, median {dist['p50']}, p90 {dist['p90']} "
        f"({len(hd)} dated segments). Area-weighted: p10 {dist_ha['p10']}, median "
        f"{dist_ha['p50']}, p90 {dist_ha['p90']}. Inside the season window "
        f"{site.season[0]:%Y-%m-%d} .. {site.season[1] - pd.Timedelta(days=1):%Y-%m-%d}: "
        f"{in_season:.1%} of dated segments.",
        "",
        md_table(month.astype({"month": str}), "{:.0f}"),
        "",
        "Date uncertainty (`date_uncertainty_days`: half the gap between the last standing and "
        "the first harvested acquisition):",
        "",
        md_table(unc, "{:g}"),
        "",
        "## Gate vs MapBiomas 2024 (majority class per segment)",
        "",
        md_table(mb_tab),
        "",
        md_table(gated_mb),
        "",
        md_table(by_conf, "{:.2f}"),
        "",
        "## Largest HV drop at harvest (hv_drop_db, median)",
        "",
        f"* {site.name}, dated cane segments: {hv[site.name]['median_dated_db']} dB "
        f"({hv[site.name]['dated_segments']} segments); all gated cane: "
        f"{hv[site.name]['median_all_cane_db']} dB",
        f"* Morelos area map, dated cane segments: {hv['morelos_area_map']['median_dated_db']} dB "
        f"({hv['morelos_area_map']['dated_segments']} segments); all gated cane: "
        f"{hv['morelos_area_map']['median_all_cane_db']} dB",
        f"* Morelos labelled cane fields (exercise 4c): {MORELOS_LABELLED_HV_DROP_DB} dB",
        "",
    ]
    (out_dir / "validation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
