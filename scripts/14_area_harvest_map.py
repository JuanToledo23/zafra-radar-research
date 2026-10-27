"""Exercise 4d, steps 4-5: cane gate and harvest detector over every segment of the AOI.

    python scripts/14_area_harvest_map.py [--segments data/fields/segments.geojson]
                                          [--skip-check]

1. Check: the vectorized zonal statistics (src/zafra/zonal.py) on the 38 fields_v2
   polygons, against the per-field tables of exercises 4b/4c. Must agree before going on.
2. Per-segment inputs, one pass per scene over all polygons:
   * Sentinel-2 NDVI series from the local 77-date stack (data/s2/, script 12) and the bare
     share after the collapse;
   * NISAR GCOV asc + desc (HV, HH, ENL) and GUNW coherence, cached per scene in
     outputs/segments_ts/ and gathered into data/features/segments_timeseries.parquet
     (GCOV, both orbits) and data/features/segments_coherence.parquet;
   * VIIRS fires within 200 m (one spatial join).
3. Cane gate (cane_gate.classify_fields, default optical variant B, p_cane_radar alongside).
4. 4b detector (xgboost default, logistic alternative), trained as in script 10 on the 26
   labelled fields_v2 cane fields, applied to every gated cane segment.
5. data/results/harvest_dates_2025_26.geojson, results/4d/harvest_map.png,
   results/4d/harvest_vs_fires_weekly.png, results/4d/area_summary.{json,md}.
"""

import argparse
import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from zafra import cane_features as CF
from zafra import features as FT
from zafra import fires as F
from zafra import plots, zonal
from zafra.detect import cane_gate as G
from zafra.detect import model as M
from zafra.paths import (
    DATA_DIR,
    FEATURES_DIR,
    FIELDS_DIR,
    FIRMS_DIR,
    NISAR_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
)

ZAFRA = "2025-26"
SEGMENTS = FIELDS_DIR / "segments.geojson"
REF = FIELDS_DIR / "fields_v2.geojson"
STACK = DATA_DIR / "s2" / "s2_ndvi_stack_2025_26.tif"
ASC_DIR = NISAR_DIR
DESC_DIR = NISAR_DIR / "GCOV_DESCENDING_t113f79"
GUNW_DIR = NISAR_DIR / "GUNW_ASCENDING_t19f11"
TS_CACHE = OUTPUTS_DIR / "segments_ts"
TS_OUT = FEATURES_DIR / "segments_timeseries.parquet"
COH_OUT = FEATURES_DIR / "segments_coherence.parquet"
FEATS_OUT = FEATURES_DIR / "segments_cane_features.parquet"
OUT_GEOJSON = DATA_DIR / "results" / "harvest_dates_2025_26.geojson"
OUT_DIR = RESULTS_DIR / "4d"
AOI = (-99.35, 18.50, -99.05, 18.75)
SEASON = (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-06-01"))
CHUNK = 500
DETECTOR = {"xgb": "p_xgb", "logistic": "p_logistic"}  # first = default


def log(msg):
    print(msg, flush=True)


# --- inputs -------------------------------------------------------------------------------


def radar_tables(gdf):
    """(acq for cane_features, gcov frames for features.assemble, enl, coh)."""
    gc, coh = zonal.radar_timeseries(gdf, ASC_DIR, DESC_DIR, GUNW_DIR, TS_CACHE, log=lambda m: None)
    gc["date"] = pd.to_datetime(gc["date"])
    acq = gc.rename(columns={"HVHV_db": "hv_db", "HHHH_db": "hh_db", "ratio_db": "cross_ratio_db"})
    frames = [gc[gc["orbit"] == o] for o in ("asc", "desc")]
    enl = gc[[*FT.KEYS, "enl"]]
    return acq, frames, enl, coh


def optical_tables(gdf):
    ndvi = zonal.ndvi_series_from_stack(gdf, STACK)
    t = chunked_table(gdf["id"], None, None, ndvi, None, None)
    req = t.dropna(subset=["collapse_low_image_id"])
    bare = zonal.bare_shares_from_stack(
        gdf, STACK, list(zip(req["field_id"], req["collapse_low_image_id"], strict=True))
    )
    return ndvi, bare


def chunked_table(ids, acq, coh, ndvi, fire_table, bare):
    """CF.build_table in chunks: it filters every input once per field, which is quadratic
    for ~10^4 polygons; pre-filtering each chunk keeps it linear."""
    ids = list(ids)
    parts = []
    for i in range(0, len(ids), CHUNK):
        sub = set(ids[i : i + CHUNK])

        def pick(df, col="field_id", sub=sub):
            return None if df is None else df[df[col].isin(sub)]

        parts.append(
            CF.build_table(
                ids[i : i + CHUNK],
                [ZAFRA],
                pick(acq),
                pick(coh),
                pick(ndvi, "id"),
                pick(fire_table),
                bare,
            )
        )
    return pd.concat(parts, ignore_index=True)


# --- 1. check against the per-field path ----------------------------------------------------


def check_against_fields_v2():
    """The vectorized path on the 38 reference polygons vs the exercise-4b/4c tables."""
    ref = gpd.read_file(REF)
    acq, _, _, coh = radar_tables(ref)
    old = pd.read_parquet(FEATURES_DIR / "fields_v2_features.parquet")
    m = old.merge(acq, on=FT.KEYS, suffixes=("_old", ""))
    d_hv = (m["hv_db_old"] - m["hv_db"]).abs()
    d_enl = (m["enl_old"] - m["enl"]).abs() / m["enl_old"]
    cm = old.dropna(subset=["coherence_mean"]).merge(
        coh.rename(columns={"sec_date": "date"}).assign(orbit="asc"), on=FT.KEYS
    )
    d_coh = (cm["coherence_mean"] - cm["coh_mean"]).abs()
    ndvi, bare = optical_tables(ref)
    new = chunked_table(ref["id"], acq, coh, ndvi, None, bare)
    oldf = pd.read_parquet(FEATURES_DIR / "fields_v2_cane_features.parquet")
    oldf = oldf[oldf["zafra"] == ZAFRA].set_index("field_id").loc[new["field_id"]]
    newf = new.set_index("field_id")
    cols = ["ndvi_max", "ndvi_min", "ndvi_longest_high_run_days", "ndvi_n_collapses"]
    d_opt = {c: float((oldf[c] - newf[c]).abs().max()) for c in cols}
    gate_old = G.classify_fields(ref, ZAFRA, features=oldf.reset_index().assign(zafra=ZAFRA))
    gate_new = G.classify_fields(ref, ZAFRA, features=new)
    agree = int((gate_old["cane_pred"] == gate_new["cane_pred"]).sum())
    out = {
        "gcov_rows": len(m),
        "hv_db_max_abs_diff": float(d_hv.max()),
        "enl_max_rel_diff": float(d_enl.max()),
        "coh_rows": len(cm),
        "coh_mean_max_abs_diff": float(d_coh.max()),
        "optical_max_abs_diff": d_opt,
        "gate_decisions_agree": f"{agree}/{len(ref)}",
        "p_cane_max_abs_diff": float((gate_old["p_cane"] - gate_new["p_cane"]).abs().max()),
    }
    log(f"check vs fields_v2: {json.dumps(out)}")
    return out


# --- 4. detector --------------------------------------------------------------------------


def train_detector():
    """The script-10 models: all labelled rows of the 26 cane fields, n_trees from CV."""
    fields = gpd.read_file(REF)
    windows = fields[["id", "harvest_window_start", "harvest_window_end"]].rename(
        columns={"id": "field_id"}
    )
    for c in ("harvest_window_start", "harvest_window_end"):
        windows[c] = pd.to_datetime(windows[c])
    cane = fields[(fields["label"] == "cana") & fields["harvest_window_start"].notna()]
    feats = pd.read_parquet(FEATURES_DIR / "fields_v2_features.parquet")
    rows = M.label_states(feats[feats["field_id"].isin(cane["id"])], windows, keep_window=True)
    _, info = M.cross_validate(rows)
    n_trees = int(np.median(info["best_iterations"]))
    return M.fit_full(rows, n_trees), n_trees


def harvest_confidence(gate_conf, p_radar, detected) -> np.ndarray:
    """high: gate sure it is cane and radar-only gate agrees; low: gate unsure and radar
    disagrees; medium otherwise; none: no sustained detection."""
    sure = np.asarray(gate_conf) == "high"
    radar = np.asarray(p_radar, dtype="float64") >= G.THRESHOLD
    out = np.where(sure & radar, "high", np.where(~sure & ~radar, "low", "medium"))
    return np.where(np.asarray(detected, dtype=bool), out, "none")


# --- main ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--segments", default=str(SEGMENTS))
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    rt = {}

    check = None if args.skip_check else check_against_fields_v2()

    seg = gpd.read_file(args.segments)
    seg["id"] = seg["seg_id"]
    log(f"{len(seg)} segments from {args.segments}")

    # step 4: inputs + gate
    t0 = time.perf_counter()
    t = time.perf_counter()
    ndvi, bare = optical_tables(seg)
    rt["step4_optical_s"] = time.perf_counter() - t
    log(f"optical series: {rt['step4_optical_s']:.0f} s ({len(ndvi)} rows)")
    t = time.perf_counter()
    acq, frames, enl, coh = radar_tables(seg)
    rt["step5_radar_zonal_s"] = time.perf_counter() - t
    log(f"radar zonal, one pass per scene: {rt['step5_radar_zonal_s']:.0f} s")
    acq.to_parquet(TS_OUT, index=False)
    coh.to_parquet(COH_OUT, index=False)
    t = time.perf_counter()
    fire_table = F.fires_per_field_fast(F.load_firms(FIRMS_DIR), seg, CF.FIRE_BUFFER_M)
    feats = chunked_table(seg["id"], acq, coh, ndvi, fire_table, bare)
    feats.to_parquet(FEATS_OUT, index=False)
    gate = G.classify_fields(seg, ZAFRA, features=feats)
    rt["step4_features_gate_s"] = time.perf_counter() - t
    rt["step4_total_s"] = rt["step4_optical_s"] + rt["step4_features_gate_s"]
    log(f"features + gate: {rt['step4_features_gate_s']:.0f} s")

    # step 5: detector on the gated cane segments
    t = time.perf_counter()
    cane_ids = set(gate.loc[gate["cane_pred"].fillna(False).astype(bool), "id"])
    acq_times = pd.concat(
        [FT.acquisition_times(d, o) for o, d in (("asc", ASC_DIR), ("desc", DESC_DIR))],
        ignore_index=True,
    )
    sub = [f[f["field_id"].isin(cane_ids)] for f in frames]
    rows = FT.assemble(
        sub,
        coh[coh["field_id"].isin(cane_ids)],
        enl[enl["field_id"].isin(cane_ids)],
        acq_times,
        fire_table[fire_table["field_id"].isin(cane_ids)],
        None,
    )
    models, n_trees = train_detector()
    pred = M.predict_rows(models, rows)
    dates = {
        m: M.dates_from_probabilities(pred, col).set_index("field_id")
        for m, col in DETECTOR.items()
    }
    rt["step5_detector_s"] = time.perf_counter() - t
    rt["step5_total_s"] = rt["step5_radar_zonal_s"] + rt["step5_detector_s"]
    log(f"detector on {len(cane_ids)} cane segments: {rt['step5_detector_s']:.0f} s")

    # output table
    out = gate[["seg_id", "source", "area_ha", "geometry"]].copy()
    for c in ("p_cane", "p_cane_radar", "cane_pred", "confidence"):
        out[c] = gate[c].to_numpy()
    out = out.rename(columns={"confidence": "gate_confidence"})
    f = feats.set_index("field_id").reindex(out["seg_id"])
    out["hv_drop_db"] = f["hv_max_single_drop_db"].to_numpy()
    out["coh_min"] = f["coh_min"].to_numpy()
    main_m = next(iter(DETECTOR))
    d = dates[main_m].reindex(out["seg_id"])
    alt = dates["logistic"].reindex(out["seg_id"])
    is_cane = out["cane_pred"].fillna(False).astype(bool).to_numpy()
    detected = d["detected"].fillna(False).astype(bool).to_numpy() & is_cane
    out["harvest_date"] = np.where(detected, d["date"].dt.strftime("%Y-%m-%d"), None)
    out["harvest_after"] = np.where(detected, d["date_after"].dt.strftime("%Y-%m-%d"), None)
    out["method"] = np.where(detected, main_m, None)
    out["harvest_date_logistic"] = np.where(
        is_cane & alt["detected"].fillna(False).astype(bool).to_numpy(),
        alt["date"].dt.strftime("%Y-%m-%d"),
        None,
    )
    out["confidence"] = harvest_confidence(out["gate_confidence"], out["p_cane_radar"], detected)
    fr = pred.set_index(["field_id", "date"])["fire_7d"]
    keys = list(zip(out["seg_id"], pd.to_datetime(out["harvest_after"]), strict=True))
    out["fires_7d_before"] = [
        int(fr.get(k, np.nan)) if detected[i] and k in fr.index else None
        for i, k in enumerate(keys)
    ]
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=seg.crs)
    out.to_file(OUT_GEOJSON, driver="GeoJSON")

    # figures
    plots.harvest_map(
        out,
        OUT_DIR / "harvest_map.png",
        title=f"Zafra {ZAFRA}: estimated harvest date per segment (gate B + 4b xgboost)",
    )
    fires = F.load_firms(FIRMS_DIR)
    fires = fires.cx[AOI[0] : AOI[2], AOI[1] : AOI[3]]
    ft = fires["acq_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    ft = ft[(ft >= SEASON[0]) & (ft < SEASON[1])]
    hd = pd.to_datetime(out.loc[detected, "harvest_date"])
    weeks = plots.harvest_vs_fire_weeks(
        hd,
        ft,
        SEASON[0],
        SEASON[1] - pd.Timedelta(days=1),
        OUT_DIR / "harvest_vs_fires_weekly.png",
        title=f"Zafra {ZAFRA}, AOI: detected harvests vs VIIRS fires per week",
    )
    weeks.to_csv(OUT_DIR / "weekly_harvest_fire.csv")
    corr = plots.week_correlation(weeks)
    rt["step4_5_wall_s"] = time.perf_counter() - t0

    cane = out[is_cane]
    summary = {
        "segments": len(out),
        "with_optical": int(feats["has_optical"].sum()),
        "with_radar": int(feats["has_radar"].sum()),
        "predicted_cane": int(is_cane.sum()),
        "cane_ha": float(cane["area_ha"].sum()),
        "all_segments_ha": float(out["area_ha"].sum()),
        "gate_confidence": out["gate_confidence"].value_counts().to_dict(),
        "gate_confidence_cane": cane["gate_confidence"].value_counts().to_dict(),
        "radar_agrees_on_cane": int((cane["p_cane_radar"] >= G.THRESHOLD).sum()),
        "cane_with_harvest_date": int(detected.sum()),
        "cane_harvest_ha": float(out.loc[detected, "area_ha"].sum()),
        "harvest_confidence": out.loc[detected, "confidence"].value_counts().to_dict(),
        "logistic_detected": int(out["harvest_date_logistic"].notna().sum()),
        "xgb_n_trees": n_trees,
        "harvest_date_quantiles": {
            q: str(hd.quantile(q).date()) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        if len(hd)
        else {},
        "fires_in_aoi_season": int(len(ft)),
        "week_corr": corr.round(3).to_dict("records"),
        "check_vs_fields_v2": check,
        "runtimes_s": {k: round(v, 1) for k, v in rt.items()},
    }
    (OUT_DIR / "area_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
