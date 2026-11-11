"""Exercises 4d/4e: cane gate and harvest detector over every segment of the AOI.

    python scripts/14_area_harvest_map.py [--site morelos] [--segments <path>] [--skip-check]
    python scripts/14_area_harvest_map.py --site brazil --scene-quality-only

Inputs, outputs, season, CRS and NISAR folders come from configs/<site>.yaml (exercise 5);
the file names below are Morelos's. The gate and the detector are always the models fitted
on the Morelos fields_v2 polygons, whatever the site. Step 1 runs only for a site with
reference_fields.

1. Check: the vectorized zonal statistics (src/zafra/zonal.py) on the 38 fields_v2
   polygons, against the per-field tables of exercises 4b/4c. Must agree before going on.
2. Per-segment inputs, one pass per scene over all polygons:
   * Sentinel-2 NDVI series from the local 77-date stack (data/s2/, script 12) and the bare
     share after the collapse;
   * NISAR GCOV asc + desc (HV, HH, ENL) and GUNW coherence, cached per scene in
     outputs/segments_ts/ and gathered into data/features/segments_timeseries.parquet
     (GCOV, both orbits) and data/features/segments_coherence.parquet;
   * VIIRS fires within 200 m (one spatial join).
   Scene quality (zafra.scene_quality) is checked on every GCOV acquisition and written to
   <results_dir>/scene_quality.csv; then the site's nisar.exclude_dates are dropped from
   both radar tables (zonal.drop_excluded) before anything else uses them.
   --scene-quality-only stops after the check (radar tables from the per-scene cache).
3. Cane gate (cane_gate.classify_fields), the site's gate_variant (Morelos: optical B;
   Brazil: radar A), with p_cane_radar and p_cane_optical alongside. The date confidence
   checks the delivered gate against the other one.
4. 4b detector (xgboost default, logistic alternative), trained as in script 10 on the 26
   labelled fields_v2 cane fields, applied to every gated cane segment.
   Exercise 4e: a date is only emitted after >= 3 standing same-orbit acquisitions
   (model.dates_with_baseline); otherwise status harvested_before_coverage / no_baseline.
   date_uncertainty_days: half the gap between the last standing and the first harvested
   acquisition (the date is the midpoint of that gap).
5. data/results/harvest_dates_2025_26.geojson (with status, wc_class, wc_disagrees),
   results/4e/harvest_map.png, results/4e/harvest_vs_fires_weekly.png (all AOI fires and
   fires within 200 m of gated cane), results/4e/weekly.csv, results/4e/area_summary.json.
   Per-segment features are reused from data/features/segments_cane_features.parquet when
   it holds the same segments (--force rebuilds).
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
from zafra import scene_quality as SQ
from zafra.detect import cane_gate as G
from zafra.detect import model as M
from zafra.paths import FEATURES_DIR, FIELDS_DIR
from zafra.site import add_site_argument, load_site

# The detector's training fields (Morelos), independent of --site.
TRAIN_FIELDS = FIELDS_DIR / "fields_v2.geojson"
CHUNK = 500
DETECTOR = {"xgb": "p_xgb", "logistic": "p_logistic"}  # first = default
WC_DISAGREE = {10: "tree cover", 30: "grassland"}  # WorldCover classes flagged when gated cane


def log(msg):
    print(msg, flush=True)


def configure(site):
    """Module settings from a site config (see zafra.site)."""
    global SITE, ZAFRA, SEGMENTS, REF, STACK, ASC_DIR, DESC_DIR, GUNW_DIR, TS_CACHE, TS_OUT
    global COH_OUT, FEATS_OUT, OUT_GEOJSON, OUT_DIR, AOI, SEASON, CRS, FIRMS_DIR
    SITE = site
    ZAFRA = site.zafra
    SEGMENTS = site.path("segments")
    REF = site.reference_fields
    STACK = site.path("ndvi_stack")
    ASC_DIR, DESC_DIR, GUNW_DIR = (site.nisar[r].dir for r in ("gcov_asc", "gcov_desc", "gunw_asc"))
    TS_CACHE = site.path("ts_cache")
    TS_OUT, COH_OUT = site.path("timeseries"), site.path("coherence")
    FEATS_OUT = site.path("cane_features")
    OUT_GEOJSON = site.path("harvest_geojson")
    OUT_DIR = site.path("results_dir")
    AOI = site.firms_bbox
    SEASON = site.season
    CRS = site.metric_crs
    FIRMS_DIR = site.firms_dir


# --- inputs -------------------------------------------------------------------------------


def radar_tables(gdf, quality_out=None):
    """(acq for cane_features, gcov frames for features.assemble, enl, coh).

    With `quality_out`, the scene-quality table of all acquisitions is written there first.
    The site's excluded acquisitions are always dropped.
    """
    gc, coh = zonal.radar_timeseries(
        gdf, ASC_DIR, DESC_DIR, GUNW_DIR, TS_CACHE, log=lambda m: None, crs=CRS
    )
    if quality_out is not None:
        scene_quality_report(gc, quality_out)
    gc, coh, dropped = zonal.drop_excluded(gc, coh, SITE)
    if any(dropped.values()):
        log(f"excluded acquisitions {[e.date for e in SITE.nisar_exclude]}: dropped {dropped}")
    gc["date"] = pd.to_datetime(gc["date"])
    acq = gc.rename(columns={"HVHV_db": "hv_db", "HHHH_db": "hh_db", "ratio_db": "cross_ratio_db"})
    frames = [gc[gc["orbit"] == o] for o in ("asc", "desc")]
    enl = gc[[*FT.KEYS, "enl"]]
    return acq, frames, enl, coh


def scene_quality_report(gc, out) -> pd.DataFrame:
    qc = SQ.scene_quality(gc)
    excluded = {o: SITE.excluded_dates(f"gcov_{o}") for o in ("asc", "desc")}
    qc["excluded"] = [
        d.date().isoformat() in excluded[o] for o, d in zip(qc["orbit"], qc["date"], strict=True)
    ]
    qc.to_csv(out, index=False, date_format="%Y-%m-%d", float_format="%.3f")
    for r in qc[qc["flagged"]].itertuples():
        state = "excluded in the site config" if r.excluded else "NOT excluded: look at it"
        log(f"scene quality: {r.orbit} {r.date.date()} flagged ({r.reason}), {state}")
    log(f"scene quality: {int(qc['flagged'].sum())} of {len(qc)} acquisitions flagged -> {out}")
    return qc


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
    fields = gpd.read_file(TRAIN_FIELDS)
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


def check_column() -> str:
    """The independent gate the delivered one is checked against: the radar-only gate for
    an optical (or combined) delivered gate, the optical gate for a radar-only one."""
    v = SITE.gate_variant
    return "p_cane_optical" if G.USES_RADAR[v] and not G.USES_OPTICAL[v] else "p_cane_radar"


def harvest_confidence(gate_conf, p_check, detected) -> np.ndarray:
    """high: gate sure it is cane and the other gate (check_column) agrees; low: gate unsure
    and the other gate disagrees; medium otherwise; none: no sustained detection. main()
    then sets "uncertain" on undated cane segments that WorldCover maps as trees or
    grassland (wc_disagrees)."""
    sure = np.asarray(gate_conf) == "high"
    radar = np.asarray(p_check, dtype="float64") >= G.THRESHOLD
    out = np.where(sure & radar, "high", np.where(~sure & ~radar, "low", "medium"))
    return np.where(np.asarray(detected, dtype=bool), out, "none")


def window_counts(win, out, detected, hd, f_all, f_cane) -> dict:
    """Fires, harvests and harvested ha with a date in [win[0], win[1])."""
    d = pd.to_datetime(out.loc[detected, "harvest_date"])
    return {
        "fires_all": int(f_all.between(*win, inclusive="left").sum()),
        "fires_cane": int(f_cane.between(*win, inclusive="left").sum()),
        "harvests": int(hd.between(*win, inclusive="left").sum()),
        "harvest_ha": float(out.loc[detected].loc[(d >= win[0]) & (d < win[1]), "area_ha"].sum()),
        "harvests_4d_rule": int(
            pd.to_datetime(out["harvest_date_no_baseline_rule"])
            .between(*win, inclusive="left")
            .sum()
        ),
    }


# --- main ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_site_argument(ap)
    ap.add_argument("--segments", help="default: the site's paths.segments")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute the per-segment features")
    ap.add_argument(
        "--scene-quality-only", action="store_true", help="scene-quality check only, then stop"
    )
    args = ap.parse_args()
    configure(load_site(args.site))
    args.segments = args.segments or str(SEGMENTS)
    for d in (OUT_DIR, OUT_GEOJSON.parent, TS_OUT.parent, FEATS_OUT.parent):
        d.mkdir(parents=True, exist_ok=True)
    rt = {}

    seg = gpd.read_file(args.segments)
    seg["id"] = seg["seg_id"]
    log(f"{len(seg)} segments from {args.segments}")
    quality_out = OUT_DIR / "scene_quality.csv"
    if args.scene_quality_only:
        radar_tables(seg, quality_out)
        return 0

    check = None if args.skip_check or REF is None else check_against_fields_v2()

    # step 4: inputs + gate
    t0 = time.perf_counter()
    t = time.perf_counter()
    acq, frames, enl, coh = radar_tables(seg, quality_out)
    rt["step5_radar_zonal_s"] = time.perf_counter() - t
    log(f"radar zonal, one pass per scene: {rt['step5_radar_zonal_s']:.0f} s")
    acq.to_parquet(TS_OUT, index=False)
    coh.to_parquet(COH_OUT, index=False)
    fire_table = F.fires_per_field_fast(F.load_firms(FIRMS_DIR), seg, CF.FIRE_BUFFER_M, crs=CRS)
    t = time.perf_counter()
    feats = pd.read_parquet(FEATS_OUT) if FEATS_OUT.exists() and not args.force else None
    if feats is not None and set(feats["field_id"]) == set(seg["id"]):
        log(f"per-segment features reused from {FEATS_OUT.name} (--force to rebuild)")
        rt["step4_features_cached"] = True
    else:
        ndvi, bare = optical_tables(seg)
        feats = chunked_table(seg["id"], acq, coh, ndvi, fire_table, bare)
        feats.to_parquet(FEATS_OUT, index=False)
    gate = G.classify_fields(seg, ZAFRA, SITE.gate_variant, features=feats)
    rt["step4_features_gate_s"] = time.perf_counter() - t
    log(f"features + gate: {rt['step4_features_gate_s']:.0f} s")

    # step 5: detector on the gated cane segments
    t = time.perf_counter()
    cane_ids = set(gate.loc[gate["cane_pred"].fillna(False).astype(bool), "id"])
    acq_times = pd.concat(
        [
            FT.acquisition_times(d, o).pipe(
                lambda t, o=o: t[
                    ~t["date"].dt.date.astype(str).isin(SITE.excluded_dates(f"gcov_{o}"))
                ]
            )
            for o, d in (("asc", ASC_DIR), ("desc", DESC_DIR))
        ],
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
        baseline_months=SITE.baseline_months,
    )
    models, n_trees = train_detector()
    pred = M.predict_rows(models, rows)
    # Dates with the baseline requirement (4e), and without it (4d) for the comparison.
    dates = {
        m: M.dates_with_baseline(pred, col).set_index("field_id") for m, col in DETECTOR.items()
    }
    dates_4d = M.dates_from_probabilities(pred, DETECTOR["xgb"]).set_index("field_id")
    dates_any = M.dates_with_baseline(pred, DETECTOR["xgb"], own_orbit=False).set_index("field_id")
    rt["step5_detector_s"] = time.perf_counter() - t
    log(f"detector on {len(cane_ids)} cane segments: {rt['step5_detector_s']:.0f} s")

    # output table
    out = gate[["seg_id", "source", "area_ha", "geometry"]].copy()
    for c in ("p_cane", "p_cane_radar", "p_cane_optical", "cane_pred", "confidence"):
        out[c] = gate[c].to_numpy()
    out = out.rename(columns={"confidence": "gate_confidence"})
    f = feats.set_index("field_id").reindex(out["seg_id"])
    out["hv_drop_db"] = f["hv_max_single_drop_db"].to_numpy()
    out["coh_min"] = f["coh_min"].to_numpy()
    main_m = next(iter(DETECTOR))
    d = dates[main_m].reindex(out["seg_id"])
    alt = dates["logistic"].reindex(out["seg_id"])
    old = dates_4d.reindex(out["seg_id"])
    is_cane = out["cane_pred"].fillna(False).astype(bool).to_numpy()
    detected = d["detected"].fillna(False).astype(bool).to_numpy() & is_cane
    detected_4d = old["detected"].fillna(False).astype(bool).to_numpy() & is_cane
    ymd = {
        k: pd.to_datetime(d[k]).dt.strftime("%Y-%m-%d").to_numpy() for k in ("date", "date_after")
    }
    out["harvest_date"] = np.where(detected, ymd["date"], None)
    out["harvest_after"] = np.where(detected, ymd["date_after"], None)
    gap = (
        pd.to_datetime(d["date_after"]).dt.normalize()
        - pd.to_datetime(d["date_before"]).dt.normalize()
    )
    out["date_uncertainty_days"] = np.where(detected, gap.dt.days.to_numpy() / 2, np.nan)
    out["status"] = np.where(is_cane, d["status"].fillna(M.STATUS_NOT_DETECTED), "not_cane")
    out["method"] = np.where(detected, main_m, None)
    alt_ok = is_cane & alt["detected"].fillna(False).astype(bool).to_numpy()
    out["harvest_date_logistic"] = np.where(
        alt_ok, pd.to_datetime(alt["date"]).dt.strftime("%Y-%m-%d"), None
    )
    out["harvest_date_no_baseline_rule"] = np.where(
        detected_4d, pd.to_datetime(old["date"]).dt.strftime("%Y-%m-%d"), None
    )
    out["confidence"] = harvest_confidence(out["gate_confidence"], out[check_column()], detected)
    fr = pred.set_index(["field_id", "date"])["fire_7d"]
    keys = list(zip(out["seg_id"], pd.to_datetime(out["harvest_after"]), strict=True))
    out["fires_7d_before"] = [
        int(fr.get(k, np.nan)) if detected[i] and k in fr.index else None
        for i, k in enumerate(keys)
    ]
    wc = seg.set_index("seg_id")["wc_class"].reindex(out["seg_id"]).to_numpy()
    out["wc_class"] = pd.array(wc, dtype="Int64")
    out["wc_disagrees"] = is_cane & np.isin(wc, list(WC_DISAGREE))
    # Gate says cane, WorldCover says trees / grassland, and no harvest was dated: the cane
    # call itself is in doubt. Kept, but marked "uncertain" rather than "none".
    out.loc[out["wc_disagrees"] & ~detected, "confidence"] = "uncertain"
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=seg.crs)
    out.to_file(OUT_GEOJSON, driver="GeoJSON")

    # fires: all in the AOI, and only those within 200 m of a gated cane segment
    fires = F.load_firms(FIRMS_DIR)
    fires = fires.cx[AOI[0] : AOI[2], AOI[1] : AOI[3]]
    ft = fires["acq_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    fires = fires[((ft >= SEASON[0]) & (ft < SEASON[1])).to_numpy()]
    cane_zone = out[is_cane].to_crs(CRS).buffer(CF.FIRE_BUFFER_M).union_all()
    in_cane = fires.to_crs(CRS).within(cane_zone).to_numpy()
    f_all = fires["acq_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    f_cane = f_all[in_cane]

    # figures and weekly series
    plots.harvest_map(
        out,
        OUT_DIR / "harvest_map.png",
        title=f"Zafra {ZAFRA}: estimated harvest date per segment (gate {SITE.gate_variant} "
        "+ 4b xgboost, 3-look baseline)",
        crs=CRS,
    )
    hd = pd.to_datetime(out.loc[detected, "harvest_date"])
    end = SEASON[1] - pd.Timedelta(days=1)
    weeks = plots.harvest_vs_fire_weeks(
        hd,
        f_all,
        SEASON[0],
        end,
        OUT_DIR / "harvest_vs_fires_weekly.png",
        title=f"Zafra {ZAFRA}, AOI: detected harvests vs VIIRS fires per week",
        fires_cane=f_cane,
    )
    ha = out.loc[detected, ["harvest_date", "area_ha"]].assign(
        week=lambda x: pd.to_datetime(x["harvest_date"]).dt.to_period("W-SUN").dt.start_time
    )
    weeks["harvest_ha"] = ha.groupby("week")["area_ha"].sum().reindex(weeks.index).fillna(0)
    weeks["harvests_no_baseline_rule"] = plots.weekly_counts(
        out.loc[detected_4d, "harvest_date_no_baseline_rule"], SEASON[0], end
    )
    weekly = weeks.rename(columns={"fires": "fires_all"})[
        ["harvests", "harvest_ha", "fires_all", "fires_cane", "harvests_no_baseline_rule"]
    ]
    weekly.round(2).to_csv(OUT_DIR / "weekly.csv")
    corr = {
        col: plots.week_correlation(weekly["harvests"], weekly[col])
        for col in ("fires_all", "fires_cane")
    }
    corr_4d = plots.week_correlation(weekly["harvests_no_baseline_rule"], weekly["fires_all"])
    rt["step4_5_wall_s"] = time.perf_counter() - t0

    # Exercise-4e diagnostic windows (site zafra.check_windows; Morelos only).
    april = SITE.check_windows.get("april")
    dec = SITE.check_windows.get("dated_dec1_20")
    cane = out[is_cane]
    status = out.loc[is_cane, "status"]
    summary = {
        "segments": len(out),
        "predicted_cane": int(is_cane.sum()),
        "cane_ha": float(cane["area_ha"].sum()),
        "status_cane": status.value_counts().to_dict(),
        "status_cane_ha": cane.groupby("status")["area_ha"].sum().round(1).to_dict(),
        "cane_with_harvest_date": int(detected.sum()),
        "cane_with_harvest_date_4d_rule": int(detected_4d.sum()),
        "cane_harvest_ha": float(out.loc[detected, "area_ha"].sum()),
        "harvest_confidence": out.loc[detected, "confidence"].value_counts().to_dict(),
        "confidence_cane_all": out.loc[is_cane, "confidence"].value_counts().to_dict(),
        "logistic_detected": int(alt_ok.sum()),
        "xgb_n_trees": n_trees,
        "harvest_date_quantiles": {
            q: str(hd.quantile(q).date()) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        if len(hd)
        else {},
        "any_orbit_variant": {
            "status_cane": dates_any.reindex(out.loc[is_cane, "seg_id"])["status"]
            .value_counts()
            .to_dict(),
            **(
                {
                    "dated_dec1_20": int(
                        pd.to_datetime(dates_any["date"])
                        .reindex(out.loc[is_cane, "seg_id"])
                        .between(*dec)
                        .sum()
                    )
                }
                if dec
                else {}
            ),
        },
        **(
            {
                "dated_dec1_20": {
                    "4d_rule": int(
                        pd.to_datetime(out["harvest_date_no_baseline_rule"]).between(*dec).sum()
                    ),
                    "baseline_rule": int(pd.to_datetime(out["harvest_date"]).between(*dec).sum()),
                }
            }
            if dec
            else {}
        ),
        "wc_disagrees": int(out["wc_disagrees"].sum()),
        "wc_disagrees_ha": float(out.loc[out["wc_disagrees"], "area_ha"].sum()),
        "fires_all_season": int(len(f_all)),
        "fires_cane_season": int(len(f_cane)),
        **({"april": window_counts(april, out, detected, hd, f_all, f_cane)} if april else {}),
        "week_corr": {k: v.round(3).to_dict("records") for k, v in corr.items()},
        "week_corr_4d_rule_fires_all": corr_4d.round(3).to_dict("records"),
        "check_vs_fields_v2": check,
        "runtimes_s": {k: round(v, 1) if isinstance(v, float) else v for k, v in rt.items()},
    }
    (OUT_DIR / "area_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
