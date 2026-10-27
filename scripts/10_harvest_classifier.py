"""Exercise 4b: multi-signal features, trained harvest detector, comparison with exercise 4.

    python scripts/10_harvest_classifier.py [--force] [--no-rain]

1. Feature table (src/zafra/features.py), one row per (field, acquisition date, orbit):
   data/features/fields_v2_features.{parquet,csv}. NISAR series are read from the
   scripts/08_multisignal.py caches in outputs/multisignal/ (rebuilt here if missing or
   stale); fires from data/fires/fields_v2_fires.csv (scripts/05_fires_per_field.py); rain
   from GPM IMERG via Earth Engine (cached in data/features/gpm_rain_3d.csv).
2. Labels: cane fields with an NDVI harvest window; 0 before, 1 after, window rows dropped.
3. Logistic regression and xgboost, GroupKFold by field (src/zafra/detect/model.py).
4. Harvest date per field from out-of-fold predictions (sustained rule). On the same fields:
   the exercise-4 rule (ascending HV, unchanged), the ratio test on ascending only (exercise 4,
   unchanged, for reference) and the ratio test on both orbits, whose per-scene decisions go
   through the same sustained rule as the models.
5. results/4b/comparison.md, results/4b/figures/<field>.png, results/4b/*.csv.
"""

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zafra import features as FT
from zafra import fires as F
from zafra import multisignal as ms
from zafra import timeseries as ts
from zafra.detect import GcovPixelProvider, detect_ratio, detect_rule
from zafra.detect import model as M
from zafra.detect import ratio_test as RT
from zafra.paths import FEATURES_DIR, FIELDS_DIR, FIRES_DIR, NISAR_DIR, OUTPUTS_DIR, RESULTS_DIR

FIELDS = FIELDS_DIR / "fields_v2.geojson"
ASC_DIR = NISAR_DIR
DESC_DIR = NISAR_DIR / "GCOV_DESCENDING_t113f79"
GUNW_DIR = NISAR_DIR / "GUNW_ASCENDING_t19f11"
MS_DIR = OUTPUTS_DIR / "multisignal"
FIRE_TABLE = FIRES_DIR / "fields_v2_fires.csv"
RAIN_CACHE = FEATURES_DIR / "gpm_rain_3d.csv"
FEATURES_OUT = FEATURES_DIR / "fields_v2_features"
OUT_DIR = RESULTS_DIR / "4b"
FIG_DIR = OUT_DIR / "figures"

CANE = "cana"
SEASON = (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-05-31"))  # Nov-May zafra
TOL = (6, 12)  # days: half the 12-day revisit, and one revisit
LATE_DAYS = 6  # a rule date more than this after the window midpoint counts as "late"
N_DRYING = 3  # drying-cane section: the N fields where the rule is latest ...
EX6_LATE = {"p2a": "field_04"}  # ... plus exercise-6 late fields that are in fields_v2
NEAR_DAYS = 20  # rows shown around the true date in the drying-cane section

METHODS = {
    "rule": "rule (asc)",
    "ratio_asc": "ratio test (asc only)",
    "ratio": "ratio test (asc+desc)",
    "logistic": "logistic",
    "xgb": "xgboost",
}
# Methods in the per-field table, the drying-cane section and the figures; ratio_asc is only a
# reference row of the summary (it is the exercise-4 detector as it was).
FIELD_METHODS = ["rule", "ratio", "logistic", "xgb"]
# Validated categorical slots (dataviz default palette, light); line styles add a second cue.
C_ASC, C_DESC, C_COH, C_RAIN = "#2a78d6", "#eb6834", "#1baf7a", "#2a78d6"
METHOD_STYLE = {
    "rule": ("#1baf7a", (0, (1, 1.5))),
    "ratio": ("#eda100", (0, (4, 2))),
    "logistic": ("#e87ba4", (0, (6, 2, 1, 2))),
    "xgb": ("#008300", "-"),
}
INK, MUTED = "#1f1f1e", "#6b6b66"

COH_FEATS = ["coherence_mean", "coherence_min", "delta_coh_vs_baseline"]
SHARP_FEATS = ["sharpness", "slope_hv_3"]
CALENDAR_FEATS = ["rain_3d_mm", "gap_days"]
# Feature sets of the ablation: the main model (M.FEATURES), and what changes when the dropped
# groups are put back or coherence is removed.
_MAIN = list(M.FEATURES)
ABLATIONS = {
    "main (no rain, gap_days, sharpness, slope)": _MAIN,
    "main + sharpness and slope": _MAIN + SHARP_FEATS,
    "main + rain and gap_days": _MAIN + CALENDAR_FEATS,
    "main + all four (first run)": _MAIN + SHARP_FEATS + CALENDAR_FEATS,
    "main without coherence": [f for f in _MAIN if f not in COH_FEATS],
}


# --- inputs -------------------------------------------------------------------------------


def cached(path, build, ok, force):
    if path.exists() and not force:
        df = pd.read_parquet(path)
        if ok(df):
            return df
        print(f"cache {path.name} stale; rebuilding")
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_inputs(fields, force, use_rain):
    ids = set(fields["id"])
    scenes = {o: ts.list_scenes(d) for o, d in (("asc", ASC_DIR), ("desc", DESC_DIR))}
    pairs = ms.list_pairs(GUNW_DIR)

    def same(df, col, dates):
        return set(df["field_id"]) == ids and set(df[col].dt.date.astype(str)) == set(dates)

    gcov = {
        o: cached(
            MS_DIR / f"gcov_{o}.parquet",
            lambda o=o, d=d: ms.extract_gcov_series(d, FIELDS, o),
            lambda df, o=o: same(df, "date", [s["date"] for s in scenes[o]]),
            force,
        )
        for o, d in (("asc", ASC_DIR), ("desc", DESC_DIR))
    }
    coh = cached(
        MS_DIR / "coherence.parquet",
        lambda: ms.extract_coherence_series(GUNW_DIR, fields),
        lambda df: "coh_min" in df and same(df, "sec_date", [p["sec_date"] for p in pairs]),
        force,
    )
    enl = pd.concat(
        [
            cached(
                MS_DIR / f"enl_{o}.parquet",
                lambda o=o, d=d: FT.enl_table(d, FIELDS, o),
                lambda df, o=o: same(df, "date", [s["date"] for s in scenes[o]]),
                force,
            )
            for o, d in (("asc", ASC_DIR), ("desc", DESC_DIR))
        ],
        ignore_index=True,
    )
    # The ENL pass re-reads the pixels: its field mean must match the series.
    chk = pd.concat(gcov.values()).merge(enl, on=FT.KEYS)
    diff = (chk["HVHV_db"] - chk["enl_mean_db"]).abs().max()
    print(f"ENL pass vs series HV mean: max |diff| {diff:.1e} dB ({len(chk)} rows)")
    if diff > 0.01:
        raise RuntimeError("ENL pixel selection does not match the series")

    acq = pd.concat(
        [FT.acquisition_times(d, o) for o, d in (("asc", ASC_DIR), ("desc", DESC_DIR))],
        ignore_index=True,
    )
    if not FIRE_TABLE.exists():
        raise FileNotFoundError(f"{FIRE_TABLE} missing: run scripts/05_fires_per_field.py first")
    fire_table = F.load_fire_table(FIRE_TABLE)

    rain = None
    if use_rain:
        if RAIN_CACHE.exists() and not force:
            rain = pd.read_csv(RAIN_CACHE, parse_dates=["date"])
            if len(rain) != len(acq) * len(fields):
                rain = None
        if rain is None:
            rain = FT.gpm_rain(acq, fields)
            if rain is not None:
                rain.to_csv(RAIN_CACHE, index=False)
    return list(gcov.values()), coh, enl, acq, fire_table, rain


# --- exercise-4 detectors -------------------------------------------------------------------


def hv_series(gcov: pd.DataFrame) -> pd.DataFrame:
    """multisignal GCOV frame -> the exercise-3 series layout the exercise-4 detectors read."""
    return pd.DataFrame(
        {
            "field_id": gcov["field_id"],
            "date": gcov["date"],
            "pol": "HVHV",
            "mean_db": gcov["HVHV_db"],
            "mean_linear": 10 ** (gcov["HVHV_db"] / 10),
            "valid_px": gcov["valid_px"],
            "mean_looks": gcov["mean_looks"],
        }
    )


def ratio_scene_states(scenes: pd.DataFrame) -> pd.Series:
    """The ratio test's per-scene decision without its own next-scene confirmation.

    A scene counts as harvested if it is after the baseline scenes, at least SUSTAIN_DB below
    the baseline and significant (step or baseline test, as in ratio_test.scene_tests). The
    confirmation is then done by M.sustained_date on both orbits merged, as for the models.
    """
    idx = scenes.groupby("field_id").cumcount()
    return (
        (idx >= RT.BASELINE_SCENES)
        & (scenes["below_base_db"] <= -RT.SUSTAIN_DB)
        & (scenes["sig_prev"] | scenes["sig_base"])
    ).astype(int)


def run_exercise4(gcov_asc: pd.DataFrame, gcov_desc: pd.DataFrame):
    """Rule (asc), ratio test (asc only), and ratio test on both orbits.

    Returns (dates: one row per field and method, ratio-test per-scene table of both orbits).
    """
    asc = hv_series(gcov_asc)
    rule = detect_rule(asc)
    out = []
    scenes = []
    for orbit, g, d in (("asc", asc, ASC_DIR), ("desc", hv_series(gcov_desc), DESC_DIR)):
        det, sc = detect_ratio(g, GcovPixelProvider(d, FIELDS), return_scenes=True)
        sc = sc.sort_values(["field_id", "date"], ignore_index=True)
        scenes.append(sc.assign(orbit=orbit, state=ratio_scene_states(sc)))
        if orbit == "asc":
            ratio_asc = det
    for name, det in (("rule", rule), ("ratio_asc", ratio_asc)):
        d = det[["field_id", "detected", "date_before", "date_detected"]].rename(
            columns={"date_detected": "date_after"}
        )
        d["date"] = d["date_before"] + (d["date_after"] - d["date_before"]) / 2
        out.append(d.assign(method=name))
    both = pd.concat(scenes, ignore_index=True)
    out.append(
        pd.DataFrame(
            [
                {"field_id": fid, **M.sustained_date(s["date"], s["state"])}
                for fid, s in both.groupby("field_id", sort=True)
            ]
        ).assign(method="ratio")
    )
    return pd.concat(out, ignore_index=True), both


# --- evaluation -----------------------------------------------------------------------------


def errors(dates: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    w = windows.copy()
    w["mid"] = w["harvest_window_start"] + (w["harvest_window_end"] - w["harvest_window_start"]) / 2
    d = dates.merge(w, on="field_id", how="inner")
    d["err_days"] = (d["date"] - d["mid"]).dt.total_seconds() / 86400
    return d


def summary_row(e: pd.DataFrame, nc: pd.DataFrame, strict_nc) -> dict:
    """Date-error and false-alarm summary for one method. strict_nc = ids of non-cane
    fields whose NDVI hypothesis is also non-cane (p5 is 'dudoso' but hypothesised cane)."""
    a = e["err_days"].abs()
    fa = nc["detected"] & nc["date"].between(*SEASON)
    fa_strict = fa & nc["field_id"].isin(strict_nc)
    return {
        "median |error| (d)": a.median(),
        "mean error (d)": e["err_days"].mean(),
        f"within ±{TOL[0]} d": f"{(a <= TOL[0]).sum()}/{len(e)}",
        f"within ±{TOL[1]} d": f"{(a <= TOL[1]).sum()}/{len(e)}",
        "missed": int((~e["detected"].astype(bool)).sum()),
        "false alarms, non-cane fields": f"{int(fa.sum())}/{len(nc)} "
        f"({int(fa_strict.sum())}/{int(nc['field_id'].isin(strict_nc).sum())} without p5)",
    }


def summary_table(err: pd.DataFrame, nc_dates: pd.DataFrame, strict_nc) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        e = err[err["method"] == m]
        nc = nc_dates[nc_dates["method"] == m]
        rows.append(
            {
                "method": METHODS[m],
                **summary_row(e, nc, strict_nc),
            }
        )
    return pd.DataFrame(rows)


def run_ablations(cane_rows, nc_feats, windows, strict_nc) -> pd.DataFrame:
    """Same CV and date rule with feature groups removed (both models)."""
    rows = []
    for name, feats in ABLATIONS.items():
        oof, info = M.cross_validate(cane_rows, feats)
        models = M.fit_full(cane_rows, int(np.median(info["best_iterations"])), feats)
        nc = M.predict_rows(models, nc_feats, feats)
        for m, col in (("logistic", "p_logistic"), ("xgb", "p_xgb")):
            e = errors(M.dates_from_probabilities(oof, col), windows)
            n = M.dates_from_probabilities(nc, col)
            rows.append({"features": name, "model": METHODS[m], **summary_row(e, n, strict_nc)})
    return pd.DataFrame(rows)


def fmt_d(t) -> str:
    return "—" if pd.isna(t) else f"{t:%m-%d}"


def fmt_e(x) -> str:
    return "missed" if pd.isna(x) else f"{x:+.0f}"


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- figure -------------------------------------------------------------------------------


def plot_field(fid, meta, feats, coh, fires_near, method_dates, out):
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]}
    )
    ax_hv, ax_coh, ax_rain = axes
    for orbit, color, label in (
        ("asc", C_ASC, "HV ascending t19"),
        ("desc", C_DESC, "HV descending t113"),
    ):
        d = feats[feats["orbit"] == orbit].sort_values("date")
        ax_hv.plot(d["date"], d["hv_db"], color=color, lw=2, marker="o", ms=4, label=label)
    ax_hv.set_ylabel("HV gamma-0 (dB)")
    # Fires: a rug of markers along the top of the HV panel (no y scale implied).
    if len(fires_near):
        t = fires_near["acq_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        ax_hv.plot(
            t,
            np.full(len(t), 0.96),
            "v",
            ms=8,
            color="#e34948",
            transform=ax_hv.get_xaxis_transform(),
            label=f"VIIRS fire ≤ {FT.FIRE_BUFFER_M} m",
            clip_on=False,
        )

    c = coh.sort_values("ref_date")
    ax_coh.hlines(c["coh_mean"], c["ref_date"], c["sec_date"], color=C_COH, lw=2)
    ax_coh.plot(
        c["sec_date"], c["coh_mean"], "o", ms=4, color=C_COH, label="HH coherence (pair end)"
    )
    ax_coh.set_ylim(0, 1)
    ax_coh.set_ylabel("coherence\n12-day asc pair")

    r = feats.dropna(subset=["rain_3d_mm"]).sort_values("date")
    if len(r):
        ax_rain.bar(r["date"], r["rain_3d_mm"], width=2.5, color=C_RAIN, alpha=0.7)
        ax_rain.set_ylim(0, max(5, r["rain_3d_mm"].max() * 1.2))
    else:
        ax_rain.text(
            0.5,
            0.5,
            "GPM rain not available",
            transform=ax_rain.transAxes,
            ha="center",
            color=MUTED,
        )
    ax_rain.set_ylabel("GPM rain\n3 d before (mm)")
    ax_rain.set_xlabel("acquisition date (UTC)")

    handles = []
    if pd.notna(meta["harvest_window_start"]):
        for ax in axes:
            ax.axvspan(
                meta["harvest_window_start"],
                meta["harvest_window_end"] + pd.Timedelta(days=1),
                color="0.5",
                alpha=0.18,
                lw=0,
            )
        handles.append(Patch(color="0.5", alpha=0.3, label="NDVI harvest window"))
    for m, (color, ls) in METHOD_STYLE.items():
        t = method_dates.get(m, pd.NaT)
        lab = f"{METHODS[m]}: {'none' if pd.isna(t) else f'{t:%Y-%m-%d}'}"
        if pd.notna(t):
            for ax in axes:
                ax.axvline(t, color=color, ls=ls, lw=2)
        handles.append(Line2D([], [], color=color, ls=ls, lw=2, label=lab))
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_hv.legend(loc="lower left", fontsize=8, frameon=False)
    ax_coh.legend(loc="upper left", fontsize=8, frameon=False)
    fig.legend(handles=handles, loc="lower center", fontsize=8, ncol=5, frameon=False)
    ax_rain.xaxis.set_major_locator(mdates.MonthLocator())
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_rain.set_xlim(pd.Timestamp("2025-11-10"), pd.Timestamp("2026-05-25"))
    fig.suptitle(f"{fid}: label {meta['label']}, {meta['area_ha']:.1f} ha", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --- report -------------------------------------------------------------------------------


def write_report(ctx) -> None:
    err, summ, near, coefs, imps, info, rain_ok, cane_ids, nc_ids = (
        ctx[k]
        for k in ("err", "summ", "near", "coefs", "imps", "info", "rain_ok", "cane_ids", "nc_ids")
    )
    wide = err.pivot(index="field_id", columns="method", values=["date", "err_days"])
    win = err.drop_duplicates("field_id").set_index("field_id")
    order = sorted(cane_ids, key=lambda s: (win.loc[s, "mid"], s))
    rows = []
    for fid in order:
        rows.append(
            {
                "field": fid,
                "optical window": f"{win.loc[fid, 'harvest_window_start']:%m-%d} → "
                f"{win.loc[fid, 'harvest_window_end']:%m-%d}",
                **{f"{METHODS[m]} date": fmt_d(wide.loc[fid, ("date", m)]) for m in FIELD_METHODS},
                **{
                    f"{METHODS[m]} err": fmt_e(wide.loc[fid, ("err_days", m)])
                    for m in FIELD_METHODS
                },
            }
        )
    L = [
        "# Exercise 4b: multi-signal trained harvest detector vs exercise-4 detectors",
        "",
        "Generated by `scripts/10_harvest_classifier.py`. Errors are in days, predicted date "
        "minus the midpoint of the Sentinel-2 NDVI harvest window (negative = early). Model dates "
        "are out-of-fold (GroupKFold by field, 5 folds). The rule runs on ascending HV, as in "
        "exercise 4. The ratio test runs on each orbit; its per-scene decisions, ascending and "
        "descending merged in time, go through the same two-in-a-row date rule as the models "
        "(~6-day sampling for both). The ascending-only ratio test of exercise 4 is a reference "
        "row in the summary.",
        "",
        "**Scope.** The detector dates the harvest on fields already known to be sugarcane. "
        "Deciding whether a field is cane is a separate upstream step (next exercise). The "
        "false-alarm column below is the evidence for why that step is needed.",
        "",
        f"Training fields: {len(cane_ids)} (`label == cana` with a harvest window). "
        f"Negative control: {len(nc_ids)} fields with another label ({', '.join(nc_ids)}). "
        "GPM rain feature: "
        + ("available." if rain_ok else "**not available (Earth Engine failed); rain is NaN**."),
        "",
        "## Per field",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "## Summary",
        "",
        md_table(summ),
        "",
        "* With 26 fields, differences of 2-3 days between methods are within one acquisition "
        "and not significant.",
        "* Missed fields are left out of the median and mean error and count as outside both "
        "tolerances.",
        "* False alarms on non-cane fields: fields with any sustained detection in Nov-May.",
        "* False alarms on cane fields outside Nov-May: **not measurable**. Every NISAR "
        f"acquisition on disk is between {ctx['first_acq']:%Y-%m-%d} and "
        f"{ctx['last_acq']:%Y-%m-%d}, all inside the zafra.",
        f"* Row-level out-of-fold scores (labelled rows only, threshold {M.THRESHOLD}): "
        + "; ".join(f"{k} {v}" for k, v in ctx["row_scores"].items())
        + ".",
        f"* xgboost best iterations per fold: {info['best_iterations']}; the full model for the "
        f"non-cane fields uses {ctx['n_trees']} trees.",
        "",
        "## Drying-cane cases",
        "",
        ctx["drying_intro"],
        "",
    ]
    for fid in ctx["late_ids"]:
        e = err[err["field_id"] == fid].set_index("method")
        L += [
            f"### {fid} (window {win.loc[fid, 'harvest_window_start']:%m-%d} → "
            f"{win.loc[fid, 'harvest_window_end']:%m-%d})",
            "",
            "Dates: "
            + ", ".join(
                f"{METHODS[m]} {fmt_d(e.loc[m, 'date'])} ({fmt_e(e.loc[m, 'err_days'])} d)"
                for m in FIELD_METHODS
            ),
            "",
            md_table(near[near["field_id"] == fid].drop(columns="field_id"), "{:.2f}"),
            "",
        ]
    L += [
        "## Ablation: which features carry the dates",
        "",
        "Same folds, same date rule. The main model leaves out rain_3d_mm and gap_days "
        "(season markers in this data: every storm and the only 24-day gap come after every "
        "harvest) and sharpness and slope_hv_3 (no gain in the first run). The rows below put "
        "them back, or remove coherence, to show why.",
        "",
        md_table(ctx["abl"]),
        "",
        "## xgboost feature importance (share of total gain, full model)",
        "",
        md_table(
            imps.rename("gain share").reset_index().rename(columns={"index": "feature"}), "{:.3f}"
        ),
        "",
        "## Logistic regression coefficients (standardized features, full model)",
        "",
        "Positive = pushes towards 'harvested'. Features are correlated (e.g. the HV deltas), "
        "so single coefficients are not effect sizes.",
        "",
        md_table(
            coefs.rename("coef").reset_index().rename(columns={"index": "feature"}), "{:+.3f}"
        ),
        "",
        f"Figures: `results/4b/figures/<field>.png` ({ctx['n_figs']} fields; regenerated by the "
        "script, not versioned). Three are copied to `docs/figures/4b_*.png`.",
        "",
    ]
    (OUT_DIR / "comparison.md").write_text("\n".join(L))


# --- main ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild every cache")
    ap.add_argument("--no-rain", action="store_true", help="skip GPM (Earth Engine)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fields = ts.load_fields(FIELDS)
    for col in ("harvest_window_start", "harvest_window_end"):
        fields[col] = pd.to_datetime(fields[col])
    windows = fields[["id", "harvest_window_start", "harvest_window_end"]].rename(
        columns={"id": "field_id"}
    )
    cane = fields[(fields["label"] == CANE) & fields["harvest_window_start"].notna()]
    nonc = fields[fields["label"] != CANE]
    cane_ids, nc_ids = sorted(cane["id"]), sorted(nonc["id"])

    # 1. features
    gframes, coh, enl, acq, fire_table, rain = load_inputs(fields, args.force, not args.no_rain)
    feats = FT.assemble(gframes, coh, enl, acq, fire_table, rain)
    rain_ok = feats["rain_3d_mm"].notna().any()
    feats.to_parquet(FEATURES_OUT.with_suffix(".parquet"), index=False)
    feats.to_csv(FEATURES_OUT.with_suffix(".csv"), index=False)
    print(
        f"features: {len(feats)} rows, {feats['field_id'].nunique()} fields -> "
        f"{FEATURES_OUT}.{{parquet,csv}}; rain {'OK' if rain_ok else 'NOT AVAILABLE'}"
    )
    print(feats[M.FEATURES[1:]].describe().T[["count", "mean", "min", "max"]].round(3).to_string())

    # 2-3. labels and cross-validation
    cane_rows = M.label_states(feats[feats["field_id"].isin(cane_ids)], windows, keep_window=True)
    lab = cane_rows[cane_rows["state"].notna()]
    print(
        f"labelled rows: {len(lab)} ({int((lab['state'] == 0).sum())} standing, "
        f"{int((lab['state'] == 1).sum())} harvested), {len(cane_rows) - len(lab)} window rows "
        f"dropped from training; {lab['field_id'].nunique()} fields"
    )
    oof, info = M.cross_validate(cane_rows)
    n_trees = int(np.median(info["best_iterations"]))
    models = M.fit_full(cane_rows, n_trees)
    nc_pred = M.predict_rows(models, feats[feats["field_id"].isin(nc_ids)])

    from sklearn.metrics import f1_score, roc_auc_score

    lo = oof[oof["state"].notna()]
    row_scores = {
        name: f"AUC {roc_auc_score(lo['state'], lo[col]):.3f}, "
        f"F1 {f1_score(lo['state'], lo[col] >= M.THRESHOLD):.3f}"
        for name, col in (("logistic", "p_logistic"), ("xgboost", "p_xgb"))
    }
    print("row-level OOF:", row_scores)

    # 4. dates
    e4, _ = run_exercise4(gframes[0], gframes[1])
    model_dates = pd.concat(
        [
            M.dates_from_probabilities(df, col).assign(method=m)
            for m, col in (("logistic", "p_logistic"), ("xgb", "p_xgb"))
            for df in (oof, nc_pred)
        ],
        ignore_index=True,
    )
    all_dates = pd.concat([e4, model_dates], ignore_index=True)
    err = errors(all_dates[all_dates["field_id"].isin(cane_ids)], windows)
    nc_dates = all_dates[all_dates["field_id"].isin(nc_ids)]
    strict_nc = sorted(nonc.loc[nonc["hyp"] != CANE, "id"])
    summ = summary_table(err, nc_dates, strict_nc)
    print(md_table(summ))
    abl = run_ablations(cane_rows, feats[feats["field_id"].isin(nc_ids)], windows, strict_nc)
    print(md_table(abl))

    # drying-cane cases: fields where the rule fired late
    r = err[err["method"] == "rule"].set_index("field_id")["err_days"]
    late = r[r > LATE_DAYS].sort_values(ascending=False)
    late_ids = list(late.index[:N_DRYING]) + [f for f in EX6_LATE if f in late.index[N_DRYING:]]
    drying_intro = (
        f"The rule date is more than {LATE_DAYS} days after the window midpoint on "
        f"{len(late)} of {len(cane_ids)} fields here, not 3 "
        f"({', '.join(f'{f} {v:+.0f} d' for f, v in late.items()) or 'none'}). "
        f"Shown in detail: the {N_DRYING} latest, plus "
        + ", ".join(f"{f} (= {o} of exercises 4-6)" for f, o in EX6_LATE.items())
        + f". Rows within ±{NEAR_DAYS} days of the window midpoint, both orbits; "
        "`sharpness` in dB, `coh` = HH coherence of the asc pair ending on that date, "
        "`p_*` = out-of-fold P(harvested)."
    )
    cols = [
        "field_id",
        "date",
        "orbit",
        "delta_hv_vs_baseline",
        "delta_hv_vs_prev",
        "slope_hv_3",
        "sharpness",
        "coherence_mean",
        "delta_coh_vs_baseline",
        "fire_7d",
        "p_logistic",
        "p_xgb",
    ]
    w = windows.set_index("field_id")
    mids = w["harvest_window_start"] + (w["harvest_window_end"] - w["harvest_window_start"]) / 2
    near = oof[oof["field_id"].isin(late_ids)].copy()
    near = near[(near["date"] - near["field_id"].map(mids)).dt.days.abs() <= NEAR_DAYS]
    near = near.sort_values(["field_id", "date"])[cols].copy()
    near["date"] = near["date"].dt.strftime("%Y-%m-%d")
    near["slope_hv_3"] = near["slope_hv_3"] * 12  # per revisit, easier to read
    near = near.rename(
        columns={
            "delta_hv_vs_baseline": "Δbase",
            "delta_hv_vs_prev": "Δprev",
            "slope_hv_3": "slope/12d",
            "coherence_mean": "coh",
            "delta_coh_vs_baseline": "Δcoh",
        }
    )

    # 5. outputs
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    nc_pred.to_csv(OUT_DIR / "noncane_predictions.csv", index=False)
    all_dates.to_csv(OUT_DIR / "dates.csv", index=False)
    pivot = all_dates.pivot(index="field_id", columns="method", values="date")
    near_fires = fire_table[fire_table["dist_edge_m"] <= FT.FIRE_BUFFER_M]
    near_fires = near_fires[
        near_fires["acq_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .between(pd.Timestamp("2025-11-10"), pd.Timestamp("2026-05-25"))
    ]
    for _, meta in fields.iterrows():
        fid = meta["id"]
        plot_field(
            fid,
            meta,
            feats[feats["field_id"] == fid],
            coh[coh["field_id"] == fid],
            near_fires[near_fires["field_id"] == fid],
            pivot.loc[fid].to_dict() if fid in pivot.index else {},
            FIG_DIR / f"{fid}.png",
        )

    write_report(
        {
            "err": err,
            "summ": summ,
            "near": near,
            "coefs": M.logistic_coefficients(models[0]),
            "imps": M.xgb_importances(models[1]),
            "info": info,
            "rain_ok": rain_ok,
            "cane_ids": cane_ids,
            "nc_ids": nc_ids,
            "late_ids": late_ids,
            "abl": abl,
            "drying_intro": drying_intro,
            "row_scores": row_scores,
            "n_trees": n_trees,
            "first_acq": feats["date"].min(),
            "last_acq": feats["date"].max(),
            "n_figs": len(fields),
        }
    )
    print(f"\nwrote {OUT_DIR / 'comparison.md'} and {len(fields)} figures in {FIG_DIR}")
    print(f"total runtime: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
