"""Exercise 4c: cane / non-cane gate for field polygons, and the gated harvest detector.

    python scripts/11_cane_gate.py [--force]

1. Per-field, per-zafra features (src/zafra/cane_features.py) for the 38 fields_v2
   polygons, zafras 2024-25 and 2025-26: data/features/fields_v2_cane_features.{parquet,csv}.
   Radar from the scripts 08/10 caches in outputs/multisignal/; Sentinel-2 NDVI (Jun 2024 ..
   May 2026) and the bare-pixel shares from Earth Engine, cached in outputs/cane_gate/;
   fires from data/fires/fields_v2_fires.csv.
2. Leave-one-field-out logistic regression for variants A (radar), A+F (radar + fire),
   B (optical), C (all), plus the NDVI rule and the radar rule (src/zafra/detect/cane_gate.py).
3. Gate + 4b detector: the harvest dates of every 4b method (results/4b/dates.csv) are kept
   only on fields the gate calls cane (gate decisions are the leave-one-out ones).
4. classify_fields on data/fields/p36_neighbours.geojson (unlabelled), every variant.
5. results/4c/comparison.md, results/4c/*.csv, results/4c/figures/drop_vs_coherence.png
   (copied to docs/figures/4c_drop_vs_coherence.png).
"""

import argparse
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zafra import cane_features as CF
from zafra import fires as F
from zafra.detect import cane_gate as G
from zafra.paths import (
    FEATURES_DIR,
    FIELDS_DIR,
    FIRES_DIR,
    NISAR_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    ROOT,
)

FIELDS = FIELDS_DIR / "fields_v2.geojson"
NEIGHBOURS = FIELDS_DIR / "p36_neighbours.geojson"
ASC_DIR = NISAR_DIR
DESC_DIR = NISAR_DIR / "GCOV_DESCENDING_t113f79"
GUNW_DIR = NISAR_DIR / "GUNW_ASCENDING_t19f11"
MS_DIR = OUTPUTS_DIR / "multisignal"
CACHE = OUTPUTS_DIR / "cane_gate"
LABEL_SERIES = OUTPUTS_DIR / "labels" / "ndvi_series.csv"  # script 07 cache, for a check
FIRE_TABLE = FIRES_DIR / "fields_v2_fires.csv"
FEATURES_OUT = FEATURES_DIR / "fields_v2_cane_features"
DATES_4B = RESULTS_DIR / "4b" / "dates.csv"
OUT_DIR = RESULTS_DIR / "4c"
FIG_DIR = OUT_DIR / "figures"
DOC_FIG = ROOT / "docs" / "figures" / "4c_drop_vs_coherence.png"

NDVI_START, NDVI_END = "2024-06-01", "2026-06-01"  # crop years of both zafras
ZAFRAS = list(CF.ZAFRAS)
TRAIN_Z = G.TRAIN_ZAFRA
SEASON = (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-05-31"))  # as in script 10
METHODS_4B = {
    "rule": "rule (asc)",
    "ratio_asc": "ratio test (asc only)",
    "ratio": "ratio test (asc+desc)",
    "logistic": "logistic",
    "xgb": "xgboost",
}
RULES = {"rule_ndvi": "rule: NDVI (script 07)", "rule_radar": "rule: radar (drop + coherence)"}
# Validated categorical slots of script 10 (dataviz default palette); marker shape is the
# second cue, so the label never depends on colour alone.
LABEL_STYLE = {
    "cana": ("#1baf7a", "o", "cana"),
    "dudoso": ("#eda100", "s", "dudoso (NDVI: no call)"),
    "no_cana": ("#e34948", "X", "no_cana"),
}
INK, MUTED = "#1f1f1e", "#6b6b66"


def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [
            "—"
            if (isinstance(v, float) and np.isnan(v))
            else floatfmt.format(v)
            if isinstance(v, float)
            else str(v)
            for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- 1. features ----------------------------------------------------------------------------


def build_features(fields, force):
    acq, coh = CF.load_radar_inputs(FIELDS, MS_DIR, ASC_DIR, DESC_DIR, GUNW_DIR, force)
    ndvi = CF.load_ndvi(fields, CACHE / "ndvi_series.csv", NDVI_START, NDVI_END, force=force)
    check = ""
    if LABEL_SERIES.exists():
        old = pd.read_csv(LABEL_SERIES, parse_dates=["date"])
        m = old.merge(ndvi, on=["id", "date"], suffixes=("_07", ""))
        d = (m["ndvi_mean_07"] - m["ndvi_mean"]).abs()
        check = (
            f"NDVI re-fetch vs script-07 cache: {len(m)} common (field, date) rows, "
            f"max |diff| {d.max():.2e}"
        )
        print(check)
    t = CF.build_table(fields["id"], ZAFRAS, acq, coh, ndvi, None)
    req = t.dropna(subset=["collapse_low_image_id"])
    bare = CF.load_bare_shares(
        fields,
        list(zip(req["field_id"], req["collapse_low_image_id"], strict=True)),
        CACHE / "bare_share.csv",
    )
    feats = CF.build_table(
        fields["id"], ZAFRAS, acq, coh, ndvi, F.load_fire_table(FIRE_TABLE), bare
    )
    feats.to_parquet(FEATURES_OUT.with_suffix(".parquet"), index=False)
    feats.to_csv(FEATURES_OUT.with_suffix(".csv"), index=False, float_format="%.4f")
    return feats, check


# --- 2. evaluation --------------------------------------------------------------------------


def run_loo(df, y):
    """LOO probabilities and decisions for every variant, plus the two rules."""
    preds = pd.DataFrame({"field_id": df["field_id"], "label": df["label"], "y": y})
    for v, feats in G.VARIANTS.items():
        preds[f"p_{v}"] = G.loo_probabilities(df, y, feats)
        preds[f"pred_{v}"] = (preds[f"p_{v}"] >= G.THRESHOLD).astype(int)
    preds["pred_rule_ndvi"] = G.rule_ndvi(df).to_numpy()
    preds["pred_rule_radar"] = G.rule_radar(df).to_numpy()
    return preds


def metrics_table(preds) -> pd.DataFrame:
    rows = []
    for key, name in [*((v, G.VARIANT_NAMES[v]) for v in G.VARIANTS), *RULES.items()]:
        e = G.evaluate(preds["y"], preds[f"pred_{key}"])
        rows.append(
            {
                "method": name,
                "accuracy": e["accuracy"],
                "precision (cane)": e["precision_cane"],
                "recall (cane)": e["recall_cane"],
                "TP": e["tp"],
                "FP": e["fp"],
                "FN": e["fn"],
                "TN": e["tn"],
            }
        )
    return pd.DataFrame(rows)


def misclassified(preds) -> pd.DataFrame:
    rows = []
    for key, name in [*((v, G.VARIANT_NAMES[v]) for v in G.VARIANTS), *RULES.items()]:
        bad = preds[preds[f"pred_{key}"] != preds["y"]]
        for _, r in bad.iterrows():
            p = r.get(f"p_{key}", np.nan)
            rows.append(
                {
                    "method": name,
                    "field": r["field_id"],
                    "label": r["label"],
                    "error": "missed cane" if r["y"] == 1 else "non-cane called cane",
                    "p_cane": f"{p:.2f}" if pd.notna(p) else "rule",
                }
            )
    return pd.DataFrame(rows)


# --- 3. gate + detector ---------------------------------------------------------------------


def gated_false_alarms(preds, dates) -> pd.DataFrame:
    """Non-cane fields with a detection in Nov-May, before and after each gate; cane fields
    the gate would drop (the detector never runs on them)."""
    in_season = dates["detected"].astype(bool) & dates["date"].between(*SEASON)
    det = dates[in_season]
    nc = preds.loc[preds["y"] == 0, "field_id"]
    cane = preds.loc[preds["y"] == 1, "field_id"]
    radar_yes = set(preds.loc[preds[f"pred_{G.RADAR_VARIANT}"] == 1, "field_id"])
    gates = {"no gate": set(preds["field_id"])}
    for v in G.VARIANTS:
        gates[G.VARIANT_NAMES[v]] = set(preds.loc[preds[f"pred_{v}"] == 1, "field_id"])
    for k, name in RULES.items():
        gates[name] = set(preds.loc[preds[f"pred_{k}"] == 1, "field_id"])
    rows = []
    for gname, keep in gates.items():
        row = {"gate": gname}
        for m, mname in METHODS_4B.items():
            fired = set(det.loc[det["method"] == m, "field_id"]) & set(nc)
            row[mname] = f"{len(fired & keep)}/{len(nc)}"
        row["non-cane passed by gate"] = ", ".join(sorted(set(nc) & keep)) or "none"
        # Fields the gate passes although the radar-only model (p_cane_radar) says no: the
        # ones a map should flag.
        skip = gname in ("no gate", G.VARIANT_NAMES[G.RADAR_VARIANT])
        row["passed, radar says no"] = "—" if skip else ", ".join(sorted(keep - radar_yes))
        row["cane fields gated out"] = f"{len(set(cane) - keep)}/{len(cane)}"
        rows.append(row)
    return pd.DataFrame(rows)


# --- 4. neighbours --------------------------------------------------------------------------


def classify_neighbours(train_df, y):
    gdf = gpd.read_file(NEIGHBOURS)
    feats = G.compute_features(gdf, TRAIN_Z, optical=True, cache_dir=CACHE / "p36_neighbours")
    out = gdf[["id", "label", "area_ha", "dist_m"]].rename(columns={"label": "ndvi_label_2024_25"})
    # The operational output: default gate (optical) plus the radar-only evidence ...
    r = G.classify_fields(gdf, TRAIN_Z, features=feats, train=(train_df, y))
    for c in ("p_cane", "cane_pred", "confidence", "gate_variant", "p_cane_radar"):
        out[c] = r[c].to_numpy()
    # ... and the two other variants, for reference.
    for v in ("A+F", "C"):
        r = G.classify_fields(gdf, TRAIN_Z, v, features=feats, train=(train_df, y))
        out[f"p_{v}"] = r["p_cane"].to_numpy()
    f = feats.set_index("field_id").loc[out["id"]]
    out["rule_ndvi"] = G.rule_ndvi(f).to_numpy()
    out["rule_radar"] = G.rule_radar(f).to_numpy()
    out["hv_drop_db"] = f["hv_max_single_drop_db"].to_numpy()
    out["hv_drop_date"] = pd.to_datetime(f["hv_drop_date"]).dt.strftime("%m-%d").to_numpy()
    out["coh_min"] = f["coh_min"].to_numpy()
    out["ndvi_collapse"] = pd.to_datetime(f["collapse_date"]).dt.strftime("%m-%d").to_numpy()
    return out, feats


# --- 5. figure ------------------------------------------------------------------------------


def plot_scatter(df, preds, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    d = df.merge(preds[["field_id", "pred_A"]], on="field_id")
    for lab, (color, marker, name) in LABEL_STYLE.items():
        s = d[d["label"] == lab]
        ax.scatter(
            s["hv_max_single_drop_db"],
            s["coh_min"],
            s=70 if marker != "X" else 90,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            label=f"{name} ({len(s)})",
            zorder=3,
        )
    # Label only the fields the radar gate (A) gets wrong.
    for _, r in d[d["pred_A"] != (d["label"] == "cana")].iterrows():
        ax.annotate(
            r["field_id"],
            (r["hv_max_single_drop_db"], r["coh_min"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    ax.axvline(G.RADAR_RULE_DROP_DB, color=MUTED, ls="--", lw=1)
    ax.text(
        G.RADAR_RULE_DROP_DB,
        ax.get_ylim()[0],
        "radar-rule drop threshold ",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.set_xlabel("largest HV drop between consecutive same-orbit acquisitions (dB)")
    ax.set_ylabel("lowest field-mean coherence of the zafra (12-day asc pair)")
    ax.set_title(
        "Zafra 2025-26: cane fields drop more and decorrelate more\n"
        "(labels from NDVI; named points = misclassified by the radar-only gate A)",
        fontsize=11,
        color=INK,
    )
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- report ---------------------------------------------------------------------------------


def write_report(ctx):
    L = [
        "# Exercise 4c: cane / non-cane gate",
        "",
        "Generated by `scripts/11_cane_gate.py`. Method: `docs/METHODS.md`, section "
        '"Field selection (cane gate)".',
        "",
        "**Circularity.** The cane labels come from Sentinel-2 NDVI phenology (script 07) plus "
        "visual tracing. Variant B and the NDVI rule are therefore partly re-learning the "
        "labelling rule, and their scores are optimistic by construction. Variant A (radar "
        "only) is the honest test; A+F adds VIIRS fire, also independent of NDVI.",
        "",
        "**Operational gate: B (optical).** It drops no cane field; A drops 3, including p36, "
        "the only grower-confirmed field. Optical selects the fields, radar dates the harvest. "
        "`classify_fields` always also returns `p_cane_radar` (variant A) as the independent "
        "radar evidence. C is not recommended (no gain over B, drops p8).",
        "",
        f"Evaluation: leave-one-field-out on {ctx['n']} fields, zafra {TRAIN_Z} "
        f"({ctx['n_cane']} `cana` = positive; {ctx['n_nc']} others = negative: "
        f"{ctx['n_dudoso']} `dudoso`, {ctx['n_no']} `no_cana`). Threshold P >= {G.THRESHOLD}. "
        "Rules have no training, so they are scored directly.",
        "",
        ctx["ndvi_check"],
        "",
        "## Leave-one-field-out results (2025-26)",
        "",
        md_table(ctx["metrics"]),
        "",
        "## Misclassified fields",
        "",
        md_table(ctx["mis"]) if len(ctx["mis"]) else "None.",
        "",
        "p5 is `dudoso` with an NDVI hypothesis of cane: calling it cane may be right.",
        "",
        "## Optical in 2024-25 (no radar that year)",
        "",
        md_table(ctx["metrics_2425"]),
        "",
        "## Gate + 4b harvest detector: false alarms on the non-cane fields",
        "",
        "A false alarm is a non-cane field with a sustained detection in Nov-May "
        "(`results/4b/dates.csv`). After gating, only fields the gate calls cane "
        "(leave-one-out decision) count. The last column is the price: cane fields the "
        "detector would never see.",
        "",
        md_table(ctx["fa"]),
        "",
        "## Logistic coefficients (standardized features, fit on all fields)",
        "",
        "Positive = towards cane. Features are correlated (the HV statistics among "
        "themselves, the NDVI ones among themselves), so single coefficients are not effect "
        "sizes.",
        "",
    ]
    for v, s in ctx["coefs"].items():
        L += [
            f"### {G.VARIANT_NAMES[v]}",
            "",
            md_table(
                s.rename("coef").reset_index().rename(columns={"index": "feature"}), "{:+.3f}"
            ),
            "",
        ]
    L += [
        "## p36 neighbours (unlabelled; SNIC segments)",
        "",
        "`ndvi_label_2024_25` is the script-07-style NDVI label computed in exercise 9, not a "
        "ground truth. All P(cane) are for 2025-26 from models fit on all 38 fields: `p_cane` is "
        "the default gate (B, optical), `p_cane_radar` the radar-only variant A, `p_A+F` and "
        "`p_C` are for reference.",
        "",
        md_table(ctx["neigh"]),
        "",
        "## Feature summary per class (2025-26, median)",
        "",
        md_table(ctx["by_class"]),
        "",
        "Figure: `results/4c/figures/drop_vs_coherence.png` (copy in "
        "`docs/figures/4c_drop_vs_coherence.png`).",
        "",
    ]
    (OUT_DIR / "comparison.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild every cache")
    args = ap.parse_args()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fields = gpd.read_file(FIELDS)
    feats, ndvi_check = build_features(fields, args.force)
    cov = feats.groupby("zafra")[["has_radar", "has_optical"]].sum()
    print("fields with data per zafra:\n" + cov.to_string())

    df, y = G.load_training(FEATURES_OUT.with_suffix(".parquet"), FIELDS, TRAIN_Z)
    preds = run_loo(df, y)
    metrics = metrics_table(preds)
    mis = misclassified(preds)
    print(md_table(metrics))
    print(md_table(mis))

    d24, y24 = G.load_training(FEATURES_OUT.with_suffix(".parquet"), FIELDS, "2024-25")
    p24 = G.loo_probabilities(d24, y24, G.VARIANTS["B"])
    m24 = []
    for name, pred in (
        (G.VARIANT_NAMES["B"], (p24 >= G.THRESHOLD).astype(int)),
        (RULES["rule_ndvi"], G.rule_ndvi(d24).to_numpy()),
    ):
        e = G.evaluate(y24, pred)
        m24.append(
            {
                "method": name,
                "accuracy": e["accuracy"],
                "precision (cane)": e["precision_cane"],
                "recall (cane)": e["recall_cane"],
                **{k.upper(): e[k] for k in ("tp", "fp", "fn", "tn")},
            }
        )
    metrics_2425 = pd.DataFrame(m24)

    coefs = {v: G.coefficients(G.fit(df, y, f), f) for v, f in G.VARIANTS.items()}

    dates = pd.read_csv(DATES_4B, parse_dates=["date", "date_before", "date_after"])
    fa = gated_false_alarms(preds, dates)
    print(md_table(fa))

    neigh, nfeats = classify_neighbours(df, y)
    print(md_table(neigh))
    nfeats.to_csv(OUT_DIR / "p36_neighbours_features.csv", index=False, float_format="%.4f")

    plot_scatter(df, preds, FIG_DIR / "drop_vs_coherence.png")
    shutil.copy(FIG_DIR / "drop_vs_coherence.png", DOC_FIG)

    show = [
        "hv_range_db",
        "hv_max_single_drop_db",
        "coh_min",
        "joint_collapse",
        "hv_pre_level_db",
        "ndvi_longest_high_run_days",
        "ndvi_n_collapses",
        "bare_share_after_collapse",
        "fire_count_in_zafra",
    ]
    by_class = (
        df.groupby("label")[show].median().T.reset_index().rename(columns={"index": "feature"})
    )

    preds.to_csv(OUT_DIR / "loo_predictions.csv", index=False, float_format="%.4f")
    neigh.to_csv(OUT_DIR / "p36_neighbours_gate.csv", index=False, float_format="%.4f")
    write_report(
        {
            "n": len(df),
            "n_cane": int(y.sum()),
            "n_nc": int((y == 0).sum()),
            "n_dudoso": int((df["label"] == "dudoso").sum()),
            "n_no": int((df["label"] == "no_cana").sum()),
            "ndvi_check": ndvi_check,
            "metrics": metrics,
            "mis": mis,
            "metrics_2425": metrics_2425,
            "fa": fa,
            "coefs": coefs,
            "neigh": neigh,
            "by_class": by_class,
        }
    )
    print(f"wrote {OUT_DIR / 'comparison.md'}; runtime {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
