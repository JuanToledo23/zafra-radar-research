"""Exercise 4: harvest date per field with a rule baseline and an intensity-ratio test.

    python scripts/04_detect_harvest.py [--fields data/fields/fields.geojson] [--buffer-m 500]

Reads the exercise-3 series, re-reads per-pixel HV windows for the first 3 scenes (to estimate
the equivalent number of looks), runs both detectors (src/zafra/detect/), cross-checks with
VIIRS fires (exercise 5 logic) and the author's reference windows, prints a markdown table and
writes outputs/detect/detections.csv and one plot per field. Method: docs/METHODS.md.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zafra import fires as F
from zafra import timeseries as ts
from zafra.detect import GcovPixelProvider, detect_ratio, detect_rule
from zafra.detect import ratio_test as RT
from zafra.paths import FIELDS_DIR, FIRMS_DIR, NISAR_DIR, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "detect"
SERIES_CSV = OUTPUTS_DIR / "timeseries" / "series_6fields_15scenes.csv"
ZAFRA_START = pd.Timestamp("2025-11-15", tz=F.LOCAL_TZ)  # same search window as exercise 5
ZAFRA_END = pd.Timestamp("2026-05-16", tz=F.LOCAL_TZ)
PRE_DAYS = 15  # fires up to this many days before the window end count as "the" burn
P_FLOOR = 1e-30  # p-values below this are shown as "<1e-30" (and clipped in plots)

# Author's eyeballed windows (exercise 3), corrected with FIRMS (exercise 5).
# None = no reliable reference for that field.
REFERENCE = {
    "field_01": ("2026-02-03", "2026-02-15"),
    "field_02": ("2026-01-22", "2026-02-03"),
    "field_03": ("2026-01-22", "2026-02-03"),
    "field_04": None,  # uncertain: sawtooth series, only 84 px
    "field_05": None,  # uncertain: gradual decline
    "field_06": ("2026-02-27", "2026-03-11"),
}


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    return f"<{P_FLOOR:.0e}" if p < P_FLOOR else f"{p:.1e}"


def fmt_window(a, b) -> str:
    return "none" if pd.isna(b) else f"{a:%m-%d}→{b:%m-%d}"


def agreement(dates: list, detected, ref) -> str:
    """'match' if the detected scene is within +/-1 scene of the reference window's end."""
    if ref is None:
        return "n/a (no ref)"
    if pd.isna(detected):
        return "no (none)"
    k = dates.index(pd.Timestamp(detected))
    k_ref = dates.index(pd.Timestamp(ref[1]))
    return "match" if abs(k - k_ref) <= 1 else f"no ({k - k_ref:+d} sc)"


def firms_for_field(zafra_fires, geom, buffer_m, end_utc) -> tuple[pd.DataFrame, str]:
    """All zafra fires near the field, and a label for those in the PRE_DAYS before end_utc."""
    near = F.fires_near_field(zafra_fires, geom, buffer_m=buffer_m)
    if end_utc is None:
        return near, f"— ({len(near)} in zafra)"
    start = end_utc - pd.Timedelta(days=PRE_DAYS)
    pre = near[(near["acq_utc"] >= start) & (near["acq_utc"] <= end_utc)]
    if pre.empty:
        return near, f"none in {PRE_DAYS} d before ({len(near)} in zafra)"
    days = sorted({f"{t:%m-%d}" for t in pre["acq_local"]})
    return near, ", ".join(days)


def plot_field(sc, det, rule, ref, near, meta, out) -> None:
    """Top: HV with baseline, thresholds and windows. Bottom: p-values (log), same dates."""
    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    to_axis = lambda t: t.tz_convert("UTC").tz_localize(None)  # noqa: E731 (naive UTC)
    base, se = det["baseline_db"], det["baseline_se_db"]

    for a in (ax, axp):
        for _, f in near.iterrows():
            a.axvline(to_axis(f["acq_utc"]), color="tab:red", linewidth=1, alpha=0.6)
        if det["detected"]:
            a.axvspan(det["date_before"], det["date_detected"], color="tab:orange", alpha=0.25)
        if rule["detected"]:
            a.axvspan(
                rule["date_before"],
                rule["date_detected"],
                fill=False,
                hatch="//",
                edgecolor="0.45",
                linewidth=0,
            )
        a.grid(alpha=0.3)

    if ref is not None:
        ax.axvspan(pd.Timestamp(ref[0]), pd.Timestamp(ref[1]), ymin=0.97, ymax=1.0, color="k")
    ax.axhspan(base - 2 * se, base + 2 * se, color="tab:green", alpha=0.2, linewidth=0)
    ax.axhline(base, color="tab:green", linewidth=1)
    ax.axhline(base - RT.SUSTAIN_DB, color="tab:green", linestyle="--", linewidth=1)
    ax.plot(sc["date"], sc["hv_db"], marker="o", color="tab:green", zorder=3)
    wet = sc[sc["wet_flag"]]
    ax.scatter(wet["date"], wet["hv_db"], marker="^", s=90, color="tab:blue", zorder=4)
    ax.set_ylabel("HV gamma-0 (dB), field mean")
    ax.legend(
        handles=[
            Line2D([], [], color="tab:green", marker="o", label="HV field mean"),
            Patch(color="tab:green", alpha=0.35, label="baseline ± 2 se"),
            Line2D([], [], color="tab:green", ls="--", label=f"baseline − {RT.SUSTAIN_DB:g} dB"),
            Line2D([], [], color="tab:blue", marker="^", ls="", label="possible wet soil"),
            Patch(color="tab:orange", alpha=0.35, label="ratio-test window"),
            Patch(fill=False, hatch="//", edgecolor="0.45", label="rule window"),
            Patch(color="k", label="reference window (top bar)"),
            Line2D([], [], color="tab:red", alpha=0.6, label="VIIRS fire ≤ buffer"),
        ],
        fontsize=8,
        loc="lower left",
        ncol=2,
    )
    title = f"{meta['id']} — {meta['name']}: "
    if det["detected"]:
        title += (
            f"harvest {det['window_mid']:%Y-%m-%d} ± {RT.HALF_WINDOW_DAYS} d, "
            f"drop {det['drop_db']:+.1f} ± {det['drop_se_db']:.1f} dB, {det['confidence']}"
        )
    else:
        title += "no harvest detected (ratio test)"
    ax.set_title(title, fontsize=10)

    pp = sc["p_prev"].clip(lower=P_FLOOR)
    pb = sc["p_base"].clip(lower=P_FLOOR)
    axp.plot(sc["date"], pp, marker="o", color="tab:purple", label="p vs previous scene")
    axp.plot(sc["date"], pb, marker="s", ls=":", color="0.3", label="p vs baseline")
    axp.axhline(RT.ALPHA_PREV, color="tab:purple", linestyle="--", linewidth=0.8)
    axp.axhline(RT.ALPHA_BASE, color="0.3", linestyle="--", linewidth=0.8)
    axp.set_yscale("log")
    axp.set_ylim(P_FLOOR / 3, 3)
    axp.set_ylabel(f"one-sided p (floor {P_FLOOR:.0e})")
    axp.legend(fontsize=8, loc="lower left")
    axp.set_xlabel("acquisition date (UTC); dashed = α 0.01 (prev), 0.001 (baseline)")
    axp.xaxis.set_major_locator(mdates.MonthLocator())
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axp.set_xlim(to_axis(ZAFRA_START), to_axis(ZAFRA_END))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fields", default=str(FIELDS_DIR / "fields.geojson"))
    p.add_argument("--buffer-m", type=float, default=500)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(SERIES_CSV, parse_dates=["date"])
    fields = ts.load_fields(args.fields).set_index("id", drop=False)
    dates = sorted(series["date"].unique())
    dates = [pd.Timestamp(d) for d in dates]
    provider = GcovPixelProvider(NISAR_DIR, args.fields)

    # Rain guard: GPM if Earth Engine works, otherwise only the HV-rise flag.
    rain = RT.gpm_rain_before(dates, fields.total_bounds)
    if rain is None:
        print(
            "GPM IMERG not available (Earth Engine not authenticated/offline): rain guard "
            f"uses only the HV rise > {RT.WET_RISE_DB} dB flag."
        )
        rain_mm = None
    else:
        rain_mm = {fid: rain for fid in fields.index}  # one IMERG cell covers all fields

    rule = detect_rule(series).set_index("field_id")
    det, scenes = detect_ratio(series, provider, rain_mm=rain_mm, return_scenes=True)
    det = det.set_index("field_id")

    # Sensitivity: the same test with N replaced by an effective N from spatial correlation.
    n_eff, n_scale = {}, {}
    for fid in det.index:
        arr, valid = provider.window(fid, dates[0])
        rx, ry = RT.spatial_autocorr(arr, valid)
        n_eff[fid] = (RT.effective_n(int(valid.sum()), rx, ry), rx, ry)
        n_scale[fid] = n_eff[fid][0] / valid.sum()
    det_eff = detect_ratio(series, provider, rain_mm=rain_mm, n_scale=n_scale).set_index("field_id")

    fires = F.load_firms(FIRMS_DIR)
    zafra = fires[(fires["acq_local"] >= ZAFRA_START) & (fires["acq_local"] < ZAFRA_END)]
    acq = {s["date"]: s["start_utc"] for s in ts.list_scenes(NISAR_DIR)}

    rows, pngs = [], []
    for fid, d in det.iterrows():
        r, ref = rule.loc[fid], REFERENCE.get(fid)
        end = d["date_detected"] if d["detected"] else None
        end_utc = None if end is None else acq.get(f"{end:%Y-%m-%d}", end.tz_localize("UTC"))
        near, firms_txt = firms_for_field(zafra, fields.loc[fid].geometry, args.buffer_m, end_utc)
        rows.append(
            {
                "field_id": fid,
                "enl": d["enl"],
                "enl_per_scene": d["enl_per_scene"],
                "product_looks": d["product_looks"],
                "n_px": d["n_px"],
                "n_eff": n_eff[fid][0],
                "rho_x": n_eff[fid][1],
                "rho_y": n_eff[fid][2],
                "baseline_dates": d["baseline_dates"],
                "baseline_db": d["baseline_db"],
                "baseline_se_db": d["baseline_se_db"],
                "rule_baseline_db": r["baseline_db"],
                "rule_detected": r["detected"],
                "rule_date_before": r["date_before"],
                "rule_date_detected": r["date_detected"],
                "ratio_detected": d["detected"],
                "ratio_date_before": d["date_before"],
                "ratio_date_detected": d["date_detected"],
                "window_mid": d["window_mid"],
                "half_window_days": d["half_window_days"],
                "trigger": d["trigger"],
                "p_prev": d["p_prev"],
                "p_base": d["p_base"],
                "step_db": d["step_db"],
                "step_se_db": d["step_se_db"],
                "drop_db": d["drop_db"],
                "drop_se_db": d["drop_se_db"],
                "confidence": d["confidence"],
                "wet_dates": d["wet_dates"],
                "neff_date_detected": det_eff.loc[fid, "date_detected"],
                "neff_p_prev": det_eff.loc[fid, "p_prev"],
                "neff_p_base": det_eff.loc[fid, "p_base"],
                "neff_confidence": det_eff.loc[fid, "confidence"],
                "firms_pre_window": firms_txt,
                "firms_n_zafra": len(near),
                "reference": "uncertain" if ref is None else f"{ref[0]}→{ref[1]}",
                "rule_agreement": agreement(dates, r["date_detected"], ref),
                "ratio_agreement": agreement(dates, d["date_detected"], ref),
            }
        )
        out = OUT_DIR / f"{fid}_detect.png"
        plot_field(scenes[scenes["field_id"] == fid], d, r, ref, near, fields.loc[fid], out)
        pngs.append(out)

    table = pd.DataFrame(rows)
    csv = OUT_DIR / "detections.csv"
    table.to_csv(csv, index=False)

    print(
        "\n| field | ENL (product looks) | baseline dB±se | rule window | ratio-test window "
        "| p_prev | p_base | drop dB±se | confidence | FIRMS burn date (≤15 d before) "
        "| reference window | agreement rule / ratio |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        ratio_win = fmt_window(t["ratio_date_before"], t["ratio_date_detected"])
        if t["ratio_detected"]:
            ratio_win += f" (mid {t['window_mid']:%m-%d} ±{t['half_window_days']} d)"
            drop = f"{t['drop_db']:+.2f} ± {t['drop_se_db']:.2f}"
        else:
            drop = "—"
        print(
            f"| {t['field_id']} | {t['enl']:.2f} ({t['product_looks']:.2f}) | "
            f"{t['baseline_db']:.2f} ± {t['baseline_se_db']:.2f} | "
            f"{fmt_window(t['rule_date_before'], t['rule_date_detected'])} | {ratio_win} | "
            f"{fmt_p(t['p_prev'])} | {fmt_p(t['p_base'])} | {drop} | "
            f"{t['confidence'] or '—'} | {t['firms_pre_window']} | {t['reference']} | "
            f"{t['rule_agreement']} / {t['ratio_agreement']} |"
        )

    print("\nWet-soil flags (HV rise > 1.5 dB) and step at the detected scene:")
    for t in rows:
        print(
            f"  {t['field_id']}: wet {t['wet_dates'] or 'none'}; baseline scenes "
            f"{t['baseline_dates']}; trigger {t['trigger'] or '—'}; step vs previous "
            f"{t['step_db']:+.2f} ± {t['step_se_db']:.2f} dB"
        )

    print("\nSensitivity: spatial correlation (N -> N_eff from lag-1 autocorrelation)")
    print(
        "| field | N px | rho_x | rho_y | N_eff | detected scene (N / N_eff) "
        "| p_prev N_eff | p_base N_eff | confidence N_eff |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        a, b = t["ratio_date_detected"], t["neff_date_detected"]
        print(
            f"| {t['field_id']} | {t['n_px']} | {t['rho_x']:.2f} | {t['rho_y']:.2f} | "
            f"{t['n_eff']:.0f} | {'—' if pd.isna(a) else f'{a:%m-%d}'} / "
            f"{'—' if pd.isna(b) else f'{b:%m-%d}'} | {fmt_p(t['neff_p_prev'])} | "
            f"{fmt_p(t['neff_p_base'])} | {t['neff_confidence'] or '—'} |"
        )

    print("\nfigures:")
    for png in pngs:
        print(f"  {png}")
    print(f"detections csv: {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
