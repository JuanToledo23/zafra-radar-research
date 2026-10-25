"""Exercise 5: cross each field's NISAR series with VIIRS active fires (FIRMS).

    python scripts/05_fires_per_field.py [--buffer-m 500] [--pre-days 15]

For each field: list VIIRS detections within --buffer-m of the polygon during the 2025/26
zafra, flag the ones in the --pre-days before the END of the field's radar drop window
(cane is burned, then cut; the cut is what makes HV backscatter drop), and flag the ones
outside the 09:00-17:00 local burning window. Prints markdown tables and writes plots to
outputs/fires/.

Read the results with two limits in mind (details in src/zafra/fires.py):
* A VIIRS pixel is ~375 m wide, so a detection within 500 m may be a neighbouring field.
* VIIRS only looks ~13:30 and ~01:30 local. No detection does not mean no burn.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from zafra import fires as F
from zafra import timeseries as ts
from zafra.paths import FIELDS_DIR, FIRMS_DIR, NISAR_DIR, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "fires"
SERIES_CSV = OUTPUTS_DIR / "timeseries" / "series_6fields_15scenes.csv"

# Search window for this exercise (local dates, inclusive). Wider than the official season
# dates used in exercise 3 so that early/late burns are not cut off.
ZAFRA_START = pd.Timestamp("2025-11-15", tz=F.LOCAL_TZ)
ZAFRA_END = pd.Timestamp("2026-05-16", tz=F.LOCAL_TZ)  # exclusive: includes all of 05-15

# Radar drop window per field, eyeballed in exercise 3: (last scene before the drop, first
# scene after it). The cut happened somewhere between the two acquisitions.
DROP_WINDOWS = {
    "field_01": ("2026-02-03", "2026-02-15"),
    "field_02": ("2026-01-22", "2026-02-03"),
    "field_03": ("2026-01-22", "2026-02-03"),
    "field_04": ("2026-03-23", "2026-04-04"),
    "field_05": ("2025-12-17", "2025-12-29"),  # suspect: gradual decline, not a clean step
    "field_06": ("2026-04-04", "2026-04-16"),
}

# Tlaquiltenango municipal rule: burning allowed only 09:00-17:00 local time.
RULE_START_H, RULE_END_H = 9, 17


def scene_times() -> dict[str, pd.Timestamp]:
    """Map scene date -> exact acquisition start (UTC), parsed from GCOV file names.

    NISAR images Morelos at ~12:05 UTC (06:05 local), *before* the ~13:30 local VIIRS
    overpass. So a fire detected on the same date as the "after" scene happened after the
    radar looked and cannot explain that scene's drop. Falls back to 12:05 UTC if the
    GCOV files are not present.
    """
    return {s["date"]: s["start_utc"] for s in ts.list_scenes(NISAR_DIR)}


def check_drop_windows(hv: pd.DataFrame) -> None:
    """Warn if the hard-coded windows disagree with the largest HV drop in the series."""
    for fid, (a, b) in DROP_WINDOWS.items():
        d = hv[hv["field_id"] == fid].sort_values("date").reset_index(drop=True)
        i = int(d["mean_db"].diff().idxmin())
        got = (f"{d['date'].iloc[i - 1]:%Y-%m-%d}", f"{d['date'].iloc[i]:%Y-%m-%d}")
        if got != (a, b):
            print(f"WARNING {fid}: largest HV drop in series is {got}, script uses {(a, b)}")


def in_rule_window(local: pd.Series) -> pd.Series:
    minutes = local.dt.hour * 60 + local.dt.minute
    return (minutes >= RULE_START_H * 60) & (minutes <= RULE_END_H * 60)


def fmt_local(t: pd.Timestamp) -> str:
    return f"{t:%Y-%m-%d %H:%M}"


def plot_field(hv: pd.DataFrame, near: pd.DataFrame, meta, step, out) -> None:
    """HV series with a vertical line per nearby fire. Everything on a UTC time axis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    to_axis = lambda t: t.tz_convert("UTC").tz_localize(None)  # noqa: E731 (naive UTC)

    ax.axvspan(to_axis(step["pre_start"]), to_axis(step["end"]), color="tab:red", alpha=0.07)
    ax.axvspan(to_axis(step["start"]), to_axis(step["end"]), color="0.6", alpha=0.25)
    for _, f in near.iterrows():
        # Solid = in the pre-step window; faint = elsewhere in the zafra.
        alpha = 0.9 if f["pre_step"] else 0.35
        ax.axvline(to_axis(f["acq_utc"]), color="tab:red", linewidth=1.2, alpha=alpha)
    for d, label in [(ZAFRA_START, "zafra start"), (ZAFRA_END, "zafra end")]:
        ax.axvline(to_axis(d), color="0.4", linestyle="--", linewidth=1)
        ax.text(
            to_axis(d), 1.01, label, transform=ax.get_xaxis_transform(), ha="center", fontsize=8
        )
    ax.plot(hv["date"], hv["mean_db"], marker="o", color="tab:green", zorder=3)

    ax.legend(
        handles=[
            Line2D([], [], color="tab:green", marker="o", label="HV field mean (dB)"),
            Line2D([], [], color="tab:red", alpha=0.9, label="VIIRS fire, pre-step window"),
            Line2D([], [], color="tab:red", alpha=0.35, label="VIIRS fire, other zafra dates"),
            plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=0.25, label="radar drop window"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color="tab:red",
                alpha=0.07,
                label=f"{step['pre_days']} days before drop end",
            ),
        ],
        fontsize=8,
        loc="lower left",
    )
    ax.set_title(
        f"{meta['id']} — {meta['name']}: {len(near)} VIIRS detections within "
        f"{step['buffer_m']} m (375 m pixels: may be a neighbouring field)",
        fontsize=10,
        pad=16,
    )
    ax.set_ylabel("HV gamma-0 (dB), field mean")
    ax.set_xlabel("time (UTC)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(
        to_axis(ZAFRA_START) - pd.Timedelta(days=7), to_axis(ZAFRA_END) + pd.Timedelta(days=7)
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_map(fields, zafra_fires, buffer_m, out) -> None:
    """Fields, their buffers and every zafra detection nearby, in UTM metres."""
    fields_m = fields.to_crs(F.METRIC_CRS)
    buffers = fields_m.buffer(buffer_m)
    xmin, ymin, xmax, ymax = buffers.total_bounds
    pad = 1500
    pts = zafra_fires.to_crs(F.METRIC_CRS).cx[xmin - pad : xmax + pad, ymin - pad : ymax + pad]

    # Months are ordered, so they get a sequential colormap (dark = early, bright = late).
    # viridis rises steadily in lightness, so the order survives colour blindness and greyscale.
    months = pd.period_range(ZAFRA_START.tz_localize(None), ZAFRA_END.tz_localize(None), freq="M")
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / (len(months) - 1)) for i, m in enumerate(months)}
    month = pts["acq_local"].dt.tz_localize(None).dt.to_period("M")

    fig, ax = plt.subplots(figsize=(11, 8))
    buffers.boundary.plot(ax=ax, color="0.5", linestyle="--", linewidth=0.8)
    fields_m.plot(ax=ax, facecolor="tab:green", edgecolor="darkgreen", alpha=0.5)
    for m in months:
        sel = pts[month == m]
        ax.scatter(
            sel.geometry.x,
            sel.geometry.y,
            s=40,
            color=colors[m],
            edgecolor="0.2",
            linewidth=0.6,
            label=f"{m:%b %Y} ({len(sel)})",
            zorder=3,
        )
    for _, f in fields_m.iterrows():
        c = f.geometry.centroid
        ax.annotate(f["id"], (c.x, c.y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("UTM 14N easting (m)")
    ax.set_ylabel("UTM 14N northing (m)")
    ax.legend(
        title="VIIRS detections (local month)",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    ax.set_title(
        f"VIIRS fires {ZAFRA_START:%Y-%m-%d} to {ZAFRA_END - pd.Timedelta(days=1):%Y-%m-%d}; "
        f"dashed = {buffer_m} m buffer\n"
        f"Dots are pixel centres of ~{F.VIIRS_PIXEL_M} m pixels (larger off-nadir)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fields", default=str(FIELDS_DIR / "fields.geojson"))
    p.add_argument("--buffer-m", type=float, default=500)
    p.add_argument("--pre-days", type=int, default=15)
    args = p.parse_args()
    buffer_m = int(args.buffer_m) if args.buffer_m.is_integer() else args.buffer_m
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = ts.load_fields(args.fields)
    series = pd.read_csv(SERIES_CSV, parse_dates=["date"])
    hv_all = series[series["pol"] == "HVHV"]
    check_drop_windows(hv_all)
    acq = scene_times()

    fires = F.load_firms(FIRMS_DIR)
    zafra = fires[(fires["acq_local"] >= ZAFRA_START) & (fires["acq_local"] < ZAFRA_END)]
    print(
        f"{len(fires)} VIIRS detections loaded; {len(zafra)} in the zafra window "
        f"{ZAFRA_START:%Y-%m-%d}..{ZAFRA_END - pd.Timedelta(days=1):%Y-%m-%d} (whole bbox)"
    )
    print(
        f"NOTE: a VIIRS pixel is ~{F.VIIRS_PIXEL_M} m (larger off-nadir), so a detection "
        f"within {buffer_m} m of a field may belong to a neighbouring field."
    )

    summary, pngs = [], []
    for _, meta in fields.iterrows():
        fid = meta["id"]
        a, b = DROP_WINDOWS[fid]
        default = lambda d: pd.Timestamp(f"{d} 12:05", tz="UTC")  # noqa: E731
        step = {
            "start": acq.get(a, default(a)),
            "end": acq.get(b, default(b)),
            "pre_days": args.pre_days,
            "buffer_m": buffer_m,
        }
        step["pre_start"] = step["end"] - pd.Timedelta(days=args.pre_days)

        near = F.fires_near_field(zafra, meta.geometry, buffer_m=args.buffer_m)
        # Pre-step = detected in the N days before the radar saw the field already cut.
        near["pre_step"] = (near["acq_utc"] >= step["pre_start"]) & (near["acq_utc"] <= step["end"])
        near["outside_rule"] = ~in_rule_window(near["acq_local"])

        print(f"\n### {fid} — {meta['name']} ({meta['municipio']})")
        print(
            f"radar drop window {a} -> {b}; pre-step window "
            f"{fmt_local(step['pre_start'].tz_convert(F.LOCAL_TZ))} -> "
            f"{fmt_local(step['end'].tz_convert(F.LOCAL_TZ))} local\n"
        )
        if near.empty:
            print(f"(no VIIRS detections within {buffer_m} m during the zafra)")
        else:
            print(
                "| date (local) | time (local) | distance_m (centroid) | dist_edge_m | frp MW "
                "| confidence | satellite | pre-step | outside 09-17 |"
            )
            print("|---|---|---|---|---|---|---|---|---|")
            for _, f in near.iterrows():
                print(
                    f"| {f['acq_local']:%Y-%m-%d} | {f['acq_local']:%H:%M} | "
                    f"{f['distance_m']:.0f} | {f['dist_edge_m']:.0f} | {f['frp']:.2f} | "
                    f"{f['confidence']} | {f['satellite']} | "
                    f"{'YES' if f['pre_step'] else ''} | "
                    f"{'YES' if f['outside_rule'] else ''} |"
                )

        pre = near[near["pre_step"]]
        if pre.empty:
            pre_txt, earliest, verdict = "—", "—", "no fire found"
        else:
            pre_txt = "; ".join(
                f"{t:%m-%d %H:%M}{' (outside 09-17)' if o else ''}"
                for t, o in zip(pre["acq_local"], pre["outside_rule"], strict=True)
            )
            first = pre.iloc[0]
            lead = (step["end"] - first["acq_utc"]).total_seconds() / 86400
            earliest = f"{first['acq_local']:%Y-%m-%d %H:%M} ({lead:.1f} d before drop end)"
            verdict = "burn-before-step"
        summary.append((fid, len(near), pre_txt, earliest, verdict))

        out = OUT_DIR / f"{fid}_fires.png"
        hv = hv_all[hv_all["field_id"] == fid].sort_values("date")
        plot_field(hv, near, meta, step, out)
        pngs.append(out)

    out_map = OUT_DIR / "map_fires.png"
    plot_map(fields, zafra, buffer_m, out_map)
    pngs.append(out_map)

    print(
        f"\n## Summary (VIIRS within {buffer_m} m, zafra "
        f"{ZAFRA_START:%Y-%m-%d}..{ZAFRA_END - pd.Timedelta(days=1):%Y-%m-%d}, "
        f"pre-step = {args.pre_days} days before drop end; times local UTC-6)\n"
    )
    print(
        "| field | n fires in zafra | fires in pre-step window (local) "
        "| earliest burn before step | verdict |"
    )
    print("|---|---|---|---|---|")
    for row in summary:
        print("| " + " | ".join(str(x) for x in row) + " |")
    print(
        f"\nA VIIRS pixel is ~{F.VIIRS_PIXEL_M} m, so a detection within {buffer_m} m may "
        "belong to a neighbouring field (several of these fields are < 1 km apart and can "
        "share detections). 'no fire found' means no VIIRS detection, not no burn: VIIRS "
        "only looks at ~01:30 and ~13:30 local."
    )

    print("\nfigures:")
    for png in pngs:
        print(f"  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
