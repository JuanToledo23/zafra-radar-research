"""Exercise 3: full-season NISAR HV/HH time series per field, for eyeballing the harvest step.

    python scripts/03_timeseries.py [--fields data/fields/fields.geojson] [--force]

Builds (or loads from cache) a table field x date x polarization of field-mean backscatter,
prints a per-field summary and a markdown table, and writes plots to outputs/timeseries/.
No detector here (that is exercise 4): the "largest drop" is just the most negative change
between consecutive scenes, a hint for where to look.
"""

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from zafra import timeseries as ts
from zafra.paths import FIELDS_DIR, NISAR_DIR, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "timeseries"
# Official 2025/26 zafra window for the region (season start / approximate end).
ZAFRA_START = pd.Timestamp("2025-11-27")
ZAFRA_END = pd.Timestamp("2026-05-05")
POL_STYLE = {"HVHV": ("HV (cross-pol)", "tab:green"), "HHHH": ("HH (co-pol)", "tab:blue")}


def summarize(hv: pd.DataFrame) -> dict:
    """HV summary for one field. `hv` = that field's HVHV rows sorted by date."""
    hv = hv.sort_values("date").reset_index(drop=True)
    # Change between each scene and the previous one (dB). The most negative value is the
    # biggest single 12-day loss of backscatter: a candidate harvest interval.
    step = hv["mean_db"].diff()
    i = int(step.idxmin())
    imin = int(hv["mean_db"].idxmin())
    return {
        "first": hv["mean_db"].iloc[0],
        "first_date": hv["date"].iloc[0],
        "last": hv["mean_db"].iloc[-1],
        "last_date": hv["date"].iloc[-1],
        "min": hv["mean_db"].iloc[imin],
        "min_date": hv["date"].iloc[imin],
        "drop": step.iloc[i],
        "drop_a": hv["date"].iloc[i - 1],
        "drop_b": hv["date"].iloc[i],
        "drop_a_db": hv["mean_db"].iloc[i - 1],
        "drop_b_db": hv["mean_db"].iloc[i],
    }


def style_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(alpha=0.3)


def zafra_lines(ax) -> None:
    for d, label in [(ZAFRA_START, "zafra start"), (ZAFRA_END, "zafra end (approx.)")]:
        ax.axvline(d, color="0.4", linestyle="--", linewidth=1)
        ax.text(d, 1.01, label, transform=ax.get_xaxis_transform(), ha="center", fontsize=8)


def plot_field(df_field: pd.DataFrame, meta: pd.Series, summ: dict, out) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for pol, (label, color) in POL_STYLE.items():
        d = df_field[df_field["pol"] == pol].sort_values("date")
        ax.plot(d["date"], d["mean_db"], marker="o", color=color, label=label)
        # Band = +/- std of the per-pixel dB values inside the field: how uneven the field is
        # (speckle + real variation). It is NOT the uncertainty of the mean, which is much
        # smaller; a step bigger than the band is a strong, field-wide change.
        ax.fill_between(
            d["date"],
            d["mean_db"] - d["std_db"],
            d["mean_db"] + d["std_db"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    zafra_lines(ax)
    ax.annotate(
        f"largest HV drop {summ['drop']:+.1f} dB\n"
        f"{summ['drop_a']:%Y-%m-%d} → {summ['drop_b']:%Y-%m-%d}",
        xy=(summ["drop_b"], summ["drop_b_db"]),
        xytext=(summ["drop_a"], summ["drop_a_db"] - 6),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=9,
        ha="center",
    )
    ax.set_title(f"{meta['id']} — {meta['name']} ({meta['municipio']})", pad=16)
    ax.set_ylabel("gamma-0 (dB), field mean")
    ax.set_xlabel("acquisition date (UTC ~12:05), NISAR track 19 frame 11 ascending")
    style_dates(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_all(df: pd.DataFrame, fields: pd.DataFrame, out) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    hv = df[df["pol"] == "HVHV"]
    for _, meta in fields.iterrows():
        d = hv[hv["field_id"] == meta["id"]].sort_values("date")
        label = f"{meta['id']} {meta['name']}"
        ax.plot(d["date"], d["mean_db"], marker="o", markersize=4, label=label)
    zafra_lines(ax)
    ax.set_title("HV (cross-pol) field mean, 2025/26 zafra", pad=16)
    ax.set_ylabel("HV gamma-0 (dB)")
    ax.set_xlabel("acquisition date (UTC)")
    style_dates(ax)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fields", default=str(FIELDS_DIR / "fields.geojson"))
    p.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = p.parse_args()
    t0 = time.perf_counter()

    scenes = ts.list_scenes(NISAR_DIR)
    if not scenes:
        sys.exit(f"no GCOV scenes found in {NISAR_DIR}")
    fields = ts.load_fields(args.fields)
    print(f"{len(scenes)} scenes {scenes[0]['date']} .. {scenes[-1]['date']}, {len(fields)} fields")

    df, cached = ts.load_or_build(scenes, args.fields, OUT_DIR, force=args.force)
    parquet, csv = ts.cache_paths(OUT_DIR, len(fields), len(scenes))
    print(f"series: {len(df)} rows ({'loaded from cache' if cached else 'computed'}) -> {parquet}")

    # --- per-field summary + plots ---------------------------------------------------------
    print(f"\n{'field':<9} {'HV first':>9} {'HV last':>8} {'min (date)':>21} {'largest drop':>36}")
    pngs, table = [], []
    for _, meta in fields.iterrows():
        d = df[df["field_id"] == meta["id"]]
        s = summarize(d[d["pol"] == "HVHV"])
        print(
            f"{meta['id']:<9} {s['first']:>9.2f} {s['last']:>8.2f} "
            f"{s['min']:>8.2f} ({s['min_date']:%Y-%m-%d}) "
            f"{s['drop']:>+8.2f} dB ({s['drop_a']:%Y-%m-%d} -> {s['drop_b']:%Y-%m-%d})"
        )
        out = OUT_DIR / f"{meta['id']}.png"
        plot_field(d, meta, s, out)
        pngs.append(out)
        table.append((meta, s))

    out_all = OUT_DIR / "all_fields_HVHV.png"
    plot_all(df, fields, out_all)
    pngs.append(out_all)

    # --- markdown table ----------------------------------------------------------------------
    print(
        "\n| field_id | name | municipio | HVHV first | HVHV min (date) | HVHV last "
        "| largest drop dB (date_a → date_b) |"
    )
    print("|---|---|---|---|---|---|---|")
    for meta, s in table:
        print(
            f"| {meta['id']} | {meta['name']} | {meta['municipio']} | {s['first']:.2f} | "
            f"{s['min']:.2f} ({s['min_date']:%Y-%m-%d}) | {s['last']:.2f} | "
            f"{s['drop']:+.2f} ({s['drop_a']:%Y-%m-%d} → {s['drop_b']:%Y-%m-%d}) |"
        )

    # --- QA: anything that makes a number less trustworthy -----------------------------------
    hv = df[df["pol"] == "HVHV"]
    issues = hv[(hv["valid_px"] < hv["inside_px"]) | hv["clipped"]]
    print("\nQA: field/date rows with clipped windows or mask-rejected pixels:", len(issues))
    for _, r in issues.iterrows():
        print(
            f"  {r['field_id']} {r['date']:%Y-%m-%d}: valid {r['valid_px']}/{r['inside_px']} px, "
            f"outside swath {r['outside_swath_px']}, invalid {r['invalid_px']}, "
            f"clipped {r['clipped']}"
        )

    print("\nfigures:")
    for png in pngs:
        print(f"  {png}")
    print(f"series csv: {csv}")
    print(f"total runtime: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
