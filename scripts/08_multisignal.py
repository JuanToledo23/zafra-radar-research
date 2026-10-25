"""Multi-signal NISAR series (GCOV asc, GCOV desc, GUNW coherence) for the 37 fields.

    python scripts/08_multisignal.py [--fields data/fields/fields_v2.geojson] [--force]

Builds (or loads from cache) one table of every NISAR signal per field per date, the input
of the harvest classifier (exercise 4b). Writes to outputs/multisignal/:
  gcov_asc.parquet, gcov_desc.parquet, coherence.parquet   per-source caches
  series_long.{parquet,csv}, series_wide.{parquet,csv}     the feature table
  field_<id>.png                                           3 panels per field
  overview_coherence.png                                   coherence of all cane fields

Sanity checks printed at the end:
  1. HV ascending from this module on the 6 original polygons (data/fields/fields.geojson)
     matches outputs/timeseries/series_6fields_15scenes.csv within 0.01 dB.
  2. For cane fields, a pair touching the harvest window has coherence < 0.3.
"""

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from zafra import multisignal as ms
from zafra import timeseries as ts
from zafra.paths import FIELDS_DIR, NISAR_DIR, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "multisignal"
ASC_DIR = NISAR_DIR
DESC_DIR = NISAR_DIR / "GCOV_DESCENDING_t113f79"
GUNW_DIR = NISAR_DIR / "GUNW_ASCENDING_t19f11"
ORIG_FIELDS = FIELDS_DIR / "fields.geojson"
REF_CSV = OUTPUTS_DIR / "timeseries" / "series_6fields_15scenes.csv"

COH_LOW = 0.3  # "coherence collapsed" threshold of the sanity check
# A pair "touches" the harvest window if its 12-day interval overlaps the window widened by
# one revisit on each side: the NDVI window is itself only known to ~5 days.
WINDOW_PAD = pd.Timedelta(days=12)
ORBIT_STYLE = {"asc": ("ascending t19", "tab:blue"), "desc": ("descending t113", "tab:orange")}


# --- cached extraction --------------------------------------------------------------------


def cached(path, fields, dates, build, date_col, force):
    """Load `path` if it holds exactly these field ids and dates, else build and save it."""
    if path.exists() and not force:
        df = pd.read_parquet(path)
        if set(df["field_id"]) == set(fields["id"]) and set(
            df[date_col].dt.date.astype(str)
        ) == set(dates):
            return df, True
        print(f"cache {path.name} does not match current fields/scenes; rebuilding")
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df, False


# --- plots --------------------------------------------------------------------------------


def style_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(alpha=0.3)


def shade_window(ax, meta) -> None:
    if pd.notna(meta["harvest_window_start"]):
        ax.axvspan(
            meta["harvest_window_start"],
            meta["harvest_window_end"] + pd.Timedelta(days=1),
            color="tab:red",
            alpha=0.15,
            linewidth=0,
            label="harvest window (NDVI)",
        )


def plot_field(meta, g: pd.DataFrame, c: pd.DataFrame, out) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for orbit, (label, color) in ORBIT_STYLE.items():
        d = g[g["orbit"] == orbit].sort_values("date")
        axes[0].plot(d["date"], d["HVHV_db"], marker="o", color=color, label=label)
        axes[1].plot(d["date"], d["ratio_db"], marker="o", color=color, label=label)
    axes[0].set_ylabel("HV gamma-0 (dB)")
    axes[1].set_ylabel("HV - HH (dB)\ncross-ratio")

    # Coherence belongs to an interval, not a day: draw each pair as a segment from its
    # reference to its secondary date, with a marker at the midpoint.
    c = c.sort_values("ref_date")
    mid = c["ref_date"] + (c["sec_date"] - c["ref_date"]) / 2
    axes[2].hlines(c["coh_mean"], c["ref_date"], c["sec_date"], color="tab:green", linewidth=2)
    axes[2].plot(mid, c["coh_mean"], "o", color="tab:green", label="mean coherence (asc pair)")
    axes[2].axhline(COH_LOW, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("HH coherence\n12-day pair")
    axes[2].set_xlabel("acquisition date (UTC)")

    for ax in axes:
        shade_window(ax, meta)
        style_dates(ax)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[2].legend(loc="upper left", fontsize=8)
    window = (
        f"window {meta['harvest_window_start']:%Y-%m-%d} .. {meta['harvest_window_end']:%Y-%m-%d}"
        if pd.notna(meta["harvest_window_start"])
        else "no harvest window"
    )
    fig.suptitle(
        f"{meta['id']} — label: {meta['label']} (hyp: {meta['hyp']}), "
        f"{meta['area_ha']:.1f} ha, {window}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_overview(fields, coh, out) -> None:
    """All cane fields' coherence on one axis, coloured by when NDVI says they were cut."""
    cane = fields[(fields["label"] == "cana") & fields["harvest_window_start"].notna()]
    t = cane["harvest_window_start"]
    norm = matplotlib.colors.Normalize(mdates.date2num(t.min()), mdates.date2num(t.max()))
    cmap = matplotlib.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, meta in cane.iterrows():
        d = coh[coh["field_id"] == meta["id"]].sort_values("ref_date")
        mid = d["ref_date"] + (d["sec_date"] - d["ref_date"]) / 2
        color = cmap(norm(mdates.date2num(meta["harvest_window_start"])))
        ax.plot(mid, d["coh_mean"], marker="o", markersize=3, linewidth=1, color=color, alpha=0.8)
    ax.axhline(COH_LOW, color="0.4", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[0], COH_LOW, f" {COH_LOW}", va="bottom", fontsize=8, color="0.4")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("harvest window start (NDVI)")
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_ylim(0, 1)
    ax.set_title(f"Mean HH coherence of {len(cane)} cane fields, 12-day ascending pairs (t19 f11)")
    ax.set_ylabel("coherence (plotted at pair midpoint)")
    ax.set_xlabel("date (UTC)")
    style_dates(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --- sanity checks ------------------------------------------------------------------------


def check_hv_vs_reference(force: bool) -> None:
    """Check 1: this module reproduces exercise 3's HV ascending numbers."""
    print(
        "\n[check 1] HV asc on the 6 original polygons vs "
        f"{REF_CSV.relative_to(OUTPUTS_DIR.parent)}"
    )
    if not REF_CSV.exists():
        print("  reference csv missing: NOT VERIFIED")
        return
    orig = ts.load_fields(ORIG_FIELDS)
    scenes = ts.list_scenes(ASC_DIR)
    mine, _ = cached(
        OUT_DIR / "check_gcov_asc_6orig.parquet",
        orig,
        [s["date"] for s in scenes],
        lambda: ms.extract_gcov_series(ASC_DIR, ORIG_FIELDS, "asc"),
        "date",
        force,
    )
    ref = pd.read_csv(REF_CSV, parse_dates=["date"])
    ref = ref[ref["pol"] == "HVHV"][["field_id", "date", "mean_db"]]
    m = ref.merge(mine[["field_id", "date", "HVHV_db"]], on=["field_id", "date"], how="outer")
    diff = (m["HVHV_db"] - m["mean_db"]).abs()
    unmatched = int(m[["HVHV_db", "mean_db"]].isna().any(axis=1).sum())
    for fid, d in m.assign(diff=diff).groupby("field_id"):
        print(f"  {fid}: {len(d)} dates, max |diff| {d['diff'].max():.2e} dB")
    ok = unmatched == 0 and diff.max() <= 0.01
    print(
        f"  rows {len(m)}, unmatched {unmatched}, max |diff| {diff.max():.2e} dB -> "
        f"{'PASS' if ok else 'FAIL'} (tolerance 0.01 dB)"
    )


def check_coherence_drop(fields, coh) -> pd.DataFrame:
    """Check 2: for each cane field, lowest coherence of pairs touching the harvest window."""
    cane = fields[(fields["label"] == "cana") & fields["harvest_window_start"].notna()]
    rows = []
    for _, meta in cane.iterrows():
        d = coh[coh["field_id"] == meta["id"]]
        lo = meta["harvest_window_start"] - WINDOW_PAD
        hi = meta["harvest_window_end"] + WINDOW_PAD
        near = (d["sec_date"] >= lo) & (d["ref_date"] <= hi)
        i_min = d.loc[near, "coh_mean"].idxmin()
        rows.append(
            {
                "field_id": meta["id"],
                "near_min": d.loc[i_min, "coh_mean"],
                "near_pair": f"{d.loc[i_min, 'ref_date']:%m-%d}/{d.loc[i_min, 'sec_date']:%m-%d}",
                # How coherent the field is on the other pairs: if it is also < 0.3 there,
                # a low value near the window says little.
                "other_median": d.loc[~near, "coh_mean"].median(),
                "is_global_min": i_min == d["coh_mean"].idxmin(),
            }
        )
    r = pd.DataFrame(rows)
    r["drop"] = r["near_min"] < COH_LOW
    print(
        f"\n[check 2] cane fields with a pair touching the harvest window (+/- 12 d) "
        f"below {COH_LOW}: {int(r['drop'].sum())}/{len(r)}"
    )
    print(
        "  of these, the field's lowest pair of the season: "
        f"{int(r['is_global_min'].sum())}/{len(r)}; "
        f"fields whose *other* pairs have median < {COH_LOW}: "
        f"{int((r['other_median'] < COH_LOW).sum())}/{len(r)}"
    )
    print(f"  {'field':<6} {'min near':>8} {'pair':>11} {'median other':>12} {'lowest?':>7}")
    for _, x in r.iterrows():
        print(
            f"  {x['field_id']:<6} {x['near_min']:>8.3f} {x['near_pair']:>11} "
            f"{x['other_median']:>12.3f} {'yes' if x['is_global_min'] else 'no':>7}"
        )
    return r


# --- main ---------------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fields", default=str(FIELDS_DIR / "fields_v2.geojson"))
    p.add_argument("--force", action="store_true", help="recompute even if the caches exist")
    args = p.parse_args()
    t0 = time.perf_counter()

    fields = ts.load_fields(args.fields)
    for col in ("harvest_window_start", "harvest_window_end"):
        fields[col] = pd.to_datetime(fields[col])
    asc, desc, pairs = ts.list_scenes(ASC_DIR), ts.list_scenes(DESC_DIR), ms.list_pairs(GUNW_DIR)
    print(
        f"{len(fields)} fields; GCOV asc {len(asc)} scenes, desc {len(desc)} scenes, "
        f"GUNW {len(pairs)} pairs"
    )

    g_asc, c1 = cached(
        OUT_DIR / "gcov_asc.parquet",
        fields,
        [s["date"] for s in asc],
        lambda: ms.extract_gcov_series(ASC_DIR, args.fields, "asc"),
        "date",
        args.force,
    )
    g_desc, c2 = cached(
        OUT_DIR / "gcov_desc.parquet",
        fields,
        [s["date"] for s in desc],
        lambda: ms.extract_gcov_series(DESC_DIR, args.fields, "desc"),
        "date",
        args.force,
    )
    coh, c3 = cached(
        OUT_DIR / "coherence.parquet",
        fields,
        [p["sec_date"] for p in pairs],
        lambda: ms.extract_coherence_series(GUNW_DIR, fields),
        "sec_date",
        args.force,
    )
    print(
        "sources: "
        + ", ".join(
            f"{n} {'cache' if c else 'computed'}"
            for n, c in (("asc", c1), ("desc", c2), ("coherence", c3))
        )
    )
    long, wide = ms.build_feature_table([g_asc, g_desc], coh, OUT_DIR)
    print(f"long table {len(long)} rows, wide table {len(wide)} rows -> {OUT_DIR}")

    # --- per-field summary + plots ---------------------------------------------------------
    g = pd.concat([g_asc, g_desc], ignore_index=True)
    print(
        "\n| field | label | n asc | n desc | n coh | HV asc min..max dB | "
        "ratio asc min..max dB | coh min..max |"
    )
    print("|---|---|---|---|---|---|---|---|")
    pngs = []
    for _, meta in fields.iterrows():
        gf, cf = g[g["field_id"] == meta["id"]], coh[coh["field_id"] == meta["id"]]
        ga = gf[gf["orbit"] == "asc"]
        n = lambda d: int(d.notna().sum())  # noqa: E731 - valid (non-NaN) values only
        print(
            f"| {meta['id']} | {meta['label']} | {n(ga['HVHV_db'])} | "
            f"{n(gf.loc[gf['orbit'] == 'desc', 'HVHV_db'])} | {n(cf['coh_mean'])} | "
            f"{ga['HVHV_db'].min():.1f}..{ga['HVHV_db'].max():.1f} | "
            f"{ga['ratio_db'].min():.1f}..{ga['ratio_db'].max():.1f} | "
            f"{cf['coh_mean'].min():.2f}..{cf['coh_mean'].max():.2f} |"
        )
        out = OUT_DIR / f"field_{meta['id']}.png"
        plot_field(meta, gf, cf, out)
        pngs.append(out)
    out_all = OUT_DIR / "overview_coherence.png"
    plot_overview(fields, coh, out_all)

    # --- QA ----------------------------------------------------------------------------------
    gq = g[(g["valid_px"] < g["inside_px"]) | g["clipped"]]
    cq = coh[(coh["valid_px"] < coh["inside_px"]) | coh["clipped"]]
    print(f"\nQA: GCOV field/date rows with clipped windows or rejected pixels: {len(gq)}")
    for _, r in gq.iterrows():
        print(
            f"  {r['field_id']} {r['orbit']} {r['date']:%Y-%m-%d}: "
            f"valid {r['valid_px']}/{r['inside_px']} px, clipped {r['clipped']}"
        )
    print(f"QA: coherence field/pair rows with NaN pixels or clipped windows: {len(cq)}")
    for _, r in cq.iterrows():
        print(
            f"  {r['field_id']} {r['ref_date']:%Y-%m-%d}/{r['sec_date']:%Y-%m-%d}: "
            f"valid {r['valid_px']}/{r['inside_px']} px, clipped {r['clipped']}"
        )
    flagged = coh["mask_flagged_px"].sum()
    print(
        f"QA: coherence pixels inside polygons flagged by the GUNW mask: {flagged} "
        f"of {coh['inside_px'].sum()} (kept unless NaN)"
    )

    # --- sanity checks -----------------------------------------------------------------------
    check_hv_vs_reference(args.force)
    check_coherence_drop(fields, coh)

    print("\nfigures:")
    for png in pngs[:3]:
        print(f"  {png}")
    print(f"  ... {len(pngs)} field_<id>.png in {OUT_DIR}")
    print(f"  {out_all}")
    print(f"total runtime: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
