"""Exercise 6: check the radar harvest windows against Sentinel-2 NDVI (an independent sensor).

    python scripts/06_optical_validation.py [--days 20] [--max-field-cloud 10] [--no-thumbs]

For each field: pull the Sentinel-2 L2A NDVI series for the season (Earth Engine), pick the
clearest scene within --days before and after each harvest window, and give a verdict:

  confirmed     NDVI_before >= 0.6, NDVI_after <= 0.35 and the drop is >= 0.3
                (green canopy before, bare field after: the cut is inside the window)
  refuted       NDVI_after >= 0.6 (canopy still standing after the window: harvest later),
                or NDVI_before <= 0.35 (field already bare before the window: harvest earlier)
  inconclusive  no clear scene on one side, or intermediate values

Windows tested: the detector's (ratio test, exercise 4) and, when it differs, the reference
window (only field_06, where a FIRMS burn on 02-23/24 points to a later cut).

Also finds the NDVI "step": the first clear scene with NDVI <= 0.35 after one >= 0.6, and
compares it with the radar window. Writes plots and thumbnails to outputs/optical/, a CSV,
and the results section of docs/VALIDATION.md. See src/zafra/optical.py for what NDVI is
and why the per-pixel cloud mask matters.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import io

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zafra import fires as F
from zafra import optical as O
from zafra.paths import FIELDS_DIR, FIRMS_DIR, OUTPUTS_DIR, ROOT

OUT_DIR = OUTPUTS_DIR / "optical"
DETECTIONS_CSV = OUTPUTS_DIR / "detect" / "detections.csv"
DOC = ROOT / "docs" / "VALIDATION.md"

SEASON_START, SEASON_END = "2025-11-01", "2026-06-01"  # end exclusive -> through 05-31

# Verdict thresholds (NDVI). Mature cane canopy sits ~0.7-0.85 here; bare/stubble ~0.1-0.3.
HIGH, LOW, MIN_DROP = 0.6, 0.35, 0.3

FIRE_BUFFER_M = 500  # same buffer as exercise 5

SECTION_START = "<!-- optical-validation:start -->"
SECTION_END = "<!-- optical-validation:end -->"


# --------------------------------------------------------------------------- windows


def load_windows() -> pd.DataFrame:
    """One row per (field, window) to test: detector window, plus reference if different."""
    det = pd.read_csv(DETECTIONS_CSV)
    rows = []
    for _, r in det.iterrows():
        dw = (r["ratio_date_before"], r["ratio_date_detected"])
        rows.append({"field_id": r["field_id"], "kind": "detector", "start": dw[0], "end": dw[1]})
        ref = str(r["reference"])
        if "→" in ref:
            rw = tuple(s.strip() for s in ref.split("→"))
            if rw != dw:
                rows.append(
                    {"field_id": r["field_id"], "kind": "reference", "start": rw[0], "end": rw[1]}
                )
    out = pd.DataFrame(rows)
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])
    return out


# --------------------------------------------------------------------------- analysis


def ndvi_step(series: pd.DataFrame, max_field_cloud: float):
    """First clear scene with NDVI <= LOW that follows a clear scene with NDVI >= HIGH.

    Returns (last_high_date, step_date) or (None, None). The cut happened somewhere between
    the two dates: that pair is the optical "harvest window", analogous to the radar one.
    Using only clear scenes matters: a single cloudy date would otherwise fake the step.
    """
    clear = series[(series["cloud_pct"] <= max_field_cloud) & series["ndvi_mean"].notna()]
    last_high = None
    for _, r in clear.iterrows():
        if r["ndvi_mean"] >= HIGH:
            last_high = r["date"]
        elif r["ndvi_mean"] <= LOW and last_high is not None:
            return last_high, r["date"]
    return None, None


def verdict(before, after) -> tuple[str, str]:
    """Apply the confirmed/refuted/inconclusive rule. Returns (verdict, reason)."""
    if before is None and after is None:
        return "inconclusive", "no clear scene before or after"
    if before is None:
        return "inconclusive", "no clear scene before"
    if after is None:
        return "inconclusive", "no clear scene after"
    b, a = before["ndvi_mean"], after["ndvi_mean"]
    if b >= HIGH and a <= LOW and b - a >= MIN_DROP:
        return "confirmed", "green before, bare after"
    if a >= HIGH:
        return "refuted", "canopy still standing after window"
    if b <= LOW:
        return "refuted", "already bare before window"
    if a <= LOW and b - a >= MIN_DROP:
        # The common borderline case: cane dries (and may be burned) in the weeks before
        # the cut, so NDVI before is already below 0.6 even though the field is standing.
        return "inconclusive", f"bare after, but NDVI before {b:.2f} < {HIGH} (drying cane?)"
    return "inconclusive", "intermediate NDVI"


def position(step, ws, we) -> tuple[str, int | None]:
    """Where the NDVI step date falls relative to the radar window, and by how many days."""
    if step is None:
        return "n/a", None
    if step < ws:
        return "before", (step - ws).days
    if step > we:
        return "after", (step - we).days
    return "inside", 0


# --------------------------------------------------------------------------- outputs


def plot_series(fid, series, windows, picks, fires_local, max_field_cloud, path):
    """NDVI vs date: clear scenes filled, cloudy hollow; radar windows shaded; fires in red."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    clear = series["cloud_pct"] <= max_field_cloud
    ax.plot(
        series.loc[clear, "date"],
        series.loc[clear, "ndvi_mean"],
        "-o",
        color="tab:green",
        ms=4,
        lw=1.2,
        zorder=3,
    )
    ax.plot(
        series.loc[~clear, "date"],
        series.loc[~clear, "ndvi_mean"],
        "o",
        mfc="none",
        mec="0.6",
        ms=4,
        zorder=2,
    )

    colors = {"detector": "tab:blue", "reference": "tab:orange"}
    for _, w in windows.iterrows():
        ax.axvspan(w["start"], w["end"], color=colors[w["kind"]], alpha=0.18, zorder=1)
    for d in fires_local:
        ax.axvline(d, color="tab:red", lw=1, alpha=0.7, zorder=1)
    for p in picks:
        ax.plot(p["date"], p["ndvi_mean"], "s", mfc="none", mec="k", ms=10, mew=1.4, zorder=4)

    ax.axhline(HIGH, color="0.5", ls="--", lw=0.8)
    ax.axhline(LOW, color="0.5", ls=":", lw=0.8)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("NDVI (field mean)")
    ax.set_title(f"{fid}: Sentinel-2 NDVI vs radar harvest window")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3)

    handles = [
        Line2D(
            [],
            [],
            color="tab:green",
            marker="o",
            ms=4,
            label=f"clear (≤{max_field_cloud:g}% cloud in field)",
        ),
        Line2D([], [], color="0.6", marker="o", mfc="none", ls="", ms=4, label="cloudy (not used)"),
        Line2D([], [], color="k", marker="s", mfc="none", ls="", ms=8, label="before/after scene"),
        Line2D([], [], color="0.5", ls="--", label=f"{HIGH} standing / {LOW} bare"),
    ]
    for kind in windows["kind"].unique():
        handles.append(Patch(color=colors[kind], alpha=0.3, label=f"radar window ({kind})"))
    if fires_local:
        handles.append(Line2D([], [], color="tab:red", label=f"VIIRS fire ≤{FIRE_BUFFER_M} m"))
    ax.legend(handles=handles, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fetch_png(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return plt.imread(io.BytesIO(r.content), format="png")


def save_thumbnails(fid, geom, rows, path) -> str | None:
    """One row of (before, after) true-colour panels per tested window. Returns a note on
    failure (thumbnails are a nice-to-have; the verdict does not depend on them)."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)), squeeze=False)
    try:
        for i, row in enumerate(rows):
            for j, side in enumerate(("before", "after")):
                ax = axes[i][j]
                ax.axis("off")
                pick = row[f"_{side}"]
                if pick is None:
                    ax.set_title(f"{row['kind']} {side}: no clear scene", fontsize=9)
                    continue
                ax.imshow(fetch_png(O.rgb_thumb_url(pick["image_id"], geom)))
                ax.set_title(
                    f"{row['kind']} {side}: {pick['date']:%Y-%m-%d}  NDVI {pick['ndvi_mean']:.2f}",
                    fontsize=9,
                )
    except Exception as e:  # network or EE quota: skip, but say why
        plt.close(fig)
        return f"thumbnails skipped ({type(e).__name__})"
    fig.suptitle(f"{fid}: Sentinel-2 true colour (B4,B3,B2), field outline in yellow")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return None


def fmt_pick(p) -> str:
    return "—" if p is None else f"{p['ndvi_mean']:.2f} ({p['date']:%m-%d})"


def markdown_table(res: pd.DataFrame) -> str:
    lines = [
        "| field | radar window | NDVI before (date) | NDVI after (date) | NDVI step date "
        "| verdict | offset vs radar (days) | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        off = (
            "—"
            if pd.isna(r["offset_days"])
            else f"{int(r['offset_days']):+d} ({r['step_vs_radar']})"
        )
        step = "—" if pd.isna(r["step_date"]) else f"{r['step_date']:%m-%d}"
        lines.append(
            f"| {r['field_id']} | {r['kind']} {r['window_start']:%m-%d}→{r['window_end']:%m-%d} "
            f"| {r['before_txt']} | {r['after_txt']} | {step} | **{r['verdict']}** | {off} "
            f"| {r['notes']} |"
        )
    return "\n".join(lines)


METHOD_MD = f"""\
### Method

* **Data:** Sentinel-2 L2A surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`, Earth Engine),
  {SEASON_START} .. 2026-05-31, tile 14QMF, scenes with tile cloud ≤ 30 %. Code:
  `src/zafra/optical.py`, driver `scripts/06_optical_validation.py`.
* **NDVI** = (B8 − B4) / (B8 + B4) per 10 m pixel. Pixels are masked per pixel with the
  Scene Classification Layer (kept: vegetation, bare soil, water, unclassified; dropped:
  cloud, cloud shadow, cirrus, dark/shadow, saturated, no data). Field value = unweighted
  mean over pixels whose centre lies in the polygon (same rule as the NISAR pixels).
* **Clear scene:** ≤ 10 % of the field's pixels masked. Same-day duplicates: keep the one
  with most clear pixels.
* **Before/after:** clearest clear scene within 20 days before the window's first radar date
  and within 20 days after its last (ties: closest to the window). Scenes inside the window
  are not used: the cut happened at an unknown time in there.
* **Verdict:** *confirmed* if NDVI_before ≥ {HIGH}, NDVI_after ≤ {LOW} and the drop ≥ {MIN_DROP};
  *refuted* if NDVI_after ≥ {HIGH} (still standing) or NDVI_before ≤ {LOW} (already bare);
  otherwise *inconclusive*.
* **NDVI step date:** first clear scene with NDVI ≤ {LOW} after a clear scene ≥ {HIGH}.
  Offset = days from the radar window to the step (0 = inside; negative = optical step
  before the window starts; positive = after it ends).
* **Why this is independent evidence:** NDVI measures leaf chlorophyll/structure in reflected
  sunlight (665 and 842 nm); the radar measures L-band (24 cm) microwave volume scattering
  from the cane. Different physics, platform, orbit, time of day (S2 ~11:00 local, NISAR
  ~06:00 local) and processing, so their failure modes (clouds/smoke vs. soil moisture,
  lodging) are largely unrelated.
"""

LIMITS_MD = """\
### Limitations

* **Cloud gaps.** The dry season is mostly clear, but a single cloudy or hazy week can leave
  no clear scene inside ±20 days. SCL misses some thin haze and smoke (common during the
  burning season), which lowers NDVI without being masked; a lone low value next to high
  ones should be treated with suspicion.
* **NDVI saturation.** Over a closed cane canopy NDVI flattens at ~0.8 and does not see the
  stalks, only the top leaves. It cannot tell a ripe field from a younger one, and drying
  or burned-but-standing cane already reads lower (0.4-0.6) before the cut. That makes the
  ≥ 0.6 "before" threshold strict for fields that were dried or burned before harvest.
* **Regrowth.** Ratoon cane re-sprouts within weeks; NDVI climbs back above 0.35 after
  ~1-3 months, so a late "after" scene can miss the bare phase.
* **Grids.** NISAR GCOV and Sentinel-2 are both 10 m UTM grids but not co-registered: their
  pixel edges do not line up, and SCL is 20 m (resampled by nearest neighbour). The same
  polygon samples slightly different ground in each sensor.
* **Mixed pixels at field edges.** A 10 m pixel on the boundary mixes the field with roads,
  furrows or a neighbouring field in a different state. In small fields (field_04 ~0.8 ha,
  ~80 pixels) edge pixels are a large share of the mean and can smear a partial harvest.
* **Partial harvests.** A field cut over several days, or in parts, gives an NDVI ramp over
  2-3 scenes, not a step; the step date is then the date the *average* crossed 0.35.
"""


def write_doc(table_md: str, could_not: list[str]) -> None:
    """Create docs/VALIDATION.md if needed and (re)write the optical section in place, so
    re-running the script updates the section instead of appending a duplicate."""
    cnv = "\n".join(f"* {c}" for c in could_not) or "* (none)"
    section = (
        f"{SECTION_START}\n## Optical validation with Sentinel-2 NDVI (exercise 6)\n\n"
        f"Generated by `scripts/06_optical_validation.py`.\n\n"
        f"{table_md}\n\n**Could not verify:**\n\n{cnv}\n\n{METHOD_MD}\n{LIMITS_MD}{SECTION_END}\n"
    )
    if DOC.exists():
        text = DOC.read_text()
    else:
        text = (
            "# Validation of the radar harvest detector\n\n"
            "Independent checks of the NISAR harvest windows (`outputs/detect/detections.csv`).\n\n"
        )
    if SECTION_START in text and SECTION_END in text:
        a = text.index(SECTION_START)
        b = text.index(SECTION_END) + len(SECTION_END) + 1
        text = text[:a] + section + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + section
    DOC.write_text(text)


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=20, help="search ±days around the window")
    ap.add_argument(
        "--max-field-cloud",
        type=float,
        default=10,
        help="max %% of field pixels masked for a scene to count as clear",
    )
    ap.add_argument("--no-thumbs", action="store_true", help="skip true-colour thumbnails")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = gpd.read_file(FIELDS_DIR / "fields.geojson").set_index("id")
    windows = load_windows()

    # FIRMS is optional context for the plots: VIIRS hot pixels near each field.
    try:
        fires = F.load_firms(FIRMS_DIR)
        fires = fires[
            (fires["acq_local"] >= pd.Timestamp(SEASON_START, tz=F.LOCAL_TZ))
            & (fires["acq_local"] < pd.Timestamp(SEASON_END, tz=F.LOCAL_TZ))
        ]
    except FileNotFoundError:
        fires = None
        print("note: no FIRMS CSVs found, plots will have no fire lines")

    results, plot_paths, could_not = [], [], []
    for fid, fw in windows.groupby("field_id", sort=True):
        geom = fields.loc[fid, "geometry"]
        print(f"{fid}: querying Sentinel-2 ...", file=sys.stderr)
        series = O.s2_ndvi_series(geom, SEASON_START, SEASON_END, max_cloud=30)
        n_clear = int((series["cloud_pct"] <= args.max_field_cloud).sum())
        last_high, step = ndvi_step(series, args.max_field_cloud)

        rows = []
        for _, w in fw.iterrows():
            before, after = O.clearest_before_after(
                series, w["start"], w["end"], days=args.days, max_field_cloud=args.max_field_cloud
            )
            v, reason = verdict(before, after)
            where, off = position(step, w["start"], w["end"])
            notes = [reason]
            if step is not None:
                notes.append(f"optical cut between {last_high:%m-%d} and {step:%m-%d}")
                # The step date is the first *bare* scene; the cut is somewhere after the
                # last green one. If that bracket overlaps the radar window they agree even
                # when the step date itself lands a few days after the window.
                if where != "inside" and last_high <= w["end"] and step >= w["start"]:
                    notes.append("optical bracket overlaps radar window")
            else:
                notes.append("no ≥0.6 → ≤0.35 step in season")
            if before is None or after is None:
                missing = " and ".join(
                    s for s, p in (("before", before), ("after", after)) if p is None
                )
                could_not.append(
                    f"{fid} ({w['kind']} {w['start']:%m-%d}→{w['end']:%m-%d}): no clear scene "
                    f"{missing} within ±{args.days} d"
                )
            elif v == "inconclusive":
                could_not.append(f"{fid} ({w['kind']}): {reason}")
            rows.append(
                {
                    "field_id": fid,
                    "kind": w["kind"],
                    "window_start": w["start"],
                    "window_end": w["end"],
                    "before_date": None if before is None else before["date"],
                    "ndvi_before": None if before is None else round(before["ndvi_mean"], 3),
                    "after_date": None if after is None else after["date"],
                    "ndvi_after": None if after is None else round(after["ndvi_mean"], 3),
                    "drop": None
                    if before is None or after is None
                    else round(before["ndvi_mean"] - after["ndvi_mean"], 3),
                    "last_high_date": last_high,
                    "step_date": step,
                    "step_vs_radar": where,
                    "offset_days": off,
                    "verdict": v,
                    "notes": "; ".join(notes),
                    "n_clear_scenes": n_clear,
                    "before_txt": fmt_pick(before),
                    "after_txt": fmt_pick(after),
                    "_before": before,
                    "_after": after,
                }
            )

        fires_local = []
        if fires is not None:
            near = F.fires_near_field(fires, geom, buffer_m=FIRE_BUFFER_M)
            fires_local = sorted(near["acq_local"].dt.tz_localize(None).dt.normalize().unique())
        picks = [p for r in rows for p in (r["_before"], r["_after"]) if p is not None]
        png = OUT_DIR / f"{fid}_ndvi.png"
        plot_series(fid, series, fw, picks, fires_local, args.max_field_cloud, png)
        plot_paths.append(png)

        if not args.no_thumbs:
            thumb = OUT_DIR / f"{fid}_rgb_before_after.png"
            note = save_thumbnails(fid, geom, rows, thumb)
            if note:
                print(f"{fid}: {note}", file=sys.stderr)
                for r in rows:
                    r["notes"] += f"; {note}"
            else:
                plot_paths.append(thumb)
        results.extend(rows)

    res = pd.DataFrame(results)
    res.drop(columns=["_before", "_after", "before_txt", "after_txt"]).to_csv(
        OUT_DIR / "validation.csv", index=False, date_format="%Y-%m-%d"
    )
    table = markdown_table(res)
    write_doc(table, could_not)

    print("\n## Optical validation (Sentinel-2 NDVI)\n")
    print(table)
    print("\n### Could not verify\n")
    print("\n".join(f"- {c}" for c in could_not) or "- (none)")
    print("\n### Files\n")
    for p in [*plot_paths, OUT_DIR / "validation.csv", DOC]:
        print(f"- {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
