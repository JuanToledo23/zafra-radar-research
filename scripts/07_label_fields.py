"""Label the 37 candidate polygons as sugarcane / not sugarcane from their NDVI phenology.

    python scripts/07_label_fields.py [--max-field-cloud 10] [--from-cache]

Input: data/fields/zafra_fields_labeled.geojson (37 polygons; `hyp` is the author's guess,
kept only for comparison; `suggested` is a failed earlier rule and is ignored).

For each polygon: Sentinel-2 NDVI series 2024-10-01 .. 2026-06-30 (src/zafra/optical.py),
then a "cane signature" test on the clear scenes, and for cane fields the 2025/26 harvest
window. Writes:
  data/fields/fields_v2.geojson      all 37 polygons, id, hyp, label, area_ha, window
  data/labels/fields_v2_labels.csv   same, tabular, with the diagnostics
  outputs/labels/ndvi_series.csv     raw per-date series (cache for --from-cache)
  outputs/labels/ndvi_grid.png       37 NDVI panels, Jun-Oct shaded

Why NDVI phenology separates cane from the other land covers here
-----------------------------------------------------------------
Sugarcane in Morelos is a semi-perennial crop: planted once, then cut once a year for
5-7 years (each regrowth is a "ratoon"). The cut happens in the dry season ("zafra",
roughly November to May), and the cane then regrows through the rainy season and stays a
dense, tall, green canopy for 10-14 months. In an NDVI time series that gives a very
specific shape:
  * long plateaus: NDVI >= 0.6 for many months on end (a closed canopy);
  * exactly one deep dip per year (harvest: green canopy -> bare stubble), in Nov-May;
  * a high peak (>= 0.75): mature cane is among the greenest, densest canopies around.
The alternatives look different:
  * annual crops (maize, sorghum, vegetables) grow for 3-5 months, so their green phase
    is short (< 150 days), and they are often cut or harvested in the rainy season
    (Jun-Oct) or cropped twice a year (two dips in one season);
  * pasture and scrub rarely reach a dense canopy (lower max) and brown out every dry
    season gradually rather than collapsing in one step, or are grazed/cut repeatedly;
  * orchards / trees stay green but seldom collapse at all (fails no test except possibly
    the plateau one; such fields end up "dudoso", not "cana", unless they have a dip).
So the three tests below are: one long plateau, at most one collapse per zafra and none
in the rainy season, and a high peak.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from zafra import optical as O
from zafra.paths import FIELDS_DIR, LABELS_DIR, OUTPUTS_DIR, ROOT

OUT_DIR = OUTPUTS_DIR / "labels"
SERIES_CSV = OUT_DIR / "ndvi_series.csv"
IN_GEOJSON = FIELDS_DIR / "zafra_fields_labeled.geojson"
ORIG_GEOJSON = FIELDS_DIR / "fields.geojson"
OUT_GEOJSON = FIELDS_DIR / "fields_v2.geojson"
OUT_CSV = LABELS_DIR / "fields_v2_labels.csv"

START, END = "2024-10-01", "2026-07-01"  # end exclusive -> through 2026-06-30
UTM = "EPSG:32614"  # UTM 14N, metric CRS for areas and overlaps in Morelos

# Scene-level pre-filter (whole 110 km tile). Looser than exercise 6's 30 % because this
# window spans two rainy seasons, where a 30 % tile filter keeps almost nothing; the
# per-field SCL cloud fraction (--max-field-cloud) is what decides if a date is usable.
MAX_TILE_CLOUD = 50

# ---- Cane signature thresholds (NDVI, field mean over clear 10 m pixels) ----
HIGH = 0.6  # closed green canopy
LOW = 0.35  # bare soil / stubble / ash
PEAK = 0.75  # test (c): mature cane reaches a very dense canopy at least once
MIN_RUN_DAYS = 150  # test (a): continuous NDVI >= HIGH for >= 150 days (5 months).
# 150 days is longer than the whole green phase of maize/sorghum (~90-120 d) but much
# shorter than the 8-12 months a ratoon stays closed, so it sits between the two.
HARVEST_HIGH = 0.5  # harvest-window rule (as in the brief): <= LOW after a scene >= 0.5.
# 0.5 instead of 0.6 because cane is often dried or burned before the cut, which already
# lowers NDVI below 0.6 on the last standing scene (see exercise 6 notes).

RAINY_MONTHS = (6, 7, 8, 9, 10)  # Jun-Oct: cane grows; no cane field is cut then
SEASONS = {  # Nov-May harvest seasons ("zafras"), end exclusive
    "2024-25": ("2024-11-01", "2025-06-01"),
    "2025-26": ("2025-11-01", "2026-06-01"),
}
MIN_CLEAR_SCENES = 10  # fewer clear scenes than this over 21 months -> "could not verify"
SMALL_HA = 0.5  # below this, edge pixels dominate the mean -> flagged unreliable


# --------------------------------------------------------------------------- series


def fetch_series(polys: gpd.GeoDataFrame) -> pd.DataFrame:
    """NDVI series for every polygon (Earth Engine), long format with an `id` column."""

    def one(row):
        print(f"{row.id}: querying Sentinel-2 ...", file=sys.stderr)
        s = O.s2_ndvi_series(row.geometry, START, END, max_cloud=MAX_TILE_CLOUD)
        return s.assign(id=row.id)

    O.init_ee()  # once, before the threads start
    # Each call is one round trip to Earth Engine; a few in parallel keep it under a minute.
    with ThreadPoolExecutor(max_workers=6) as ex:
        parts = list(ex.map(one, polys.itertuples()))
    return pd.concat(parts, ignore_index=True)


def clear_scenes(series: pd.DataFrame, max_field_cloud: float) -> pd.DataFrame:
    """Scenes with <= max_field_cloud % of the field masked and a valid NDVI, by date."""
    ok = (series["cloud_pct"] <= max_field_cloud) & series["ndvi_mean"].notna()
    return series[ok].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- tests


def longest_high_run(clear: pd.DataFrame) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Longest run of consecutive clear scenes with NDVI >= HIGH, in days (first->last).

    Only a clear scene below HIGH breaks a run; cloudy dates are skipped, not counted as
    breaks (otherwise every rainy season would split the plateau). The flip side: a long
    cloud gap can hide a dip, which is why fields with few clear scenes are reported.
    """
    best, best_span = 0, (None, None)
    run_start = prev = None
    for _, r in clear.iterrows():
        if r["ndvi_mean"] >= HIGH:
            run_start = run_start if run_start is not None else r["date"]
            prev = r["date"]
            if (prev - run_start).days > best:
                best, best_span = (prev - run_start).days, (run_start, prev)
        else:
            run_start = prev = None
    return best, *best_span


def collapses(clear: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Every collapse: a clear scene >= HIGH followed (later) by the first clear scene <= LOW.

    Returns (last_high_date, low_date) pairs. After a collapse the detector re-arms only
    when NDVI is back >= HIGH, so the weeks of bare stubble after one cut count once, and
    values wobbling between LOW and HIGH (drying cane, haze) never count.
    """
    out, last_high = [], None
    for _, r in clear.iterrows():
        if r["ndvi_mean"] >= HIGH:
            last_high = r["date"]
        elif r["ndvi_mean"] <= LOW and last_high is not None:
            out.append((last_high, r["date"]))
            last_high = None
    return out


def season_of(date: pd.Timestamp) -> str | None:
    """'2024-25' / '2025-26' for a date inside a Nov-May zafra; 'rainy' for Jun-Oct."""
    if date.month in RAINY_MONTHS:
        return "rainy"
    for name, (a, b) in SEASONS.items():
        if pd.Timestamp(a) <= date < pd.Timestamp(b):
            return name
    return None  # cannot happen for dates inside START..END, kept for safety


def harvest_window(clear: pd.DataFrame, season: str = "2025-26"):
    """2025/26 harvest window: first clear scene <= LOW after one >= HARVEST_HIGH, in season.

    Window = (previous clear scene, that scene), as in exercise 6: the cut happened
    somewhere between the last clear look before the drop and the first bare look.
    The ">= 0.5" scene may precede the season (e.g. October), the low one must be inside.
    """
    a, b = (pd.Timestamp(d) for d in SEASONS[season])
    armed = False
    prev_date = None
    for _, r in clear.iterrows():
        if r["ndvi_mean"] >= HARVEST_HIGH:
            armed = True
        elif r["ndvi_mean"] <= LOW and armed:
            if a <= r["date"] < b:
                return prev_date, r["date"]
            armed = False  # a dip outside this season; wait for regrowth
        prev_date = r["date"]
    return None, None


def label_field(clear: pd.DataFrame) -> dict:
    """Apply tests (a), (b), (c) and return the label, reason and diagnostics."""
    run_days, run_a, run_b = longest_high_run(clear)
    ndvi_max = clear["ndvi_mean"].max() if len(clear) else float("nan")
    cols = collapses(clear)
    by_season = {s: 0 for s in SEASONS} | {"rainy": 0}
    for _, low in cols:
        s = season_of(low)
        if s is not None:
            by_season[s] += 1
    rainy_dates = [low for _, low in cols if season_of(low) == "rainy"]

    ok_a = run_days >= MIN_RUN_DAYS
    ok_b = by_season["rainy"] == 0 and all(by_season[s] <= 1 for s in SEASONS)
    ok_c = ndvi_max >= PEAK

    if not ok_b:
        label = "no_cana"
        why = []
        if rainy_dates:
            why.append(
                "collapse in Jun-Oct (" + ", ".join(f"{d:%Y-%m-%d}" for d in rainy_dates) + ")"
            )
        for s in SEASONS:
            if by_season[s] >= 2:
                why.append(f"{by_season[s]} collapses in {s}")
        reason = "; ".join(why)
    elif ok_a and ok_c:
        label = "cana"
        reason = f"plateau {run_days} d, ≤1 collapse/season, max {ndvi_max:.2f}"
    else:
        label = "dudoso"
        why = []
        if not ok_a:
            why.append(f"longest plateau {run_days} d < {MIN_RUN_DAYS}")
        if not ok_c:
            why.append(f"max {ndvi_max:.2f} < {PEAK}")
        reason = "; ".join(why)

    return {
        "label": label,
        "reason": reason,
        "ndvi_max": ndvi_max,
        "run_days": run_days,
        "run_start": run_a,
        "run_end": run_b,
        "n_collapses_2024-25": by_season["2024-25"],
        "n_collapses_2025-26": by_season["2025-26"],
        "n_collapses_rainy": by_season["rainy"],
        "collapse_dates": ", ".join(f"{low:%Y-%m-%d}" for _, low in cols),
    }


# --------------------------------------------------------------------------- id mapping


def map_original_fields(polys: gpd.GeoDataFrame) -> tuple[dict, list[str]]:
    """Match field_01..06 (exercises 1-6) to the polygons by overlap (IoU >= 0.5).

    Returns ({p_id: field_id}, lines describing each original field's match or miss).
    """
    orig = gpd.read_file(ORIG_GEOJSON).to_crs(UTM)
    p = polys.to_crs(UTM)
    mapping, lines = {}, []
    for _, f in orig.iterrows():
        inter = p.geometry.intersection(f.geometry).area
        iou = inter / p.geometry.union(f.geometry).area
        i = iou.idxmax()
        if iou[i] >= 0.5:
            mapping[p.loc[i, "id"]] = f["id"]
            lines.append(f"{f['id']} = {p.loc[i, 'id']} (IoU {iou[i]:.2f})")
        else:
            d = p.geometry.distance(f.geometry)
            j = d.idxmin()
            lines.append(
                f"{f['id']}: no matching polygon (nearest {p.loc[j, 'id']}, {d[j]:.0f} m away)"
            )
    return mapping, lines


# --------------------------------------------------------------------------- outputs


def plot_grid(res: pd.DataFrame, series: pd.DataFrame, max_field_cloud: float, path) -> None:
    """6 x 7 grid of NDVI panels (37 used), rainy season shaded, title = id + label."""
    colors = {"cana": "tab:green", "no_cana": "tab:red", "dudoso": "tab:orange"}
    fig, axes = plt.subplots(7, 6, figsize=(20, 21), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, (_, r) in zip(flat, res.iterrows(), strict=False):
        s = series[series["id"] == r["id"]]
        clear = s["cloud_pct"] <= max_field_cloud
        for y in (2024, 2025, 2026):  # Jun-Oct shading (no cane harvest expected)
            ax.axvspan(pd.Timestamp(f"{y}-06-01"), pd.Timestamp(f"{y}-11-01"), color="0.88", lw=0)
        ax.plot(s.loc[clear, "date"], s.loc[clear, "ndvi_mean"], "-o", color="k", ms=2, lw=0.8)
        ax.plot(s.loc[~clear, "date"], s.loc[~clear, "ndvi_mean"], "o", mfc="none", mec="0.6", ms=2)
        if pd.notna(r["harvest_window_start"]):
            ax.axvspan(
                r["harvest_window_start"], r["harvest_window_end"], color="tab:blue", alpha=0.4
            )
        ax.axhline(HIGH, color="0.5", ls="--", lw=0.6)
        ax.axhline(LOW, color="0.5", ls=":", lw=0.6)
        title = f"{r['id']}  {r['label']}" + (
            f" ({r['orig_field']})" if pd.notna(r["orig_field"]) else ""
        )
        ax.set_title(title, fontsize=10, color=colors[r["label"]], fontweight="bold")
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
        ax.grid(alpha=0.3)
    for ax in flat[len(res) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=8, rotation=45)
    fig.suptitle(
        "Sentinel-2 NDVI per candidate field (black = clear scenes, hollow = cloudy; grey band = "
        f"Jun-Oct; blue = 2025/26 harvest window; dashed {HIGH}, dotted {LOW})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fmt(x, spec=".2f") -> str:
    return "—" if pd.isna(x) else format(x, spec)


def markdown_table(res: pd.DataFrame) -> str:
    lines = [
        "| id | hyp | label | reason | area_ha | ndvi_max | n_collapses_2024-25 "
        "| n_collapses_2025-26 | harvest_window_2025-26 | notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        win = (
            "—"
            if pd.isna(r["harvest_window_start"])
            else f"{r['harvest_window_start']:%Y-%m-%d}→{r['harvest_window_end']:%Y-%m-%d}"
        )
        lines.append(
            f"| {r['id']} | {r['hyp']} | **{r['label']}** | {r['reason']} | {r['area_ha']:.2f} "
            f"| {fmt(r['ndvi_max'])} | {r['n_collapses_2024-25']} | {r['n_collapses_2025-26']} "
            f"| {win} | {r['notes']} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--max-field-cloud",
        type=float,
        default=10,
        help="max %% of field pixels masked for a scene to count as clear",
    )
    ap.add_argument(
        "--from-cache", action="store_true", help=f"reuse {SERIES_CSV.relative_to(ROOT)}"
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # `suggested` (an earlier, failed rule) and the S1/WorldCover features are dropped:
    # the label below comes only from the NDVI series.
    polys = gpd.read_file(IN_GEOJSON)[["id", "hyp", "geometry"]]
    polys["area_ha"] = polys.to_crs(UTM).area / 1e4
    mapping, map_lines = map_original_fields(polys)

    if args.from_cache:
        series = pd.read_csv(SERIES_CSV, parse_dates=["date"])
    else:
        series = fetch_series(polys)
        series.to_csv(SERIES_CSV, index=False, date_format="%Y-%m-%d")

    rows, could_not = [], []
    for _, p in polys.iterrows():
        clear = clear_scenes(series[series["id"] == p["id"]], args.max_field_cloud)
        out = label_field(clear)
        ws, we = harvest_window(clear) if out["label"] == "cana" else (None, None)

        notes = [f"{len(clear)} clear scenes"]
        if p["id"] in mapping:
            notes.append(f"= {mapping[p['id']]}")
        if p["area_ha"] < SMALL_HA:
            notes.append("small (unreliable)")
        if out["collapse_dates"]:
            notes.append(f"collapses {out['collapse_dates']}")
        if out["label"] == "cana" and ws is None:
            notes.append("no 2025/26 step found")
            could_not.append(f"{p['id']}: cana but no ≥{HARVEST_HIGH}→≤{LOW} step in 2025/26")
        if len(clear) < MIN_CLEAR_SCENES:
            could_not.append(f"{p['id']}: only {len(clear)} clear scenes (< {MIN_CLEAR_SCENES})")
        if out["label"] == "cana" and ws is not None and (we - ws).days > 30:
            could_not.append(
                f"{p['id']}: harvest window {(we - ws).days} d wide (cloud gap) "
                f"{ws:%Y-%m-%d}→{we:%Y-%m-%d}"
            )
        if p["area_ha"] < SMALL_HA:
            could_not.append(f"{p['id']}: {p['area_ha']:.2f} ha < {SMALL_HA} ha (edge pixels)")

        rows.append(
            {
                "id": p["id"],
                "hyp": p["hyp"],
                **out,
                "area_ha": round(p["area_ha"], 3),
                "n_clear_scenes": len(clear),
                "harvest_window_start": ws,
                "harvest_window_end": we,
                "orig_field": mapping.get(p["id"]),
                "notes": "; ".join(notes),
            }
        )
    res = pd.DataFrame(rows)

    # Rainy-season cloud gaps: the plateau test bridges them, so report the longest gap
    # between clear scenes for every field (a long gap could hide a dip).
    for fid, s in series.groupby("id"):
        c = clear_scenes(s, args.max_field_cloud)
        if len(c) >= 2:
            gaps = c["date"].diff().dt.days
            k = gaps.idxmax()
            if gaps[k] > 60:
                could_not.append(
                    f"{fid}: {int(gaps[k])}-day gap without a clear scene "
                    f"({c.loc[k - 1, 'date']:%Y-%m-%d}→{c.loc[k, 'date']:%Y-%m-%d})"
                )

    # Outputs. The GeoJSON keeps the requested properties plus `orig_field` (field_0x id
    # when the polygon is one of the six fields used in exercises 1-6, else null).
    out_cols = ["id", "hyp", "label", "area_ha", "harvest_window_start", "harvest_window_end"]
    gdf = polys[["id", "geometry"]].merge(res[[*out_cols, "orig_field"]], on="id")
    for c in ("harvest_window_start", "harvest_window_end"):
        gdf[c] = gdf[c].dt.strftime("%Y-%m-%d")  # NaT -> null in GeoJSON
    gdf[[*out_cols, "orig_field", "geometry"]].to_file(OUT_GEOJSON, driver="GeoJSON")
    res.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d", float_format="%.4f")
    plot_grid(res, series, args.max_field_cloud, OUT_DIR / "ndvi_grid.png")

    agree = res["hyp"] == res["label"]
    print("\n## Cane / not-cane labels from Sentinel-2 NDVI phenology\n")
    print(
        f"Tests: (a) NDVI ≥ {HIGH} for ≥ {MIN_RUN_DAYS} d continuous; (b) collapses "
        f"(≥ {HIGH} → ≤ {LOW}) ≤ 1 per Nov-May season and 0 in Jun-Oct; (c) max ≥ {PEAK}. "
        f"Clear scene = ≤ {args.max_field_cloud:g} % of field masked by SCL.\n"
    )
    print(markdown_table(res))
    print("\n### Agreement with the author's hypothesis\n")
    print(f"- agree: {int(agree.sum())} / {len(res)}")
    print(f"- disagree: {int((~agree & (res['label'] != 'dudoso')).sum())}")
    print(f"- dudoso (no call): {int((res['label'] == 'dudoso').sum())}")
    print("\n" + pd.crosstab(res["hyp"], res["label"]).to_string())
    print("\n### Original field ids\n")
    print("\n".join(f"- {line}" for line in map_lines))
    print("\n### Could not verify\n")
    print("\n".join(f"- {c}" for c in could_not) or "- (none)")
    print("\n### Files\n")
    for path in (OUT_GEOJSON, OUT_CSV, SERIES_CSV, OUT_DIR / "ndvi_grid.png"):
        print(f"- {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
