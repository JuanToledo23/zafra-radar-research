"""Exercise 4e, steps 4-5: merge cane fragments, score the merged layer, slim export.

    python scripts/15_merge_fragments.py

Input: data/results/harvest_dates_2025_26.geojson (scripts/14_area_harvest_map.py).

1. Adjacent gated cane segments (>= 10 m of shared boundary) are merged when their harvest
   dates differ by <= 6 days, or both have no date and the same status
   (segments.merge_groups; a merged group's dates never span more than 6 days). Date,
   p_cane, p_cane_radar, hv_drop_db, coh_min are area-weighted; status, confidence,
   wc_class from the largest member; fires_7d_before is recounted for the merged polygon.
   Non-cane segments are left as they are.
2. IoU against the 38 fields_v2 polygons, unmerged vs merged (cane merged + non-cane as is).
   If the merged layer scores better (more fields with IoU >= 0.5, then median IoU) it is
   the delivered layer, otherwise the unmerged one is.
3. data/results/harvest_dates_2025_26_merged.geojson (all columns),
   data/results/harvest_map_2025_26_slim.geojson (gated cane polygons of the delivered
   layer, 5 m simplified, map columns only; non-cane left out to stay under 15 MB),
   results/4e/merge.md and merge_summary.json, results/4e/harvest_map.png redrawn from
   the delivered layer.
"""

import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from zafra import cane_features as CF
from zafra import fires as F
from zafra import plots
from zafra import segments as S
from zafra.paths import DATA_DIR, FIELDS_DIR, FIRMS_DIR, RESULTS_DIR

IN = DATA_DIR / "results" / "harvest_dates_2025_26.geojson"
MERGED = DATA_DIR / "results" / "harvest_dates_2025_26_merged.geojson"
SLIM = DATA_DIR / "results" / "harvest_map_2025_26_slim.geojson"
REF = FIELDS_DIR / "fields_v2.geojson"
OUT_DIR = RESULTS_DIR / "4e"
METRIC = S.METRIC_CRS
WEIGHTED = ["p_cane", "p_cane_radar", "hv_drop_db", "coh_min"]
FIRST = [
    "seg_id",
    "source",
    "cane_pred",
    "gate_confidence",
    "status",
    "confidence",
    "method",
    "wc_class",
    "wc_disagrees",
    "harvest_after",
]
SLIM_COLS = [
    "segment_id",
    "area_ha",
    "p_cane",
    "p_cane_radar",
    "harvest_date",
    "status",
    "confidence",
    "hv_drop_db",
    "coh_min",
    "wc_disagrees",
]
SLIM_SIMPLIFY_M = 5.0
SLIM_DIGITS = 6  # decimal degrees: ~0.1 m


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def merge_cane(gdf):
    g = gdf.to_crs(METRIC)
    is_cane = g["cane_pred"].fillna(False).astype(bool)
    cane = g[is_cane].reset_index(drop=True)
    pairs = S.adjacent_pairs(cane)
    groups = S.merge_groups(cane["harvest_date"], cane["status"], pairs)
    merged = S.dissolve_groups(cane, groups, "harvest_date", weighted=WEIGHTED, first=FIRST)
    merged["harvest_date"] = pd.to_datetime(merged["harvest_date"]).dt.strftime("%Y-%m-%d")
    merged["harvest_date"] = merged["harvest_date"].where(merged["harvest_date"] != "NaT")
    # the largest member's id, plus a marker that it stands for several segments
    merged["seg_id"] = np.where(
        merged["n_merged"] > 1, "m" + merged["seg_id"].astype(str), merged["seg_id"]
    )
    rest = g[~is_cane].assign(n_merged=1, member_ids=g.loc[~is_cane, "seg_id"])
    out = pd.concat([merged, rest], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=METRIC), len(pairs)


def recount_fires(gdf):
    """VIIRS detections <= 200 m in the 7 days before harvest_after, per dated polygon."""
    dated = gdf[gdf["harvest_after"].notna()]
    ft = F.fires_per_field_fast(
        F.load_firms(FIRMS_DIR), dated.assign(id=dated["seg_id"]), CF.FIRE_BUFFER_M
    )
    ft["t"] = ft["acq_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    after = dated.set_index("seg_id")["harvest_after"].pipe(pd.to_datetime)
    ft["after"] = ft["field_id"].map(after)
    win = (ft["t"] < ft["after"]) & (ft["t"] >= ft["after"] - pd.Timedelta(days=7))
    n = ft[win].groupby("field_id").size()
    has = gdf["harvest_after"].notna()
    return gdf["seg_id"].map(n).fillna(0).where(has)


def iou_tables(unmerged, merged):
    ref = gpd.read_file(REF)
    rows, per = [], []
    for name, layer in (("unmerged", unmerged), ("merged", merged)):
        m = S.match_fields(ref, layer).merge(
            ref[["id", "label"]], left_on="field_id", right_on="id"
        )
        per.append(m.assign(layer=name))
        for subset, mm in (("all 38", m), ("26 cane", m[m["label"] == "cana"])):
            rows.append({"layer": name, "fields": subset, **S.summarize_matches(mm)})
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True)


def slim(gdf):
    s = gdf.to_crs(METRIC).copy()
    s["geometry"] = s.geometry.simplify(SLIM_SIMPLIFY_M, preserve_topology=True)
    s = S.fix_geometries(s)
    s = s.rename(columns={"seg_id": "segment_id"})
    s["area_ha"] = s["area_ha"].round(2)
    for c in ("p_cane", "p_cane_radar", "coh_min"):
        s[c] = s[c].round(3)
    s["hv_drop_db"] = s["hv_drop_db"].round(2)
    s["wc_disagrees"] = s["wc_disagrees"].fillna(False).astype(bool)
    s = s[[*SLIM_COLS, "geometry"]].to_crs("EPSG:4326")
    s.geometry = s.geometry.set_precision(10**-SLIM_DIGITS)
    return s


def main() -> int:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file(IN)
    merged, n_pairs = merge_cane(g)
    merged["fires_7d_before"] = recount_fires(merged)
    merged.to_crs("EPSG:4326").to_file(MERGED, driver="GeoJSON")

    is_cane_u = g["cane_pred"].fillna(False).astype(bool)
    is_cane_m = merged["cane_pred"].fillna(False).astype(bool)
    n_groups = int((is_cane_m & (merged["n_merged"] > 1)).sum())
    sizes = pd.DataFrame(
        [
            {
                "layer": "unmerged",
                "cane polygons": int(is_cane_u.sum()),
                "median cane ha": float(g.loc[is_cane_u, "area_ha"].median()),
                "mean cane ha": float(g.loc[is_cane_u, "area_ha"].mean()),
                "dated cane polygons": int((is_cane_u & g["harvest_date"].notna()).sum()),
                "all polygons": len(g),
            },
            {
                "layer": "merged",
                "cane polygons": int(is_cane_m.sum()),
                "median cane ha": float(merged.loc[is_cane_m, "area_ha"].median()),
                "mean cane ha": float(merged.loc[is_cane_m, "area_ha"].mean()),
                "dated cane polygons": int((is_cane_m & merged["harvest_date"].notna()).sum()),
                "all polygons": len(merged),
            },
        ]
    )
    summ, per = iou_tables(g, merged)
    print(md_table(sizes))
    print(md_table(summ))

    def score(layer):
        r = summ[(summ["layer"] == layer) & (summ["fields"] == "all 38")].iloc[0]
        return (r["n_iou_ge_0_5"], r["median_iou"])

    better = score("merged") > score("unmerged")
    delivered = merged if better else g
    name = "merged" if better else "unmerged"
    print(f"delivered layer: {name}")

    # Only gated cane polygons (every status): with the ~33 500 non-cane segments the file is
    # 36-61 MB even at 10 m tolerance; the map shows non-cane as basemap.
    s = slim(delivered[delivered["cane_pred"].fillna(False).astype(bool)])
    s.to_file(SLIM, driver="GeoJSON")
    size_mb = SLIM.stat().st_size / 1e6
    print(f"slim GeoJSON: {len(s)} polygons, {size_mb:.1f} MB")

    plots.harvest_map(
        delivered,
        OUT_DIR / "harvest_map.png",
        title="Zafra 2025-26: estimated harvest date "
        f"({name} segments, gate B + 4b xgboost, 3-look baseline)",
    )

    wide = per.pivot(index="field_id", columns="layer", values="iou")
    lab = per.drop_duplicates("field_id").set_index("field_id")[["label", "field_ha"]]
    tab = lab.join(wide[["unmerged", "merged"]]).assign(delta=lambda x: x["merged"] - x["unmerged"])
    tab = tab.reset_index().sort_values("delta", ascending=False)
    lines = [
        "# Exercise 4e: merging cane fragments",
        "",
        "Generated by `scripts/15_merge_fragments.py`. Adjacent gated cane segments (>= "
        f"{S.MIN_SHARED_EDGE_M:.0f} m shared boundary) merged when their harvest dates differ by "
        f"<= {S.MERGE_MAX_DAYS} days (a group never spans more), or both undated with the same "
        f"status. {n_pairs} adjacent cane pairs.",
        "",
        "## Size",
        "",
        md_table(sizes),
        "",
        "## IoU against the 38 fields_v2 polygons",
        "",
        md_table(summ),
        "",
        f"Delivered layer: **{name}**. Slim GeoJSON: {len(s)} polygons, {size_mb:.1f} MB.",
        "",
        "## Per field",
        "",
        md_table(tab),
        "",
    ]
    (OUT_DIR / "merge.md").write_text("\n".join(lines))
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(
            {
                "adjacent_cane_pairs": n_pairs,
                "merged_groups_with_2plus": n_groups,
                "sizes": sizes.to_dict("records"),
                "iou": summ.to_dict("records"),
                "delivered": name,
                "slim_mb": round(size_mb, 2),
                "slim_polygons": len(s),
                "runtime_s": round(time.perf_counter() - t0, 1),
            },
            indent=2,
            default=str,
        )
    )
    print(f"runtime {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
