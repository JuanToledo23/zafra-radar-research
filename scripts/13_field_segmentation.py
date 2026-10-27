"""Exercise 4d, steps 2-3: segment the supply area into candidate fields and score them.

    python scripts/13_field_segmentation.py [--stage sam|polygons|all] [--force]

Inputs: data/s2/ from scripts/12_s2_images.py.

1. SAM (ViT-B, samgeo automatic mask generation, CPU, tiled) on
   (a) the clearest Dec-Jan scene as true colour (B4, B3, B2), and
   (b) a 3-band image of the crop-year NDVI stack (median, std, min).
   Binary mask rasters in data/s2/sam_mask_{rgb,temporal}.tif (git-ignored).
2. Classical comparison: watershed of the NDVI-stack edges (segments.classic_labels).
3. Polygons, cleaned (0.5-60 ha, 5 m simplify, invalid fixed, WorldCover built-up / water
   dropped): data/fields/segments_{sam_rgb,sam_temporal,classic}.geojson.
4. IoU against the 38 hand-traced fields_v2 polygons, per field and summary; the best
   variant is copied to data/fields/segments.geojson. Tables in results/4d/.
"""

import argparse
import json
import shutil
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from zafra import segments as S
from zafra.paths import DATA_DIR, FIELDS_DIR, RESULTS_DIR

S2_DIR = DATA_DIR / "s2"
CKPT_DIR = DATA_DIR / "models" / "sam"
SCENE = sorted(S2_DIR.glob("s2_scene_*.tif"))
TEMPORAL = S2_DIR / "s2_temporal_2025_26.tif"
WORLDCOVER = S2_DIR / "worldcover_2021.tif"
REF = FIELDS_DIR / "fields_v2.geojson"
OUT_DIR = RESULTS_DIR / "4d"
RUNTIME_LOG = OUT_DIR / "runtimes.json"
VARIANTS = {
    "sam_rgb": "SAM on the 2026-01-10 true-colour scene",
    "sam_temporal": "SAM on the NDVI stack (median, std, min)",
    "classic": "Felzenszwalb on the NDVI stack (median, std, min)",
}


def log_runtime(key, seconds, extra=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.loads(RUNTIME_LOG.read_text()) if RUNTIME_LOG.exists() else {}
    d[key] = {"seconds": round(seconds, 1), **(extra or {})}
    RUNTIME_LOG.write_text(json.dumps(d, indent=2))


def stage_sam(force):
    inputs = {
        "rgb": (SCENE[-1], (1, 2, 3)),
        "temporal": (TEMPORAL, (1, 2, 3)),
    }
    for name, (src, bands) in inputs.items():
        inp, out = S2_DIR / f"sam_input_{name}.tif", S2_DIR / f"sam_mask_{name}.tif"
        if out.exists() and not force:
            print(f"{out.name}: exists, skipped")
            continue
        limits = S.write_sam_input(src, bands, inp)
        print(f"{inp.name}: stretch {limits}")
        t = time.perf_counter()
        n = S.run_sam(inp, out, CKPT_DIR, device="cpu")
        dt = time.perf_counter() - t
        print(f"SAM {name}: {n} tiles in {dt:.0f} s ({dt / n:.1f} s/tile)")
        log_runtime(
            f"step2_sam_{name}",
            dt,
            {"tiles": n, "device": "cpu", "model": "vit_b", "sam_kwargs": S.SAM_KWARGS},
        )


def polygons_from_labels(labels, transform):
    gdf = S.labels_to_polygons(labels, transform)
    clean, counts = S.clean_segments(gdf, worldcover_tif=WORLDCOVER)
    return clean, counts


def stage_polygons():
    out, counts = {}, {}
    for name in ("rgb", "temporal"):
        with rasterio.open(S2_DIR / f"sam_mask_{name}.tif") as ds:
            mask, tr = ds.read(1), ds.transform
        t = time.perf_counter()
        out[f"sam_{name}"], counts[f"sam_{name}"] = polygons_from_labels(S.mask_to_labels(mask), tr)
        print(f"sam_{name}: polygons in {time.perf_counter() - t:.0f} s, {counts[f'sam_{name}']}")

    t = time.perf_counter()
    with rasterio.open(TEMPORAL) as ds:
        med, std, mn = ds.read([1, 2, 3])
        tr = ds.transform
    lab = S.classic_labels(med, std, mn)
    out["classic"], counts["classic"] = polygons_from_labels(lab, tr)
    dt = time.perf_counter() - t
    log_runtime("step2b_classic", dt)
    print(f"classic: {dt:.0f} s, {counts['classic']}")

    for name, g in out.items():
        g = g.drop(columns=["label"]).copy()
        g.insert(0, "seg_id", [f"{name[:1]}{name[-1]}{i:05d}" for i in range(len(g))])
        g["source"] = name
        g["area_ha"] = g["area_ha"].round(3)
        g = g.to_crs("EPSG:4326")
        g.geometry = g.geometry.set_precision(1e-7)  # ~1 cm; keeps the files small
        g.to_file(FIELDS_DIR / f"segments_{name}.geojson", driver="GeoJSON")
        out[name] = g
    (OUT_DIR / "cleaning_counts.json").write_text(json.dumps(counts, indent=2))
    return out, counts


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(per, summ, best, counts):
    """results/4d/segmentation.md: summary per variant and the per-field table."""
    flag = np.where(
        per["over_segmented"],
        "over",
        np.where(
            per["under_segmented"], "under", np.where(per["best_segment"].isna(), "missed", "")
        ),
    )
    per = per.assign(flag=flag)
    wide = per.pivot(index="field_id", columns="variant", values="iou")
    ref = per.drop_duplicates("field_id").set_index("field_id")[["label", "field_ha"]]
    best_rows = per[per["variant"] == best].set_index("field_id")
    tab = ref.join(wide[list(VARIANTS)]).join(
        best_rows[["segment_ha", "flag"]].add_prefix(f"{best}_")
    )
    tab = tab.reset_index().sort_values(best, ascending=False)
    cnt = pd.DataFrame(counts).T.reset_index().rename(columns={"index": "variant"})
    lines = [
        "# Exercise 4d: field segmentation vs the 38 hand-traced fields_v2 polygons",
        "",
        "Generated by `scripts/13_field_segmentation.py`. IoU = intersection / union of a "
        "field and the segment that overlaps it most. Over-segmented: >= 2 segments each "
        f"cover > {S.OVER_SHARE:.0%} of the field. Under-segmented: that segment is > "
        f"{S.UNDER_RATIO}x the field area. Missed: no segment touches the field after cleaning.",
        "",
        "## Summary",
        "",
        md_table(summ.assign(variant=summ["variant"].map(lambda v: f"{v} ({VARIANTS[v]})"))),
        "",
        f"Chosen: **{best}** (most fields with IoU >= 0.5, then median IoU) -> "
        "`data/fields/segments.geojson`.",
        "",
        "## Cleaning counts",
        "",
        md_table(cnt),
        "",
        "## Per field (IoU)",
        "",
        md_table(tab),
        "",
    ]
    (OUT_DIR / "segmentation.md").write_text("\n".join(lines))


def stage_evaluate(segs, counts):
    ref = gpd.read_file(REF)
    per, summ = [], []
    for name, g in segs.items():
        m = S.match_fields(ref, g).assign(variant=name)
        per.append(m)
        summ.append({"variant": name, "n_segments": len(g), **S.summarize_matches(m)})
    per, summ = pd.concat(per, ignore_index=True), pd.DataFrame(summ)
    per = per.merge(ref[["id", "label"]], left_on="field_id", right_on="id").drop(columns="id")
    per.to_csv(OUT_DIR / "segmentation_iou.csv", index=False, float_format="%.3f")
    summ.to_csv(OUT_DIR / "segmentation_summary.csv", index=False, float_format="%.3f")
    print(summ.to_string(index=False))
    # Best: highest share with IoU >= 0.5, then median IoU.
    best = summ.sort_values(["n_iou_ge_0_5", "median_iou"], ascending=False).iloc[0]["variant"]
    shutil.copy(FIELDS_DIR / f"segments_{best}.geojson", FIELDS_DIR / "segments.geojson")
    print(f"best variant: {best} -> data/fields/segments.geojson")
    (OUT_DIR / "segmentation_choice.json").write_text(json.dumps({"best": best}, indent=2))
    write_report(per, summ, best, counts)
    return per, summ, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", choices=["sam", "polygons", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="re-run SAM even if masks exist")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("sam", "all"):
        stage_sam(args.force)
    if args.stage in ("polygons", "all"):
        segs, counts = stage_polygons()
        stage_evaluate(segs, counts)
    return 0


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    sys.exit(main())
