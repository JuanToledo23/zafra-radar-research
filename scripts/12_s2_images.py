"""Exercise 4d, step 1: Sentinel-2 inputs for field segmentation over the supply area.

    python scripts/12_s2_images.py [--site morelos] [--force]

AOI, grid CRS, windows and file names come from configs/<site>.yaml (exercise 5). Morelos:
the FIRMS bbox of the Ingenio Emiliano Zapata supply area (lon -99.35..-99.05,
lat 18.50..18.75), on a 10 m UTM 14N grid (EPSG:32614). Writes under the site's s2_dir
(Morelos data/s2/, git-ignored):

  s2_scene_<date>.tif   the clearest single scene of zafra.scene_window (Morelos
                        2025-12-01 .. 2026-01-31, full canopy):
                        B4, B3, B2, B8 (reflectance 0-1) and NDVI, SCL-masked per pixel
  paths.temporal        per-pixel NDVI median, std, min and clear-look count over the
                        crop year (Morelos 2025-06-01 .. 2026-05-31), SCL-masked per look
  paths.worldcover      ESA WorldCover v200 class (for removing built-up / water segments)
  paths.ndvi_stack      one int16 band per acquisition date of the crop year (Morelos 77):
                        NDVI x 10000 where SCL-clear, -20000 masked, -32768 off-swath. Used
                        to compute per-segment NDVI series locally (step 4), instead of one
                        Earth Engine request per polygon
  s2_images.json        what was chosen and how it was downloaded

Download: ee.Image.getDownloadURL on 1000 x 1000 px tiles (400 x 400 for the 77-band stack;
one request is capped at ~48 MB),
each tile on an explicit crs_transform of the same grid, pasted into one array.
"""

import argparse
import json
import sys
import time

from zafra import optical
from zafra.site import add_site_argument, load_site


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_site_argument(ap)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    site = load_site(args.site)
    s2_dir = site.path("s2_dir")
    s2_dir.mkdir(parents=True, exist_ok=True)
    crop_year = tuple(f"{t:%Y-%m-%d}" for t in site.crop_year)
    grid = optical.aoi_grid(site.aoi, epsg=site.epsg)
    print(f"grid: {grid['width']} x {grid['height']} px at 10 m, EPSG:{grid['epsg']}")
    meta = {"aoi_lonlat": site.aoi, "grid": grid, "tile_px": optical.TILE_PX}

    t0 = time.perf_counter()
    ranks = optical.scene_clear_fractions(grid, *site.scene_window)
    print(ranks[["date", "clear_frac"]].head(8).to_string(index=False))
    best = ranks.iloc[0]
    day = best["date"].strftime("%Y-%m-%d")
    meta["scene"] = {
        "date": day,
        "clear_frac_aoi": float(best["clear_frac"]),
        "granules": best["image_ids"],
        "candidates": [
            {"date": d.strftime("%Y-%m-%d"), "clear_frac": round(float(f), 4)}
            for d, f in zip(ranks["date"], ranks["clear_frac"], strict=True)
        ],
    }

    stack_name = site.path("ndvi_stack").name
    jobs = {
        f"s2_scene_{day.replace('-', '')}.tif": optical.scene_image(day, grid),
        site.path("temporal").name: optical.temporal_composite(grid, *crop_year),
        site.path("worldcover").name: optical.worldcover_image(),
    }
    stack, dates, tile_cloud = optical.ndvi_stack_image(grid, *crop_year)
    meta["ndvi_stack"] = {
        "dates": [f"{d:%Y-%m-%d}" for d in dates],
        "tile_cloud_pct": [round(float(c), 2) for c in tile_cloud],
        "max_tile_cloud": 50,
    }
    jobs[stack_name] = stack
    meta_path = s2_dir / "s2_images.json"
    prev = json.loads(meta_path.read_text()).get("files", {}) if meta_path.exists() else {}
    meta["files"] = {}
    for name, img in jobs.items():
        path = s2_dir / name
        if path.exists() and not args.force:
            print(f"{name}: exists, skipped")
            meta["files"][name] = prev.get(name, {})
            continue
        t = time.perf_counter()
        if name == stack_name:
            _, n = optical.download_tiled(img, grid, path, tile_px=400, dtype="int16")
        else:
            _, n = optical.download_tiled(img, grid, path)
        dt = time.perf_counter() - t
        print(f"{name}: {n} tiles in {dt:.0f} s")
        meta["files"][name] = {"tiles": n, "seconds": round(dt, 1)}
    meta["runtime_s"] = round(time.perf_counter() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"done in {meta['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
