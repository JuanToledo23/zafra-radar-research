"""Exercise 9: NISAR raster layers for the web map (COGs + XYZ PNG tiles).

    python scripts/18_web_rasters.py [--site morelos] [--no-tiles] [--max-zoom 14]

The site's `web_rasters` block (configs/<site>.yaml) names the GCOV source (one orbit), six
dates, the boxcar size and the tile zooms. For each date, over the site AOI snapped to whole
z14 tiles (zafra.rasters.web_grid):

1. read HVHV, HHHH and the GCOV mask, boxcar in linear power over valid pixels, reproject
   to EPSG:3857 (z14 pixel size), dB;
2. data/rasters/<site>/hv_<date>.tif, hh_<date>.tif, cross_<date>.tif (HV - HH), float32
   dB clipped to the fixed stretch, NaN nodata;
3. hv_change_<d1>_<d2>.tif for consecutive dates and hv_change_season.tif (lowest HV over
   the dates minus the first date), from the unclipped dB;
4. palette PNG tiles data/tiles/<site>/<layer>/{z}/{x}/{y}.png and
   data/tiles/<site>/tiles_index.json (layers, dates, bounds, ramp, units, sizes).

Morelos only (it has the labelled fields): results/9/raster_check.md compares the HV
raster at the centroid of p36 and two other labelled fields with the 4b field means
(data/features/fields_v2_features.parquet, ascending), and results/9/p36_hv_before_after.png
shows HV around p36 on the dates before / after its harvest. The Sentinel-1 VH series of
the p36 ground-truth check (script 09, outputs/front/*.csv: p36 and its neighbour segments,
2024-10 .. 2025-06) is re-exported to
data/results/morelos/s1_series.parquet if missing. Runtimes: results/9/runtimes.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from zafra import gcov as G
from zafra import rasters as R
from zafra.paths import DATA_DIR, FEATURES_DIR, OUTPUTS_DIR, RESULTS_DIR
from zafra.site import add_site_argument, load_site

OUT_RESULTS = RESULTS_DIR / "9"
CHECK_FIELDS = ("p36", "p16", "p10")  # p36 (ground truth) + the two largest other cane fields
DATES_4B = RESULTS_DIR / "4b" / "dates.csv"
S1_CACHES = (OUTPUTS_DIR / "front" / "p36_s1_vh.csv", OUTPUTS_DIR / "front" / "neigh_s1.csv")


def tag(date: str) -> str:
    return pd.Timestamp(date).strftime("%Y%m%d")


def find_gcov(src, date: str) -> Path:
    d = tag(date)
    hits = sorted(src.dir.glob(f"NISAR_L2_PR_GCOV_*_{src.track:03d}_{src.direction[0]}_*_{d}T*.h5"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{src.dir}: expected one GCOV for {date}, found {len(hits)}")
    return hits[0]


def read_date(path: Path, grid: dict, boxcar: int) -> dict:
    """HV / HH dB on the web grid for one GCOV file (unclipped, NaN = nodata)."""
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    with G.open_gcov(path) as f:
        g = G.read_grid(f)
        minfo = G.mask_info(f)
        bounds = transform_bounds(grid["crs"], f"EPSG:{g['epsg']}", *grid["bounds_merc"])
        # margin: the boxcar half-width plus a few pixels for bilinear resampling
        aoi = box(*bounds).buffer((boxcar + 4) * abs(g["dx"]))
        rows, cols, transform = G.window_for_geometry(g, aoi)
        valid = G.valid_from_mask(G.read_window(f, "A", "mask", rows, cols), minfo)
        out = {"valid_frac_src": float(valid.mean())}
        for pol, key in (("HVHV", "hv"), ("HHHH", "hh")):
            lin = R.speckle_filter(G.read_window(f, "A", pol, rows, cols), valid, boxcar)
            web = R.reproject_to_grid(lin, transform, g["epsg"], grid)
            out[key] = R.to_db(web).astype("float32")
    return out


def layer_entry(name, kind, path, dates, tiles=None) -> dict:
    lo, hi = R.STRETCH[kind]
    e = {
        "id": name,
        "kind": kind,
        "dates": dates,
        "cog": str(path.relative_to(DATA_DIR.parent)),
        "units": "dB",
        "label": R.KIND_LABEL[kind],
        "stretch": [lo, hi],
        "ramp": [{"at": round(lo + p * (hi - lo), 3), "rgb": list(c)} for p, c in R.ramp_for(kind)],
        "nodata": "transparent (PNG palette index 0); NaN in the COG",
    }
    if tiles:
        e.update(tiles)
    return e


def export_s1(site) -> str:
    """Re-export the script-09 Sentinel-1 VH caches to data/results/<site>/s1_series.parquet."""
    out = DATA_DIR / "results" / site.name / "s1_series.parquet"
    if out.exists():
        return f"{out.relative_to(DATA_DIR.parent)} exists, kept"
    if site.name != "morelos" or not all(p.exists() for p in S1_CACHES):
        return "no Sentinel-1 series computed for this site"
    p36 = pd.read_csv(S1_CACHES[0], parse_dates=["date"]).assign(field_id="p36", kind="field")
    nb = pd.read_csv(S1_CACHES[1], parse_dates=["date"])
    nb = nb.rename(columns={"seg_id": "field_id"}).assign(kind="p36_neighbour_segment")
    df = pd.concat([p36, nb], ignore_index=True)
    df = df.rename(columns={"mean_db": "vh_db", "mean_linear": "vh_linear"})
    df = df[
        [
            "field_id",
            "kind",
            "date",
            "orbit",
            "rel_orbit",
            "vh_db",
            "vh_linear",
            "valid_px",
            "image_id",
        ]
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["field_id", "orbit", "date"]).to_parquet(out, index=False)
    span = f"{df['date'].min().date()}..{df['date'].max().date()}"
    return (
        f"{out.relative_to(DATA_DIR.parent)}: {len(df)} rows, {df['field_id'].nunique()} "
        f"polygons (p36 + {nb['field_id'].nunique()} neighbour segments), {span}"
    )


# ------------------------------------------------------------------------------- checks


def raster_check(site, raster_dir: Path, dates: list[str]) -> pd.DataFrame:
    """HV raster at the field centroid and over the polygon vs the 4b field mean (asc)."""
    import rasterio
    from rasterio.features import geometry_mask

    fields = gpd.read_file(site.reference_fields).set_index("id").loc[list(CHECK_FIELDS)]
    fields = fields.to_crs(3857)
    feats = pd.read_parquet(FEATURES_DIR / "fields_v2_features.parquet")
    feats = feats[feats["orbit"] == "asc"].set_index(["field_id", "date"])["hv_db"]
    rows = []
    for fid, geom in fields.geometry.items():
        c = geom.centroid
        for d in dates:
            path = raster_dir / f"hv_{tag(d)}.tif"
            with rasterio.open(path) as src:
                win = rasterio.windows.from_bounds(*geom.bounds, transform=src.transform)
                win = win.round_offsets().round_lengths()
                arr = src.read(1, window=win)
                inside = geometry_mask([geom], arr.shape, src.window_transform(win), invert=True)
            vals = arr[inside & np.isfinite(arr)]
            poly_db = float(10 * np.log10(np.mean(10 ** (vals / 10)))) if vals.size else np.nan
            ref = float(feats.get((fid, pd.Timestamp(d)), np.nan))
            cen = R.sample_merc(path, c.x, c.y)
            rows.append(
                {
                    "field": fid,
                    "date": d,
                    "field_mean_4b_db": ref,
                    "centroid_db": cen,
                    "centroid_minus_4b": cen - ref,
                    "raster_poly_mean_db": poly_db,
                    "poly_minus_4b": poly_db - ref,
                }
            )
    return pd.DataFrame(rows)


def before_after(site, dates: list[str]) -> tuple[str, str]:
    """Selected dates just before p36's 4b xgb window and just after its 2026 ground truth."""
    d4b = pd.read_csv(DATES_4B)
    r = d4b[(d4b["field_id"] == "p36") & (d4b["method"] == "xgb")].iloc[0]
    gt = gpd.read_file(site.reference_fields).set_index("id").loc["p36", "ground_truth_date_2026"]
    start = pd.Timestamp(r["date_before"])
    end = max(pd.Timestamp(r["date_after"]), pd.Timestamp(gt))
    ts = [pd.Timestamp(d) for d in dates]
    before = max(t for t in ts if t < start)
    after = min(t for t in ts if t > end)
    return before.date().isoformat(), after.date().isoformat()


def plot_p36(site, raster_dir: Path, d0: str, d1: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    p36 = gpd.read_file(site.reference_fields).set_index("id").loc[["p36"]].to_crs(3857)
    x0, y0, x1, y1 = p36.total_bounds
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.3
    bounds = (cx - half, cy - half, cx + half, cy + half)
    lo, hi = R.STRETCH["hv"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), constrained_layout=True)
    for ax, d in zip(axes, (d0, d1), strict=True):
        with rasterio.open(raster_dir / f"hv_{tag(d)}.tif") as src:
            win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            arr = src.read(1, window=win)
            wb = rasterio.windows.bounds(win, src.transform)
        im = ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi, extent=(wb[0], wb[2], wb[1], wb[3]))
        p36.boundary.plot(ax=ax, color="#ffcc00", linewidth=1.6)
        ax.set_title(f"NISAR HV, {d} (ascending t19)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.colorbar(im, ax=axes, shrink=0.8, label="HV gamma-0 (dB), 5x5 boxcar")
    fig.suptitle("p36 La Marometa (21 ha), before / after the 2026 harvest (ground truth 02-26)")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def check_md(df: pd.DataFrame, before: str, after: str, png: str) -> str:
    tab = df.copy()
    for c in tab.columns[2:]:
        fmt = "{:+.2f}" if "minus" in c else "{:.2f}"
        tab[c] = tab[c].map(fmt.format)
    head = "| " + " | ".join(tab.columns) + " |\n|" + "---|" * len(tab.columns) + "\n"
    body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in tab.itertuples(index=False))
    a = df["centroid_minus_4b"].abs()
    b = df["poly_minus_4b"].abs()
    return f"""# Exercise 9: web raster sanity check (Morelos)

HV raster (`data/rasters/morelos/hv_<date>.tif`: 5x5 boxcar in linear power, EPSG:3857 at
the z14 pixel size, dB) against the exercise-4b field means
(`data/features/fields_v2_features.parquet`, ascending t19: mean of linear power over the
pixel centres inside the polygon on the native 10 m grid, then dB).

* `centroid_db`: the raster at the field centroid (one ~50 m boxcar footprint);
* `raster_poly_mean_db`: mean of the raster's linear power over the polygon (checks the
  reprojection and the dB/linear round trip; small fields lose edge pixels to the boxcar).

Fields: p36 (ground truth) and the two largest other cane fields, p16 (9.0 ha) and p10
(6.9 ha).

{head}{body}
* Centroid vs 4b: |diff| median {a.median():.2f} dB, max {a.max():.2f} dB;
  {int((a <= 1).sum())} of {len(a)} within 1 dB.
* Polygon mean vs 4b: |diff| median {b.median():.2f} dB, max {b.max():.2f} dB;
  {int((b <= 1).sum())} of {len(b)} within 1 dB.

The polygon means reproduce the 4b values, so georeferencing, the mask, the boxcar and the
linear/dB handling are right. The centroid samples miss the ~1 dB target on half the
field-dates. That comes from texture inside the fields, not from the raster: one 50 m
footprint is not the field mean. p36 before the cut is patchy (its centroid is +2-3 dB
above the field mean; see the figure). A web popup should show the field mean (API /
series), not the pixel under the cursor.

`{png}`: HV around p36 on {before} (last selected date before the 4b xgb window
2026-02-10 .. 02-15) and {after} (first after the 2026-02-26 ground truth); yellow = p36.
"""


# --------------------------------------------------------------------------------- main


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_site_argument(ap)
    ap.add_argument("--no-tiles", action="store_true", help="COGs only")
    ap.add_argument("--max-zoom", type=int, default=None, help="override web_rasters.zooms[1]")
    args = ap.parse_args()

    site = load_site(args.site, register=False)
    cfg = site.web_rasters
    if not cfg:
        sys.exit(f"configs/{site.name}.yaml has no web_rasters block")
    src = site.nisar[cfg["source"]]
    dates = [str(d) for d in cfg["dates"]]
    boxcar = int(cfg.get("boxcar", 5))
    zooms = (int(cfg["zooms"][0]), int(args.max_zoom or cfg["zooms"][1]))
    raster_dir = DATA_DIR / "rasters" / site.name
    tile_dir = DATA_DIR / "tiles" / site.name
    grid = R.web_grid(site.aoi)
    t_all = time.perf_counter()
    times: dict[str, float] = {}
    print(
        f"{site.name}: {src.direction} t{src.track} f{src.frame}, {len(dates)} dates, "
        f"grid {grid['width']}x{grid['height']} px at {grid['res']:.2f} m, zooms {zooms}"
    )

    layers, hv = [], {}

    def emit(name, kind, arr, layer_dates):
        t = time.perf_counter()
        path = R.write_cog(
            raster_dir / f"{name}.tif",
            R.clip_stretch(arr, kind),
            grid,
            {"kind": kind, "dates": ",".join(layer_dates), "units": "dB"},
        )
        tiles = None
        if not args.no_tiles:
            tiles = R.write_tiles(R.clip_stretch(arr, kind), grid, kind, tile_dir / name, zooms)
        layers.append(layer_entry(name, kind, path, layer_dates, tiles))
        times[f"write_{name}"] = round(time.perf_counter() - t, 1)

    valid_frac = {}
    for d in dates:
        t = time.perf_counter()
        path = find_gcov(src, d)
        r = read_date(path, grid, boxcar)
        times[f"read_{tag(d)}"] = round(time.perf_counter() - t, 1)
        hv[d] = r["hv"]
        valid_frac[d] = float(np.isfinite(r["hv"]).mean())
        print(
            f"  {d}: {path.name[:60]}... read {times[f'read_{tag(d)}']} s, "
            f"valid {valid_frac[d]:.1%}"
        )
        emit(f"hv_{tag(d)}", "hv", r["hv"], [d])
        emit(f"hh_{tag(d)}", "hh", r["hh"], [d])
        emit(f"cross_{tag(d)}", "cross", r["hv"] - r["hh"], [d])

    for d0, d1 in zip(dates[:-1], dates[1:], strict=True):
        emit(f"hv_change_{tag(d0)}_{tag(d1)}", "change", R.change(hv[d0], hv[d1]), [d0, d1])
    emit("hv_change_season", "change", R.season_change([hv[d] for d in dates]), dates)

    sizes = {"rasters_bytes": dir_size(raster_dir)}
    if not args.no_tiles:
        sizes["tiles_bytes"] = dir_size(tile_dir)
        index = {
            "site": site.name,
            "description": site.description,
            "source": {
                "product": "NISAR L2 GCOV (gamma-0, frequency A)",
                "orbit": src.direction.lower(),
                "track": src.track,
                "frame": src.frame,
            },
            "dates": dates,
            "url_template": "{layer}/{z}/{x}/{y}.png",
            "tile_size": R.TILE_PX,
            "minzoom": zooms[0],
            "maxzoom": zooms[1],
            "bounds": [round(v, 6) for v in R.grid_bounds_lonlat(grid)],
            "aoi": list(site.aoi),
            "processing": f"{boxcar}x{boxcar} boxcar in linear power over GCOV-mask-valid "
            "pixels, bilinear to EPSG:3857 at the z14 pixel size, dB, clipped to the stretch; "
            "change = HV(d2) - HV(d1) from unclipped dB; season = min over dates - first date",
            "valid_fraction": valid_frac,
            "total_tile_bytes": sizes["tiles_bytes"],
            "layers": layers,
        }
        (tile_dir / "tiles_index.json").write_text(json.dumps(index, indent=2) + "\n")

    s1_note = export_s1(site)
    print(f"  S1: {s1_note}")

    if site.name == "morelos" and site.reference_fields:
        t = time.perf_counter()
        OUT_RESULTS.mkdir(parents=True, exist_ok=True)
        df = raster_check(site, raster_dir, dates)
        d0, d1 = before_after(site, dates)
        png = OUT_RESULTS / "p36_hv_before_after.png"
        plot_p36(site, raster_dir, d0, d1, png)
        md = check_md(df, d0, d1, str(png.relative_to(DATA_DIR.parent)))
        (OUT_RESULTS / "raster_check.md").write_text(md)
        times["sanity_check"] = round(time.perf_counter() - t, 1)
        print(md)

    times["total"] = round(time.perf_counter() - t_all, 1)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    rt_path = OUT_RESULTS / "runtimes.json"
    rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
    rt[site.name] = {"seconds": times, **sizes, "n_layers": len(layers), "zooms": list(zooms)}
    rt_path.write_text(json.dumps(rt, indent=2) + "\n")
    mb = {k: round(v / 2**20, 1) for k, v in sizes.items()}
    print(f"  done in {times['total']} s; sizes MB {mb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
