"""Exercise 1b: read HH and HV backscatter for one field from one GCOV scene.

    python scripts/02_field_backscatter.py data/nisar/<granule>.h5 \
        [--field data/fields/field_01.geojson]

Steps (each one explained in comments below):
  polygon -> grid CRS -> array window -> read only that window -> pixel masks -> stats.
Writes outputs/<field>_<YYYYMMDD>_HVHV.png, ..._HVHV_context.png and <field>_<YYYYMMDD>.json.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt

from zafra import gcov
from zafra.paths import FIELDS_DIR, OUTPUTS_DIR

VMIN, VMAX = -25, -5  # dB display range for HV over cropland


def print_table(stats: dict) -> None:
    print(
        f"{'pol':<6} {'valid_px':>8} {'mean_dB':>8} {'median_dB':>9} "
        f"{'std_dB':>7} {'mean_looks':>10}"
    )
    for pol, s in stats.items():
        if s["valid_px"] == 0:
            print(f"{pol:<6} {0:>8}  (no valid pixels)")
            continue
        print(
            f"{pol:<6} {s['valid_px']:>8} {s['mean_db']:>8.2f} {s['median_db']:>9.2f} "
            f"{s['std_db']:>7.2f} {s.get('mean_looks', float('nan')):>10.2f}"
        )


def plot_window(f, res: dict, date: str, scale: float, out: Path) -> None:
    """HVHV in dB for the field window (scale=1) or a larger context window (scale=2)."""
    grid = res["grid"]
    rows, cols, transform = gcov.window_for_geometry(grid, res["geom"], scale=scale)
    hv_db = gcov.to_db(gcov.read_window(f, grid["freq"], "HVHV", rows, cols))
    if grid["has_mask"]:
        m = gcov.read_window(f, grid["freq"], "mask", rows, cols)
        hv_db[~gcov.valid_from_mask(m, res["mask_info"])] = float("nan")

    # extent = map coordinates of the outer pixel edges, so the image and polygon line up.
    left, top = transform.c, transform.f
    right = left + (cols.stop - cols.start) * transform.a
    bottom = top + (rows.stop - rows.start) * transform.e

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        hv_db,
        cmap="gray",
        vmin=VMIN,
        vmax=VMAX,
        extent=(left, right, bottom, top),
        interpolation="nearest",  # show real pixels, no smoothing
    )
    geoms = getattr(res["geom"], "geoms", [res["geom"]])
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, color="red", linewidth=1.5)
    label = "field window" if scale == 1 else f"{scale:g}x context"
    ax.set_title(f"{res['field_id']}  HV gamma-0  {date}  ({label})")
    ax.set_xlabel(f"easting (m, EPSG:{grid['epsg']})")
    ax.set_ylabel("northing (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(axis="x", labelrotation=45)
    fig.colorbar(im, ax=ax, label="HV gamma-0 (dB)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("h5", help="path to a NISAR GCOV .h5 file")
    p.add_argument("--field", default=str(FIELDS_DIR / "field_01.geojson"))
    p.add_argument("--freq", default="A", help="frequency band (A = 40 MHz main band)")
    args = p.parse_args()

    with gcov.open_gcov(args.h5) as f:
        ident = gcov.read_identification(f)
        date = ident["start_time"][:10]
        ymd = date.replace("-", "")

        # extract_field does: reproject polygon to the grid EPSG, compute the window,
        # rasterize the polygon (pixel centre inside = in), AND it with the GCOV validity
        # mask, then compute stats per polarization. See src/zafra/gcov.py.
        res = gcov.extract_field(f, args.field, args.freq)
        grid, minfo = res["grid"], res["mask_info"]

        print(f"scene   {ident['granule_id']}")
        print(
            f"date    {date}   track {ident['track']} frame {ident['frame']} "
            f"{ident['orbit_direction']}"
        )
        print(f"field   {res['field_id']} reprojected to EPSG:{grid['epsg']}")
        print(
            f"window  rows {res['window']['rows']}  cols {res['window']['cols']}  "
            f"({grid['dx']:g} m pixels)"
        )
        print(
            f"pixels  {res['inside_px']} inside polygon, {res['mask_rejected_px']} rejected by mask"
        )
        if minfo:
            print(f"mask    {minfo['path']}: {minfo['description']}")
        print()
        print_table(res["stats"])

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        png = OUTPUTS_DIR / f"{res['field_id']}_{ymd}_HVHV.png"
        png_ctx = OUTPUTS_DIR / f"{res['field_id']}_{ymd}_HVHV_context.png"
        plot_window(f, res, date, 1.0, png)
        plot_window(f, res, date, 5.0, png_ctx)

    out_json = OUTPUTS_DIR / f"{res['field_id']}_{ymd}.json"
    payload = {
        "field_id": res["field_id"],
        "field_file": args.field,
        "scene": {k: v for k, v in ident.items()},
        "grid": {
            "freq": grid["freq"],
            "epsg": grid["epsg"],
            "dx_m": grid["dx"],
            "dy_m": grid["dy"],
            "range_bandwidth_hz": grid["range_bandwidth_hz"],
        },
        "window": res["window"],
        "inside_px": res["inside_px"],
        "mask_rejected_px": res["mask_rejected_px"],
        "mask": minfo,
        "stats": res["stats"],
        "outputs": [str(png), str(png_ctx)],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {png}\nsaved {png_ctx}\nsaved {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
