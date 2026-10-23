"""Exercise 1a: look inside a NISAR GCOV file without loading any image data.

    python scripts/01_inspect_gcov.py data/nisar/<granule>.h5

Prints (1) who/when/where the scene is, (2) the HDF5 tree of the parts we care about,
(3) a summary of the map grid, (4) a plain-language "what this means" block.
"""

import argparse
import sys

import h5py
import numpy as np

from zafra import gcov

# Groups to walk. Image layers are listed (name, shape, dtype) but never read.
WALK = [gcov.IDENT, f"{gcov.METADATA}", gcov.GRIDS]


def short_value(ds: h5py.Dataset) -> str:
    """Show the value of tiny datasets (scalars, short lists); skip anything bigger."""
    if ds.size > 4:
        return ""
    v = gcov._decode(ds[()])
    s = repr(v)
    return s if len(s) <= 60 else s[:57] + "..."


def walk(f: h5py.File, root: str) -> None:
    print(f"\n{root}")

    def visit(name, obj):
        depth = name.count("/")
        indent = "  " * (depth + 1)
        leaf = name.rsplit("/", 1)[-1]
        if isinstance(obj, h5py.Group):
            print(f"{indent}{leaf}/")
        else:
            print(f"{indent}{leaf:<45} {str(obj.shape):<16} {str(obj.dtype):<8} {short_value(obj)}")

    f[root].visititems(visit)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("h5", help="path to a NISAR GCOV .h5 file")
    args = p.parse_args()

    with gcov.open_gcov(args.h5) as f:
        # 1. Identification -------------------------------------------------------------
        ident = gcov.read_identification(f)
        print("=" * 80, "\nIDENTIFICATION\n" + "=" * 80)
        for k, v in ident.items():
            if k != "missing":
                print(f"  {k:<22} {v}")
        print(f"  not in file: {ident['missing'] or 'none'}")

        # 2. Tree -----------------------------------------------------------------------
        print("\n" + "=" * 80, "\nHDF5 TREE (name, shape, dtype, value if tiny)\n" + "=" * 80)
        for root in WALK:
            walk(f, root)

        # 3. Grid summary (frequency A = the 40 MHz main band we will use) ---------------
        print("\n" + "=" * 80, "\nGRID SUMMARY\n" + "=" * 80)
        for freq in gcov._decode(f[f"{gcov.IDENT}/listOfFrequencies"][()]):
            g = gcov.read_grid(f, freq)
            looks_shape = None
            if g["has_number_of_looks"]:
                looks_shape = f[f"{g['path']}/numberOfLooks"].shape
            print(f"frequency {freq}")
            print(f"  polarizations      {g['polarizations']}")
            print(f"  EPSG               {g['epsg']}")
            print(f"  shape (rows, cols) {g['shape']}")
            print(
                f"  x range (m)        {g['x'][0]:.1f} .. {g['x'][-1]:.1f}"
                f"  (coordinates are pixel {g['pixel_convention']}s)"
            )
            print(f"  y range (m)        {g['y'][0]:.1f} .. {g['y'][-1]:.1f}")
            print(f"  pixel spacing (m)  dx={g['dx']}, dy={g['dy']}")
            print(f"  range bandwidth    {g['range_bandwidth_hz'] / 1e6:g} MHz")
            print(f"  numberOfLooks      {looks_shape}")
            print(f"  mask present       {g['has_mask']}")
            if g["has_mask"]:
                print(f"  mask description   {gcov.mask_info(f, freq)['description']}")

        # Values for the explanation block, all read from the file.
        gA = gcov.read_grid(f, "A")
        md = gcov.METADATA
        norm = gcov._decode(
            f[
                f"{md}/processingInformation/parameters/rtc/outputBackscatterNormalizationConvention"
            ][()]
        )
        expr = gcov._decode(
            f[f"{md}/processingInformation/parameters/rtc/outputBackscatterExpressionConvention"][
                ()
            ]
        )
        sw = f"{md}/sourceData/swaths/frequencyA"
        inc_near = float(f[f"{sw}/nearRangeIncidenceAngle"][()])
        inc_far = float(f[f"{sw}/farRangeIncidenceAngle"][()])
        rng_res = float(f[f"{sw}/rangeResolution"][()])
        az_res = float(f[f"{sw}/sceneCenterAlongTrackResolution"][()])
        masking_applied = gcov._decode(
            f[f"{md}/processingInformation/parameters/validSamplesSubSwathMaskingApplied"][()]
        )
        # Slant-range resolution projects onto the ground as res / sin(incidence angle).
        gr_near = rng_res / np.sin(np.radians(inc_near))
        gr_far = rng_res / np.sin(np.radians(inc_far))
        width_km = (gA["x"][-1] - gA["x"][0]) / 1000
        height_km = (gA["y"][0] - gA["y"][-1]) / 1000
        layer_gb = np.prod(gA["shape"]) * 4 / 1e9

    print("\n" + "=" * 80, "\nWHAT THIS MEANS\n" + "=" * 80)
    lines = [
        f"Scene: track {ident['track']}, frame {ident['frame']}, "
        f"{ident['orbit_direction'].lower()} pass, radar looking "
        f"{ident['look_direction'].lower()}, acquired {ident['start_time'][:19]} UTC.",
        "Same track+frame+direction on another date = same viewing geometry, so pixels are "
        "directly comparable over time.",
        f"Pixels are {norm} backscatter as {expr.lower()}: unitless power ratios; convert to dB "
        "with 10*log10 after averaging.",
        f"Polarizations {', '.join(gA['polarizations'])}: HH = co-pol (surface + stalks), "
        "HV = cross-pol (volume scattering in the canopy, the best harvest signal).",
        f"Grid: EPSG:{gA['epsg']} (UTM, metres), {abs(gA['dx']):g} m pixels, "
        f"bounding box ~{width_km:.0f} x {height_km:.0f} km (the swath is a tilted strip inside "
        f"it, the rest is fill); one layer is ~{layer_gb:.1f} GB, so always read windows.",
        f"Native resolution is {rng_res:.1f} m in slant range (= {gr_far:.1f}-{gr_near:.1f} m "
        f"on the ground, far to near range) x {az_res:.1f} m along track; the {abs(gA['dx']):g} m "
        "grid averages several radar samples ('looks') per pixel to reduce speckle.",
        f"Incidence angle spans {inc_near:.1f}-{inc_far:.1f} deg across the swath; a given field "
        "always sits at about the same angle on this track.",
        f"Range bandwidth {gA['range_bandwidth_hz'] / 1e6:g} MHz (frequency A) sets the range "
        "resolution; frequency B (5 MHz) is a coarse side band we ignore.",
        f"mask layer: 0 = invalid sample, 255 = outside the swath, other = valid subswath. "
        f"Masking already applied to the pixels: {masking_applied} -> apply it ourselves.",
        f"Maturity is not recorded in the file (CRID {ident['crid']} = composite release); "
        "it comes from the ASF collection name (PROVISIONAL).",
    ]
    for line in lines:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
