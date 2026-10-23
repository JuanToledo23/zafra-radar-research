"""Exercise 1c: compare one field's backscatter between two dates (b minus a).

    python scripts/03_compare_two_scenes.py <h5_a> <h5_b> [--field data/fields/field_01.geojson]

Differences are taken between field means in dB (each mean computed in linear power first).
A difference in dB is a ratio in linear: -3 dB means b has half the power of a.
"""

import argparse
import sys

from zafra import gcov
from zafra.paths import FIELDS_DIR


def extract(path: str, field: str, freq: str) -> tuple[str, dict]:
    with gcov.open_gcov(path) as f:
        date = gcov.read_identification(f)["start_time"][:10]
        res = gcov.extract_field(f, field, freq)
    return date, res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("h5_a", help="earlier scene")
    p.add_argument("h5_b", help="later scene")
    p.add_argument("--field", default=str(FIELDS_DIR / "field_01.geojson"))
    p.add_argument("--freq", default="A")
    args = p.parse_args()

    date_a, a = extract(args.h5_a, args.field, args.freq)
    date_b, b = extract(args.h5_b, args.field, args.freq)

    # Same track/frame -> same grid, so the same polygon falls on the same pixels.
    if a["window"] != b["window"]:
        print("warning: field windows differ between scenes (different grids?)", file=sys.stderr)

    print(f"field {a['field_id']}   a = {date_a}   b = {date_b}\n")
    print(f"{'':<12}{'a: ' + date_a:>16}{'b: ' + date_b:>16}{'b - a':>10}")
    pols = [pol for pol in a["stats"] if pol in b["stats"]]
    diffs = {}
    for pol in pols:
        sa, sb = a["stats"][pol], b["stats"][pol]
        print(f"{pol}")
        for key, label in [
            ("valid_px", "valid_px"),
            ("mean_db", "mean_dB"),
            ("median_db", "median_dB"),
            ("std_db", "std_dB"),
            ("mean_looks", "mean_looks"),
        ]:
            va, vb = sa.get(key), sb.get(key)
            if va is None or vb is None:
                continue
            fmt = "{:>16d}" if key == "valid_px" else "{:>16.2f}"
            d = vb - va
            dfmt = "{:>+10d}" if key == "valid_px" else "{:>+10.2f}"
            print(f"  {label:<10}" + fmt.format(va) + fmt.format(vb) + dfmt.format(d))
        diffs[pol] = sb["mean_db"] - sa["mean_db"]

    print()
    for pol in ("HVHV", "HHHH"):
        if pol in diffs:
            d = diffs[pol]
            ratio = 10 ** (d / 10)
            print(f"{pol} change (b - a): {d:+.2f} dB  (b has {ratio:.2f}x the power of a)")

    # One sentence, driven by the sign of the cross-pol change (the canopy-volume signal).
    if "HVHV" in diffs:
        d = diffs["HVHV"]
        if d < 0:
            msg = (
                f"HV is {abs(d):.1f} dB lower on {date_b}: less backscatter in b, consistent with "
                "less standing canopy volume (e.g. the cane was cut between the dates)."
            )
        else:
            msg = (
                f"HV is {d:.1f} dB higher on {date_b}: more backscatter in b, consistent with more "
                "canopy volume or a wetter surface (regrowth, rain, residue), not a fresh harvest."
            )
        print(f"\nInterpretation: {msg}")
        # Rough uncertainty: std/sqrt(N) assumes independent pixels. Neighbouring 10 m pixels
        # share radar samples, so the true error is larger; provisional calibration can also
        # shift whole scenes by a fraction of a dB. Hence the conservative ~1 dB rule of thumb.
        s = a["stats"]["HVHV"]
        se = s["std_db"] / s["valid_px"] ** 0.5
        print(
            f"Caveat: {s['valid_px']} pixels, {s['std_db']:.1f} dB pixel spread -> naive standard "
            f"error {se:.2f} dB, but pixels are correlated and calibration is provisional, so "
            "treat changes under ~1 dB with caution. One field, two dates is not proof of harvest."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
