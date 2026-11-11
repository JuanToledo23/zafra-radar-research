"""Exercise 9 follow-up: L-band (NISAR) vs C-band (Sentinel-1) at harvest, Morelos fields.

    python scripts/19_l_vs_c.py [--force-pull]

1. Sentinel-1 GRD VH and VV field means (zafra.sentinel1: linear-power means, every
   track kept apart) for the 38 fields_v2 polygons, 2025-11-01 .. 2026-05-31, both passes,
   from Earth Engine (optical.init_ee). Cached in data/results/morelos/s1_series_2025_26.parquet
   (--force-pull refetches).
2. Drop at harvest for the 26 labelled cane fields, per sensor, polarization and track:
   median of the field-mean dB values in the 30 days before the field's NDVI harvest
   window (fields_v2 harvest_window_start/end, script 07: independent of both radars)
   minus the same after it; acquisitions inside the window are not used. NISAR from
   data/features/fields_v2_features.parquet (exercise 4b), ascending t19 and descending
   t113. Orbits and tracks are never averaged.
3. results/9/l_vs_c.md (median and IQR of the drop per series) and
   results/9/p36_l_vs_c.png (p36: NISAR HV and S1 VH series, harvest window, ground truth).
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from zafra import optical
from zafra import sentinel1 as S1
from zafra.paths import DATA_DIR, FEATURES_DIR, FIELDS_DIR, RESULTS_DIR

START, END = "2025-11-01", "2026-06-01"  # end exclusive: through 2026-05-31
WINDOW_D = 30  # days before / after the NDVI harvest window
FIELDS = FIELDS_DIR / "fields_v2.geojson"
S1_OUT = DATA_DIR / "results" / "morelos" / "s1_series_2025_26.parquet"
OUT = RESULTS_DIR / "9"
NISAR_TRACK = {"asc": 19, "desc": 113}
# validated categorical slots 1-5 (dataviz reference palette), fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def pull(fields: gpd.GeoDataFrame) -> pd.DataFrame:
    optical.init_ee()
    frames = []
    for i, (fid, geom) in enumerate(fields.geometry.items(), 1):
        df = S1.field_series(geom, START, END).assign(field_id=fid)
        frames.append(df)
        print(f"  S1 {i}/{len(fields)} {fid}: {len(df)} acquisitions", flush=True)
    df = pd.concat(frames, ignore_index=True)
    S1_OUT.parent.mkdir(parents=True, exist_ok=True)
    df = df[["field_id", *[c for c in df.columns if c != "field_id"]]]
    df.to_parquet(S1_OUT, index=False)
    return df


def long_series(s1: pd.DataFrame) -> pd.DataFrame:
    """One table: field_id, date, sensor, pol, series (sensor/orbit/track label), db."""
    nis = pd.read_parquet(FEATURES_DIR / "fields_v2_features.parquet")
    rows = []
    for pol, col in (("HV", "hv_db"), ("HH", "hh_db")):
        t = nis[["field_id", "date", "orbit", col]].rename(columns={col: "db"})
        t["series"] = (
            "NISAR " + pol + " " + t["orbit"] + " t" + t["orbit"].map(NISAR_TRACK).astype(str)
        )
        rows.append(t.assign(sensor="NISAR L", pol=pol))
    for pol, col in (("VH", "vh_db"), ("VV", "vv_db")):
        t = s1[["field_id", "date", "orbit", "rel_orbit", col]].rename(columns={col: "db"})
        t["series"] = "S1 " + pol + " " + t["orbit"] + " t" + t["rel_orbit"].astype(str)
        rows.append(t.assign(sensor="S1 C", pol=pol).drop(columns="rel_orbit"))
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out


def drops(series: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per field and series: n before / after, median before / after, drop (after - before)."""
    rows = []
    for (fid, name), s in series.groupby(["field_id", "series"]):
        if fid not in labels.index:
            continue
        w0, w1 = labels.loc[fid, "harvest_window_start"], labels.loc[fid, "harvest_window_end"]
        pre = s[(s["date"] >= w0 - pd.Timedelta(days=WINDOW_D)) & (s["date"] < w0)]["db"]
        post = s[(s["date"] > w1) & (s["date"] <= w1 + pd.Timedelta(days=WINDOW_D))]["db"]
        rows.append(
            {
                "field_id": fid,
                "series": name,
                "sensor": s["sensor"].iloc[0],
                "pol": s["pol"].iloc[0],
                "n_pre": len(pre),
                "n_post": len(post),
                "drop_db": post.median() - pre.median() if len(pre) and len(post) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summary(d: pd.DataFrame) -> pd.DataFrame:
    def agg(s):
        x = s["drop_db"].dropna()
        return pd.Series(
            {
                "fields": len(x),
                "median_drop_db": x.median(),
                "iqr_low": x.quantile(0.25),
                "iqr_high": x.quantile(0.75),
                "fields_le_m1db": int((x <= -1).sum()),
                "acq_pre_median": s["n_pre"].median(),
                "acq_post_median": s["n_post"].median(),
            }
        )

    order = {"HV": 0, "VH": 1, "HH": 2, "VV": 3}
    t = d.groupby(["series", "sensor", "pol"]).apply(agg, include_groups=False).reset_index()
    return t.sort_values(by="pol", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)


def md_table(t: pd.DataFrame) -> str:
    head = (
        "| series | fields | median drop (dB) | IQR (dB) | fields <= -1 dB | "
        "acq. before / after (median) |\n|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r.series} | {r.fields:.0f} | {r.median_drop_db:+.2f} | {r.iqr_low:+.2f} .. "
        f"{r.iqr_high:+.2f} | {r.fields_le_m1db:.0f}/{r.fields:.0f} | "
        f"{r.acq_pre_median:.0f} / {r.acq_post_median:.0f} |\n"
        for r in t.itertuples()
    )
    return head + body


def plot_p36(series: pd.DataFrame, labels: pd.DataFrame, gt, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = series[(series["field_id"] == "p36") & series["pol"].isin(["HV", "VH"])]
    names = sorted(s["series"].unique(), key=lambda n: (not n.startswith("NISAR"), n))
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    w0, w1 = labels.loc["p36", ["harvest_window_start", "harvest_window_end"]]
    ax.axvspan(w0, w1, color="#d9d9d9", lw=0, label="NDVI harvest window")
    ax.axvline(gt, color="#555555", lw=1, ls=":", label="ground truth: burn + cut (02-26)")
    for name, c in zip(names, COLORS, strict=False):
        t = s[s["series"] == name].sort_values("date")
        nisar = name.startswith("NISAR")
        ax.plot(t["date"], t["db"], color=c, lw=2, ls="-" if nisar else "--",
                marker="o" if nisar else "s", ms=4.5, label=name)  # fmt: skip
    ax.set_ylabel("field mean backscatter (dB)")
    ax.set_title("p36 La Marometa (21 ha), zafra 2025-26: NISAR L-band HV vs Sentinel-1 C-band VH")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="lower left")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force-pull", action="store_true", help="refetch S1 from Earth Engine")
    args = ap.parse_args()

    f = gpd.read_file(FIELDS).set_index("id")
    if S1_OUT.exists() and not args.force_pull:
        s1 = pd.read_parquet(S1_OUT)
        print(f"S1 series reused from {S1_OUT.name} (--force-pull to refetch)")
    else:
        s1 = pull(f.to_crs(4326))
    print(f"S1: {len(s1)} rows, {s1['field_id'].nunique()} fields, tracks "
          f"{sorted(s1.groupby(['orbit', 'rel_orbit']).groups)}")  # fmt: skip

    cane = f[f["label"] == "cana"].copy()
    for c in ("harvest_window_start", "harvest_window_end"):
        cane[c] = pd.to_datetime(cane[c])
    series = long_series(s1)
    d = drops(series, cane)
    t = summary(d)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "l_vs_c_drops.csv", index=False, float_format="%.3f")
    gt = pd.Timestamp(f.loc["p36", "ground_truth_date_2026"])
    png = OUT / "p36_l_vs_c.png"
    plot_p36(series, cane, gt, png)

    hv = d[d["series"] == "NISAR HV asc t19"].set_index("field_id")["drop_db"]
    paired = {
        name: (hv - d[d["series"] == name].set_index("field_id")["drop_db"]).dropna()
        for name in sorted(d.loc[d["pol"] == "VH", "series"].unique())
    }
    pair_lines = "".join(
        f"* NISAR HV asc t19 minus {k}: median {v.median():+.2f} dB over {len(v)} fields, "
        f"HV deeper on {int((v < 0).sum())}\n"
        for k, v in paired.items()
    )
    tracks = s1.groupby(["orbit", "rel_orbit"]).agg(
        acq=("date", "nunique"), platforms=("platform", lambda p: "".join(sorted(set(p))))
    )
    track_lines = "".join(
        f"* {o} t{r}: {row.acq} dates, Sentinel-1{row.platforms}\n"
        for (o, r), row in tracks.iterrows()
    )
    md = f"""# Exercise 9: L-band (NISAR) vs C-band (Sentinel-1) at harvest

Generated by `scripts/19_l_vs_c.py`. 26 labelled cane fields (fields_v2, `label == cana`,
p36 included), zafra 2025-26.

**Drop at harvest** = median field-mean dB in the {WINDOW_D} days *after* the field's NDVI
harvest window minus the median in the {WINDOW_D} days *before* it (script 07 window,
optical, independent of both radars); acquisitions inside the window are not used.
Negative = darker after the cut. Each orbit / track is its own series; none is averaged.
Field means in linear power, then dB, for both sensors.

Sentinel-1 GRD (`COPERNICUS/S1_GRD`, IW, sigma-0; NISAR is gamma-0, so absolute levels are
not comparable, drops are), {START} .. 2026-05-31,
`data/results/morelos/s1_series_2025_26.parquet` ({len(s1)} rows, 38 fields):
{track_lines}
## Cross-pol (volume scattering: the harvest signal)

{md_table(t[t["pol"].isin(["HV", "VH"])])}
Paired, per field:
{pair_lines}
## Co-pol

{md_table(t[t["pol"].isin(["HH", "VV"])])}
Per field and series: `results/9/l_vs_c_drops.csv`.

`{png.relative_to(OUT.parent.parent)}`: p36, NISAR HV (solid) and S1 VH (dashed) per
orbit / track, NDVI harvest window shaded, ground truth dotted.
"""
    (OUT / "l_vs_c.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
