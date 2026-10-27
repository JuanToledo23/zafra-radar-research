"""Per-field, per-zafra features for the cane / non-cane gate (exercise 4c).

One row per (field_id, zafra). A zafra is Nov 1 .. May 31 ("2024-25", "2025-26").

Which time window each group uses
---------------------------------
* **Radar** (NISAR GCOV asc + desc, GUNW asc coherence): the zafra itself. Every NISAR
  acquisition on disk is between 2025-11-18 and 2026-05-22, so only 2025-26 has radar;
  `has_radar` says so per row and radar features are NaN otherwise.
* **Optical** (Sentinel-2 NDVI): the *crop year* that ends with the zafra, Jun 1 .. May 31.
  The long closed-canopy plateau that marks cane (script 07) lies mostly in the rainy season
  *before* the cut, and an annual crop's rainy-season harvest is exactly what separates it
  from cane. Counting only inside Nov-May would hide both.
* **Fire** (VIIRS): the zafra, detections whose pixel centre is <= 200 m from the polygon.

Why these radar features
------------------------
Cane is a tall, dense canopy, a strong L-band volume scatterer, so HV is high before the
cut (hv_pre_level_db). The cut removes that volume in one step: one large HV drop between
two consecutive acquisitions of the same orbit (hv_max_single_drop_db), and the 12-day
pair that spans it decorrelates (coh_min). The two should coincide in time (joint_collapse).
Pasture, scrub and built-up land drift slowly over the dry season or stay flat, so their
largest single step is small and not tied to a coherence minimum.

Orbits are never mixed in a series (multisignal.py): per-orbit statistics are computed
separately and then averaged (range, std, levels) or the extreme is taken (largest drop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZAFRAS = {  # end exclusive
    "2024-25": ("2024-11-01", "2025-06-01"),
    "2025-26": ("2025-11-01", "2026-06-01"),
}
HIGH = 0.6  # closed green canopy (script 07)
LOW = 0.35  # bare soil / stubble / ash (script 07)
COLLAPSE_DROP = 0.3  # NDVI drop that counts as a collapse ...
COLLAPSE_DAYS = 30  # ... relative to the highest clear scene in the previous 30 days
JOINT_DAYS = 12  # HV drop and coherence minimum within one revisit -> joint collapse
FIRE_BUFFER_M = 200
PRE_MONTHS = (11, 12)  # Nov-Dec: cane still standing everywhere here
MAX_FIELD_CLOUD = 10  # % of the field masked for an S2 scene to count as clear (script 07)

RADAR_COLUMNS = [
    "hv_range_db",
    "hv_std_db",
    "hh_std_db",
    "cross_ratio_mean_db",
    "cross_ratio_std_db",
    "hv_max_single_drop_db",
    "coh_min",
    "coh_mean",
    "coh_range",
    "joint_collapse",
    "hv_pre_level_db",
    "enl_median",
]
OPTICAL_COLUMNS = [
    "ndvi_max",
    "ndvi_min",
    "ndvi_days_above_0_6",
    "ndvi_n_collapses",
    "ndvi_longest_high_run_days",
    "ndvi_collapse_month",
    "bare_share_after_collapse",
]
TABLE_COLUMNS = [
    "field_id",
    "zafra",
    "has_radar",
    "n_radar_acq",
    *RADAR_COLUMNS,
    "hv_drop_date",
    "coh_min_date",
    "has_optical",
    "n_clear_scenes",
    *OPTICAL_COLUMNS,
    "collapse_date",
    "collapse_low_image_id",
    "fire_count_in_zafra",
]


def zafra_bounds(zafra: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end exclusive) of a zafra; accepts '2025-26' or '2025/26'."""
    a, b = ZAFRAS[zafra.replace("/", "-")]
    return pd.Timestamp(a), pd.Timestamp(b)


def crop_year_bounds(zafra: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Jun 1 before the zafra .. end of the zafra (end exclusive)."""
    a, b = zafra_bounds(zafra)
    return pd.Timestamp(year=a.year, month=6, day=1), b


# --- radar (pure) -------------------------------------------------------------------------


def radar_summary(acq: pd.DataFrame, coh: pd.DataFrame) -> dict:
    """Zafra summary of one field's radar series.

    acq: rows of one field inside the zafra with date, orbit, hv_db, hh_db, cross_ratio_db,
         enl (the exercise-4b feature table layout).
    coh: that field's coherence pairs inside the zafra with sec_date, coh_mean (field mean;
         the per-pixel minimum is noise, see multisignal.COH_COLUMNS).
    """
    out = dict.fromkeys(RADAR_COLUMNS, np.nan) | {
        "has_radar": False,
        "n_radar_acq": int(len(acq)),
        "hv_drop_date": pd.NaT,
        "coh_min_date": pd.NaT,
    }
    acq = acq.dropna(subset=["hv_db"])
    if acq.empty:
        return out
    per = []
    best_drop, drop_date = np.inf, pd.NaT
    for _, d in acq.groupby("orbit", sort=True):
        d = d.sort_values("date")
        hv = d["hv_db"].to_numpy()
        pre = d.loc[d["date"].dt.month.isin(PRE_MONTHS), "hv_db"]
        per.append(
            {
                "hv_range_db": np.nanmax(hv) - np.nanmin(hv),
                "hv_std_db": d["hv_db"].std(),
                "hh_std_db": d["hh_db"].std(),
                "cross_ratio_mean_db": d["cross_ratio_db"].mean(),
                "cross_ratio_std_db": d["cross_ratio_db"].std(),
                "hv_pre_level_db": pre.median() if len(pre) else np.nan,
            }
        )
        steps = np.diff(hv)
        if len(steps) and np.nanmin(steps) < best_drop:
            i = int(np.nanargmin(steps))
            best_drop, drop_date = float(steps[i]), d["date"].iloc[i + 1]
    out.update(pd.DataFrame(per).mean().to_dict())
    out["hv_max_single_drop_db"] = best_drop if np.isfinite(best_drop) else np.nan
    out["hv_drop_date"] = drop_date
    out["enl_median"] = acq["enl"].median()
    out["has_radar"] = True

    c = coh.dropna(subset=["coh_mean"])
    if len(c):
        k = c["coh_mean"].idxmin()
        out["coh_min"] = c.loc[k, "coh_mean"]
        out["coh_mean"] = c["coh_mean"].mean()
        out["coh_range"] = c["coh_mean"].max() - c["coh_mean"].min()
        out["coh_min_date"] = c.loc[k, "sec_date"]
        if pd.notna(drop_date):
            gap = abs((pd.Timestamp(drop_date) - pd.Timestamp(out["coh_min_date"])).days)
            out["joint_collapse"] = int(gap <= JOINT_DAYS)
    return out


# --- optical (pure) -----------------------------------------------------------------------


def longest_high_run(dates, ndvi, high: float = HIGH) -> int:
    """Longest run of consecutive clear scenes with NDVI >= high, in days (first -> last).

    Same rule as scripts/07_label_fields.py: only a clear scene below `high` breaks a run.
    """
    best, start = 0, None
    for t, v in zip(pd.to_datetime(pd.Series(dates)), ndvi, strict=True):
        if v >= high:
            start = t if start is None else start
            best = max(best, (t - start).days)
        else:
            start = None
    return best


def days_above(dates, ndvi, high: float = HIGH) -> int:
    """Days covered by runs of clear scenes >= high (sum of every run's first -> last span)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    v = np.asarray(ndvi, dtype="float64")
    total = 0
    for i in range(1, len(v)):
        if v[i] >= high and v[i - 1] >= high:
            total += (dates[i] - dates[i - 1]).days
    return total


def ndvi_collapses(dates, ndvi, drop=COLLAPSE_DROP, days=COLLAPSE_DAYS, high=HIGH) -> list[dict]:
    """Every collapse: a clear scene more than `drop` below the highest clear scene of the
    previous `days` days. After one, the detector re-arms only once NDVI is back >= `high`,
    so the weeks of stubble after a cut count once.

    Returns dicts with date (first scene after the drop) and size (positive NDVI drop).
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    v = np.asarray(ndvi, dtype="float64")
    out, armed = [], True
    for i in range(len(v)):
        if v[i] >= high:
            armed = True
        prev = (dates < dates[i]) & (dates >= dates[i] - pd.Timedelta(days=days))
        if armed and prev.any():
            size = v[prev.to_numpy()].max() - v[i]
            if size > drop:
                out.append({"date": dates[i], "size": float(size), "i": i})
                armed = False
    return out


def optical_summary(clear: pd.DataFrame, zafra: str) -> dict:
    """Crop-year summary of one field's clear Sentinel-2 scenes (date, ndvi_mean, image_id).

    The main collapse is the largest one; bare_share_after_collapse is filled in later from
    the scene `collapse_low_image_id` (lowest clear NDVI within 30 days after the collapse).
    """
    a, b = crop_year_bounds(zafra)
    c = clear[(clear["date"] >= a) & (clear["date"] < b)].sort_values("date")
    c = c.reset_index(drop=True)
    out = dict.fromkeys(OPTICAL_COLUMNS, np.nan) | {
        "has_optical": False,
        "n_clear_scenes": int(len(c)),
        "collapse_date": pd.NaT,
        "collapse_low_image_id": None,
    }
    if c.empty:
        return out
    v = c["ndvi_mean"]
    cols = ndvi_collapses(c["date"], v)
    out |= {
        "has_optical": True,
        "ndvi_max": v.max(),
        "ndvi_min": v.min(),
        "ndvi_days_above_0_6": days_above(c["date"], v),
        "ndvi_n_collapses": len(cols),
        "ndvi_longest_high_run_days": longest_high_run(c["date"], v),
    }
    if cols:
        main = max(cols, key=lambda x: x["size"])
        after = c[(c["date"] >= main["date"])]
        after = after[after["date"] <= main["date"] + pd.Timedelta(days=COLLAPSE_DAYS)]
        low = after.loc[after["ndvi_mean"].idxmin()]
        out["collapse_date"] = main["date"]
        out["ndvi_collapse_month"] = main["date"].month
        if "image_id" in c:
            out["collapse_low_image_id"] = low["image_id"]
    return out


def fire_count(fire_table: pd.DataFrame, field_id: str, zafra: str) -> int:
    """VIIRS detections <= FIRE_BUFFER_M from the polygon edge, dated (UTC) in the zafra."""
    a, b = zafra_bounds(zafra)
    f = fire_table[
        (fire_table["field_id"] == field_id) & (fire_table["dist_edge_m"] <= FIRE_BUFFER_M)
    ]
    t = pd.to_datetime(f["acq_utc"], utc=True).dt.tz_localize(None)
    return int(((t >= a) & (t < b)).sum())


# --- assembly -----------------------------------------------------------------------------


def build_table(
    field_ids,
    zafras,
    acq: pd.DataFrame | None,
    coh: pd.DataFrame | None,
    ndvi: pd.DataFrame | None,
    fire_table: pd.DataFrame | None,
    bare: dict | None = None,
) -> pd.DataFrame:
    """One row per (field, zafra) with TABLE_COLUMNS.

    acq:  per-acquisition radar table (field_id, date, orbit, hv_db, hh_db, cross_ratio_db,
          enl), e.g. data/features/fields_v2_features.parquet. None -> no radar.
    coh:  multisignal.extract_coherence_series output. None -> no coherence.
    ndvi: optical.s2_ndvi_series rows with an `id` column (all scenes; filtered to clear
          here). None -> no optical.
    fire_table: fires.fires_per_field output (any buffer >= 200 m). None -> fire count NaN.
    bare: {(field_id, image_id): share} from optical.s2_bare_share; missing -> NaN.
    """
    empty_acq = pd.DataFrame(columns=["field_id", "date", "orbit", "hv_db"])
    empty_coh = pd.DataFrame(columns=["field_id", "sec_date", "coh_mean"])
    acq = empty_acq if acq is None else acq
    coh = empty_coh if coh is None else coh
    if ndvi is not None:
        ok = (ndvi["cloud_pct"] <= MAX_FIELD_CLOUD) & ndvi["ndvi_mean"].notna()
        ndvi = ndvi[ok]
    bare = bare or {}
    rows = []
    for fid in field_ids:
        fa = acq[acq["field_id"] == fid]
        fc = coh[coh["field_id"] == fid]
        fn = None if ndvi is None else ndvi[ndvi["id"] == fid]
        for z in zafras:
            a, b = zafra_bounds(z)
            r = radar_summary(
                fa[(fa["date"] >= a) & (fa["date"] < b)],
                fc[(fc["sec_date"] >= a) & (fc["sec_date"] < b)],
            )
            o = (
                optical_summary(fn, z)
                if fn is not None
                else optical_summary(pd.DataFrame(columns=["date", "ndvi_mean"]), z)
            )
            if o["collapse_low_image_id"] is not None:
                o["bare_share_after_collapse"] = bare.get((fid, o["collapse_low_image_id"]), np.nan)
            fires = np.nan if fire_table is None else fire_count(fire_table, fid, z)
            rows.append({"field_id": fid, "zafra": z, **r, **o, "fire_count_in_zafra": fires})
    return pd.DataFrame(rows).reindex(columns=TABLE_COLUMNS)


# --- inputs from disk (NISAR, Earth Engine, FIRMS) -------------------------------------------


def load_radar_inputs(fields_path, cache_dir, asc_dir, desc_dir, gunw_dir, force=False):
    """(acq, coh) for the polygons in `fields_path`, cached per source in `cache_dir`.

    Uses the same extractors and cache names as scripts 08/10 (gcov_asc/desc, enl_asc/desc,
    coherence .parquet), so pointing `cache_dir` at outputs/multisignal reuses the
    fields_v2 caches. A cache is reused only if it holds exactly these ids and dates.
    """
    from pathlib import Path

    from zafra import features as FT
    from zafra import multisignal as ms
    from zafra import timeseries as ts

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fields = ts.load_fields(fields_path)
    ids = set(fields["id"])

    def cached(name, build, col, dates):
        path = cache_dir / f"{name}.parquet"
        if path.exists() and not force:
            df = pd.read_parquet(path)
            if set(df["field_id"]) == ids and set(df[col].dt.date.astype(str)) == set(dates):
                return df
        df = build()
        df.to_parquet(path, index=False)
        return df

    frames = []
    for o, d in (("asc", asc_dir), ("desc", desc_dir)):
        dates = [s["date"] for s in ts.list_scenes(d)]
        g = cached(
            f"gcov_{o}", lambda d=d, o=o: ms.extract_gcov_series(d, fields_path, o), "date", dates
        )
        e = cached(f"enl_{o}", lambda d=d, o=o: FT.enl_table(d, fields_path, o), "date", dates)
        frames.append(g.merge(e[[*FT.KEYS, "enl"]], on=FT.KEYS, how="left"))
    acq = pd.concat(frames, ignore_index=True).rename(
        columns={"HVHV_db": "hv_db", "HHHH_db": "hh_db", "ratio_db": "cross_ratio_db"}
    )
    pairs = [p["sec_date"] for p in ms.list_pairs(gunw_dir)]
    coh = cached(
        "coherence", lambda: ms.extract_coherence_series(gunw_dir, fields), "sec_date", pairs
    )
    return acq, coh


def load_ndvi(fields, cache_csv, start: str, end: str, max_tile_cloud: float = 50, force=False):
    """S2 NDVI series (optical.s2_ndvi_series) for every polygon, long format with `id`.

    Cached in `cache_csv`; only polygons missing from the cache are fetched.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from zafra import optical

    cache_csv = Path(cache_csv)
    have = pd.DataFrame()
    if cache_csv.exists() and not force:
        have = pd.read_csv(cache_csv, parse_dates=["date"])
    missing = [r for r in fields.itertuples() if have.empty or r.id not in set(have["id"])]
    if missing:
        optical.init_ee()

        def one(r):
            return optical.s2_ndvi_series(r.geometry, start, end, max_cloud=max_tile_cloud).assign(
                id=r.id
            )

        with ThreadPoolExecutor(max_workers=6) as ex:
            new = list(ex.map(one, missing))
        have = pd.concat([have, *new], ignore_index=True)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache_csv, index=False, date_format="%Y-%m-%d")
    return have[have["id"].isin(set(fields["id"]))].reset_index(drop=True)


def load_bare_shares(fields, requests, cache_csv) -> dict:
    """{(id, image_id): share of clear pixels with NDVI <= LOW}, cached in `cache_csv`."""
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from zafra import optical

    cache_csv = Path(cache_csv)
    have = (
        pd.read_csv(cache_csv)
        if cache_csv.exists()
        else pd.DataFrame(columns=["id", "image_id", "bare_share"])
    )
    known = set(zip(have["id"], have["image_id"], strict=True))
    todo = [(f, i) for f, i in requests if (f, i) not in known]
    if todo:
        optical.init_ee()
        geom = fields.set_index("id").geometry

        def one(req):
            f, i = req
            return {"id": f, "image_id": i, "bare_share": optical.s2_bare_share(geom[f], i, LOW)}

        with ThreadPoolExecutor(max_workers=6) as ex:
            new = pd.DataFrame(list(ex.map(one, todo)))
        have = pd.concat([have, new], ignore_index=True)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache_csv, index=False)
    return {(r.id, r.image_id): r.bare_share for r in have.itertuples()}


def fire_table_for(fields) -> pd.DataFrame:
    """fires.fires_per_field for arbitrary polygons, from the raw FIRMS CSVs."""
    from zafra import fires
    from zafra.paths import FIRMS_DIR

    return fires.fires_per_field(fires.load_firms(FIRMS_DIR), fields, buffer_m=FIRE_BUFFER_M)
