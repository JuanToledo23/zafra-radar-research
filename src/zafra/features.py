"""Per-acquisition feature table for the trained harvest detector (exercise 4b).

One row per (field_id, acquisition_date, orbit). Ascending (t19 f11, ~06:05 local) and
descending (t113 f79, ~19:13 local the day before) stay separate rows with an `orbit` column;
they are never averaged (different geometry and time of day, see multisignal.py).

What each feature is for
------------------------
* **Field-relative HV.** Absolute backscatter differs between fields (row direction, soil,
  variety), so the classifier sees HV only as a change: vs the field's own pre-season
  baseline (median of the first 3 acquisitions of that orbit in Nov-Dec) and vs the
  previous acquisition of the same orbit.
* **Drying vs cutting.** Cane drying in the field before harvest lowers HV slowly over
  several scenes; a cut removes the canopy between two scenes. `slope_hv_3` is the local
  trend (dB/day, least squares over the previous 3 acquisitions, not including this one);
  `sharpness` = delta_hv_vs_prev - slope_hv_3 * gap_days is how much larger the last step
  was than that trend predicts. A ramp gives ~0; a cut gives a large negative value.
* **Coherence.** HH coherence of the 12-day ascending GUNW pair that *ends* on this
  acquisition (the first date on which the change it measures is observable). A cut
  rearranges every scatterer, so the pair spanning it decorrelates; slow drying does not.
  NaN on descending rows (no descending GUNW downloaded).
* **Fire.** VIIRS detections whose pixel centre is within 200 m of the polygon in the 7
  days before the acquisition (cane is often burned just before it is cut). Read from
  data/fires/fields_v2_fires.csv (scripts/05_fires_per_field.py).
* **Rain.** GPM IMERG V07 total over the 3 days before the acquisition at the field centroid
  (wet soil raises backscatter and its drying mimics a drop). ~11 km pixels: all fields here
  share one or two pixels. NaN if Earth Engine is unavailable.

Everything is causal: a row uses only its own and earlier acquisitions (the baseline uses
the first three of the season, which precede every harvest in this data set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zafra import timeseries as ts

BASELINE_N = 3
BASELINE_MONTHS = (11, 12)  # Nov-Dec: start of the zafra, cane still standing
SLOPE_N = 3
FIRE_DAYS = 7
FIRE_BUFFER_M = 200
RAIN_DAYS = 3
GPM_COLLECTION = "NASA/GPM_L3/IMERG_V07"
GPM_SCALE_M = 11132  # native 0.1 degree

KEYS = ["field_id", "date", "orbit"]
FEATURE_TABLE_COLUMNS = [
    *KEYS,
    "acq_utc",
    "hv_db",
    "hh_db",
    "cross_ratio_db",
    "valid_px",
    "n_looks",
    "enl",
    "delta_hv_vs_baseline",
    "delta_hv_vs_prev",
    "slope_hv_3",
    "sharpness",
    "coherence_mean",
    "coherence_min",
    "delta_coh_vs_baseline",
    "fire_7d",
    "fire_frp_max_7d",
    "rain_3d_mm",
    "gap_days",
]


# --- pure helpers (unit-tested) -----------------------------------------------------------


def season_baseline(
    dates: pd.Series, values: pd.Series, n: int = BASELINE_N, months=BASELINE_MONTHS
) -> float:
    """Median of the first `n` non-NaN values dated in `months` (default Nov-Dec; a site's
    zafra.baseline_months), fallback: first `n` overall."""
    dates, values = pd.to_datetime(pd.Series(dates)).reset_index(drop=True), pd.Series(values)
    values = values.reset_index(drop=True)
    ok = values.notna()
    early = ok & dates.dt.month.isin(months)
    pick = values[early] if early.sum() >= n else values[ok]
    return float(pick.iloc[:n].median()) if len(pick) else np.nan


def trailing_slope(t_days: np.ndarray, y: np.ndarray, k: int = SLOPE_N) -> np.ndarray:
    """Least-squares slope of y over the k points *before* each index (NaN if fewer)."""
    t_days, y = np.asarray(t_days, dtype="float64"), np.asarray(y, dtype="float64")
    out = np.full(len(y), np.nan)
    for i in range(k, len(y)):
        tt, yy = t_days[i - k : i], y[i - k : i]
        if np.isfinite(yy).all():
            out[i] = np.polyfit(tt, yy, 1)[0]
    return out


def hv_dynamics(dates, hv_db, baseline_months=BASELINE_MONTHS) -> pd.DataFrame:
    """Baseline/step/trend/sharpness features for one field and orbit, sorted by date."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    hv = pd.Series(hv_db, dtype="float64").reset_index(drop=True)
    t = (dates - dates.iloc[0]).dt.total_seconds().to_numpy() / 86400
    gap = np.r_[np.nan, np.diff(t)]
    prev = hv.diff().to_numpy()
    slope = trailing_slope(t, hv.to_numpy())
    return pd.DataFrame(
        {
            "delta_hv_vs_baseline": hv.to_numpy()
            - season_baseline(dates, hv, months=baseline_months),
            "delta_hv_vs_prev": prev,
            "slope_hv_3": slope,
            "sharpness": prev - slope * gap,
            "gap_days": gap,
        }
    )


def fire_window_stats(fire_times: pd.Series, frp: pd.Series, acq_times, days: int = FIRE_DAYS):
    """(count, max FRP) of detections in [t - days, t) for each acquisition time t."""
    fire_times = pd.to_datetime(pd.Series(fire_times), utc=True).reset_index(drop=True)
    frp = np.asarray(frp, dtype="float64")
    counts, frp_max = [], []
    for t in pd.to_datetime(pd.Series(acq_times), utc=True):
        sel = ((fire_times >= t - pd.Timedelta(days=days)) & (fire_times < t)).to_numpy()
        counts.append(int(sel.sum()))
        frp_max.append(float(frp[sel].max()) if sel.any() else 0.0)
    return np.array(counts), np.array(frp_max)


# --- inputs from disk ---------------------------------------------------------------------


def acquisition_times(scene_dir, orbit: str) -> pd.DataFrame:
    """date (UTC day) -> exact acquisition start (UTC), from the GCOV file names."""
    rows = [
        {"date": pd.Timestamp(s["date"]), "orbit": orbit, "acq_utc": s["start_utc"]}
        for s in ts.list_scenes(scene_dir)
    ]
    return pd.DataFrame(rows)


def enl_table(scene_dir, fields_geojson, orbit: str) -> pd.DataFrame:
    """Per-(field, date) HV pixel count and ENL (mean^2 / var of linear pixel intensities).

    Same pixel selection as the series (ratio_test.GcovPixelProvider: centre inside the
    polygon, GCOV mask valid, finite and > 0). The mean is cross-checked against the series
    via gcov.field_stats by the caller.
    """
    from zafra import gcov
    from zafra.detect.ratio_test import GcovPixelProvider

    prov = GcovPixelProvider(scene_dir, fields_geojson)
    rows = []
    for date in sorted(prov.paths):
        for fid in prov.fields["id"]:
            arr, valid = prov.window(fid, date)
            st = gcov.field_stats(arr, valid)
            px = arr[valid]
            enl = float(px.mean() ** 2 / px.var(ddof=1)) if px.size >= 2 else np.nan
            rows.append(
                {
                    "field_id": fid,
                    "date": pd.Timestamp(date),
                    "orbit": orbit,
                    "enl": enl,
                    "enl_px": st["valid_px"],
                    "enl_mean_db": st.get("mean_db", np.nan),
                }
            )
    return pd.DataFrame(rows)


def gpm_rain(acq: pd.DataFrame, fields, days: int = RAIN_DAYS) -> pd.DataFrame | None:
    """GPM IMERG V07 rain (mm) in the `days` before each acquisition, at each field centroid.

    acq: rows of (date, orbit, acq_utc). One Earth Engine call per acquisition (all fields
    at once). Returns field_id, date, orbit, rain_3d_mm, or None if Earth Engine fails
    (auth, network, quota): the caller then leaves rain as NaN and records it.
    IMERG `precipitation` is a rate in mm/hr per 30-min image, so the sum is x 0.5 h.
    """
    try:
        import ee

        from zafra import optical

        optical.init_ee()
        cents = fields.to_crs("EPSG:32614").centroid.to_crs("EPSG:4326")
        fc = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point([p.x, p.y]), {"field_id": fid})
                for fid, p in zip(fields["id"], cents, strict=True)
            ]
        )
        rows = []
        for _, a in acq.iterrows():
            end = pd.Timestamp(a["acq_utc"]).tz_convert("UTC").tz_localize(None)
            start = end - pd.Timedelta(days=days)
            img = (
                ee.ImageCollection(GPM_COLLECTION)
                .filterDate(start.isoformat(), end.isoformat())
                .select("precipitation")
                .sum()
                .multiply(0.5)
            )
            res = img.reduceRegions(fc, ee.Reducer.first(), scale=GPM_SCALE_M).getInfo()
            for f in res["features"]:
                rows.append(
                    {
                        "field_id": f["properties"]["field_id"],
                        "date": a["date"],
                        "orbit": a["orbit"],
                        "rain_3d_mm": f["properties"].get("first", np.nan),
                    }
                )
        return pd.DataFrame(rows)
    except Exception as e:  # noqa: BLE001 (any EE failure -> no rain feature, flagged)
        print(f"GPM rain unavailable ({type(e).__name__}); rain_3d_mm left NaN")
        return None


# --- assembly -----------------------------------------------------------------------------


def assemble(
    gcov_frames: list[pd.DataFrame],
    coh: pd.DataFrame,
    enl: pd.DataFrame,
    acq: pd.DataFrame,
    fire_table: pd.DataFrame,
    rain: pd.DataFrame | None,
    baseline_months=BASELINE_MONTHS,
) -> pd.DataFrame:
    """Join the per-source tables into FEATURE_TABLE_COLUMNS (one row per field/date/orbit).

    gcov_frames: multisignal.extract_gcov_series outputs (HVHV_db, HHHH_db, ratio_db, ...).
    coh: multisignal.extract_coherence_series output (ascending pairs).
    enl: enl_table outputs; acq: acquisition_times outputs; fire_table: fires.fires_per_field.
    baseline_months: months of the HV and coherence baselines (site zafra.baseline_months).
    """
    g = pd.concat(gcov_frames, ignore_index=True)
    g = g.rename(
        columns={
            "HVHV_db": "hv_db",
            "HHHH_db": "hh_db",
            "ratio_db": "cross_ratio_db",
            "mean_looks": "n_looks",
        }
    )
    g = g.merge(enl[[*KEYS, "enl"]], on=KEYS, how="left")
    g = g.merge(acq, on=["date", "orbit"], how="left")

    parts = []
    for _, d in g.groupby(["field_id", "orbit"], sort=True):
        d = d.sort_values("date").reset_index(drop=True)
        d = pd.concat([d, hv_dynamics(d["date"], d["hv_db"], baseline_months)], axis=1)
        parts.append(d)
    g = pd.concat(parts, ignore_index=True)

    c = coh.assign(orbit="asc", date=coh["sec_date"]).rename(
        columns={"coh_mean": "coherence_mean", "coh_min": "coherence_min"}
    )
    base = {
        fid: season_baseline(
            d.sort_values("sec_date")["sec_date"],
            d.sort_values("sec_date")["coherence_mean"],
            months=baseline_months,
        )
        for fid, d in c.groupby("field_id")
    }
    c["delta_coh_vs_baseline"] = c["coherence_mean"] - c["field_id"].map(base)
    g = g.merge(
        c[[*KEYS, "coherence_mean", "coherence_min", "delta_coh_vs_baseline"]],
        on=KEYS,
        how="left",
    )

    near = fire_table[fire_table["dist_edge_m"] <= FIRE_BUFFER_M]
    g["fire_7d"], g["fire_frp_max_7d"] = 0, 0.0
    for fid, idx in g.groupby("field_id").groups.items():
        f = near[near["field_id"] == fid]
        n, mx = fire_window_stats(f["acq_utc"], f["frp"], g.loc[idx, "acq_utc"])
        g.loc[idx, "fire_7d"], g.loc[idx, "fire_frp_max_7d"] = n, mx

    if rain is not None:
        g = g.merge(rain, on=KEYS, how="left")
    else:
        g["rain_3d_mm"] = np.nan

    return g.reindex(columns=FEATURE_TABLE_COLUMNS).sort_values(
        ["field_id", "acq_utc"], ignore_index=True
    )
