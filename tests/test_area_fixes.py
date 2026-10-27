"""Tests for exercise 4e: the baseline requirement of the detector and fragment merging."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from zafra import segments as S
from zafra.detect import model as M

X0, Y0 = 480_000.0, 2_060_000.0


def series(spec):
    """'d0 a0 d1 ...' -> (times, orbits, states), 6 days apart, starting Nov 18."""
    toks = spec.split()
    t = pd.date_range("2025-11-18", periods=len(toks), freq="6D")
    orbits = ["asc" if s[0] == "a" else "desc" for s in toks]
    return t, orbits, [int(s[1]) for s in toks]


# --- baseline requirement ------------------------------------------------------------------


def test_three_standing_looks_in_own_orbit_gives_a_date():
    t, o, s = series("d0 a0 d0 a0 d0 a0 d1 a1 d1")
    r = M.baseline_date(t, o, s)
    assert r["detected"] and r["status"] == M.STATUS_HARVESTED
    assert r["date_after"] == t[6]
    assert r["date"] == t[5] + (t[6] - t[5]) / 2  # same midpoint as sustained_date
    assert r["date"] == M.sustained_date(t, s)["date"]


def test_series_starting_harvested_is_before_coverage():
    t, o, s = series("d1 a1 d1 a1 d1")
    r = M.baseline_date(t, o, s)
    assert not r["detected"] and pd.isna(r["date"])
    assert r["status"] == M.STATUS_BEFORE_COVERAGE


def test_early_candidate_has_no_baseline():
    # asc candidate at the 3rd asc look: only 2 standing asc looks before it
    t, o, s = series("d0 a0 d0 a0 d0 a1 d1 a1")
    r = M.baseline_date(t, o, s)
    assert not r["detected"] and r["status"] == M.STATUS_NO_BASELINE


def test_other_orbit_does_not_count_unless_asked():
    # 3 standing desc looks, but the candidate is asc with 2 asc looks before it
    t, o, s = series("d0 a0 d0 a0 d0 a1 d1")
    assert M.baseline_date(t, o, s)["status"] == M.STATUS_NO_BASELINE
    assert M.baseline_date(t, o, s, own_orbit=False)["status"] == M.STATUS_HARVESTED


def test_unconfirmed_flicker_does_not_void_baseline():
    # one unconfirmed asc 1 early on; still 3+ standing asc looks before the asc candidate
    t, o, s = series("d0 a1 d0 a0 d0 a0 d0 a0 d0 a1 d1")
    r = M.baseline_date(t, o, s)
    assert r["status"] == M.STATUS_HARVESTED and r["date_after"] == t[9]


def test_no_sustained_detection():
    t, o, s = series("d0 a0 d1 a0 d0 a1 d0")
    r = M.baseline_date(t, o, s)
    assert r["status"] == M.STATUS_NOT_DETECTED and not r["detected"]


def test_dates_with_baseline_matches_old_dates_when_baseline_exists():
    t, o, s = series("d0 a0 d0 a0 d0 a0 d0 a1 d1 a1")
    df = pd.DataFrame(
        {
            "field_id": "f",
            "date": t.normalize(),
            "acq_utc": t.tz_localize("UTC"),
            "orbit": o,
            "p": np.array(s, float),
        }
    )
    new = M.dates_with_baseline(df, "p").iloc[0]
    old = M.dates_from_probabilities(df, "p").iloc[0]
    assert new["status"] == M.STATUS_HARVESTED and new["date"] == old["date"]


# --- merging -------------------------------------------------------------------------------


def strip(i, w=100):
    """Squares side by side along x (shared 100 m edges)."""
    return box(X0 + i * w, Y0, X0 + (i + 1) * w, Y0 + 100)


def test_adjacent_pairs_need_a_shared_edge():
    g = gpd.GeoDataFrame(
        geometry=[strip(0), strip(1), box(X0 + 200, Y0 + 100, X0 + 300, Y0 + 200)],
        crs=S.METRIC_CRS,
    )
    # 0-1 share a 100 m edge; 1-2 touch only at a corner
    assert S.adjacent_pairs(g) == [(0, 1)]


def test_merge_by_date_and_status():
    dates = ["2026-01-10", "2026-01-14", "2026-02-20", None, None, None]
    status = ["harvested", "harvested", "harvested", "no_baseline", "no_baseline", "not_detected"]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    grp = S.merge_groups(dates, status, pairs)
    assert grp[0] == grp[1]  # 4 days apart
    assert grp[2] != grp[1]  # 37 days apart
    assert grp[3] == grp[4]  # both undated, same status
    assert grp[4] != grp[5]  # undated, different status
    assert grp[2] != grp[3]  # dated next to undated


def test_merge_chain_cannot_drift_beyond_max_days():
    dates = ["2026-01-01", "2026-01-05", "2026-01-09"]  # 4 + 4 days, 8 end to end
    grp = S.merge_groups(dates, ["harvested"] * 3, [(0, 1), (1, 2)])
    assert len(set(grp)) == 2


def test_dissolve_area_weighted_date_and_columns():
    g = gpd.GeoDataFrame(
        {
            "seg_id": ["a", "b"],
            "harvest_date": ["2026-01-01", "2026-01-05"],
            "p_cane": [0.9, 0.5],
            "status": ["harvested", "harvested"],
        },
        geometry=[strip(0, 300), box(X0 + 300, Y0, X0 + 400, Y0 + 100)],  # 3 ha and 1 ha
        crs=S.METRIC_CRS,
    )
    out = S.dissolve_groups(
        g, np.array([0, 0]), "harvest_date", weighted=["p_cane"], first=["status"]
    )
    assert len(out) == 1
    r = out.iloc[0]
    assert r["area_ha"] == pytest.approx(4.0)
    assert r["harvest_date"] == pd.Timestamp("2026-01-02")  # (3*0 + 1*4) / 4 = day 1
    assert r["p_cane"] == pytest.approx(0.8)
    assert r["n_merged"] == 2 and r["member_ids"] == "a,b"
    assert r.geometry.area == pytest.approx(40_000)
