"""Tests for the exercise-9 follow-up: scene-quality check (zafra.scene_quality) and excluded
acquisitions (zafra.site, zafra.zonal.drop_excluded). Synthetic tables only; no NISAR files."""

import numpy as np
import pandas as pd
import pytest

from zafra import scene_quality as SQ
from zafra import zonal
from zafra.site import load_site


def gcov_table(p95_shift=None, enl=None, n_dates=7, n_fields=200, seed=0):
    """Per-field HV dB and ENL on n_dates ascending dates; optional per-date changes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-06-24", periods=n_dates, freq="12D")
    rows = []
    for i, d in enumerate(dates):
        hv = rng.normal(-15, 2, n_fields) + (p95_shift or {}).get(i, 0.0)
        e = np.full(n_fields, (enl or {}).get(i, 3.4))
        rows.append(pd.DataFrame({"field_id": range(n_fields), "date": d, "orbit": "asc",
                                  "HVHV_db": hv, "enl": e}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_clean_series_not_flagged():
    qc = SQ.scene_quality(gcov_table())
    assert len(qc) == 7 and not qc["flagged"].any()


def test_flags_hv_p95_shift_and_enl_jump():
    qc = SQ.scene_quality(gcov_table(p95_shift={3: -2.6}, enl={5: 4.9}))
    flagged = qc.set_index(qc["date"].dt.strftime("%m-%d"))["flagged"]
    assert flagged.sum() == 2
    r3, r5 = qc.iloc[3], qc.iloc[5]
    assert (
        r3["flagged"]
        and "HV p95" in r3["reason"]
        and r3["d_hv_p95_db"] == pytest.approx(-2.6, abs=0.4)
    )
    assert (
        r5["flagged"] and "ENL" in r5["reason"] and r5["d_enl_rel"] == pytest.approx(4.9 / 3.4 - 1)
    )


def test_thresholds_are_exclusive_of_small_changes():
    qc = SQ.scene_quality(gcov_table(p95_shift={3: -1.5}, enl={5: 3.4 * 1.25}))
    assert not qc["flagged"].any()


def test_orbits_are_checked_separately():
    a = gcov_table()
    d = gcov_table(enl={i: 2.0 for i in range(7)}).assign(orbit="desc")
    qc = SQ.scene_quality(pd.concat([a, d]))  # desc ENL far from asc, but steady in itself
    assert not qc["flagged"].any() and set(qc["orbit"]) == {"asc", "desc"}


def test_brazil_config_excludes_0718_for_gcov_and_gunw():
    site = load_site("brazil", register=False)
    assert site.excluded_dates("gcov_asc") == {"2026-07-18"}
    assert site.excluded_dates("gunw_asc") == {"2026-07-18"}
    assert site.excluded_dates("gcov_desc") == set()
    assert all(e.reason for e in site.nisar_exclude)
    assert load_site("morelos", register=False).nisar_exclude == ()


def test_drop_excluded():
    site = load_site("brazil", register=False)
    gc = pd.DataFrame({
        "field_id": [1, 1, 1, 1],
        "orbit": ["asc", "asc", "desc", "asc"],
        "date": pd.to_datetime(["2026-07-06", "2026-07-18", "2026-07-18", "2026-08-11"]),
    })  # fmt: skip
    coh = pd.DataFrame({
        "field_id": [1, 1, 1],
        "ref_date": pd.to_datetime(["2026-07-06", "2026-07-18", "2026-08-11"]),
        "sec_date": pd.to_datetime(["2026-07-18", "2026-08-11", "2026-08-23"]),
    })  # fmt: skip
    gc2, coh2, dropped = zonal.drop_excluded(gc, coh, site)
    assert dropped == {"gcov_rows": 1, "coherence_rows": 2}
    assert list(gc2["orbit"]) == ["asc", "desc", "asc"]  # descending 07-18 is kept
    assert list(coh2["ref_date"].dt.strftime("%m-%d")) == ["08-11"]
