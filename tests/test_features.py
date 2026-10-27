"""Synthetic checks for the exercise-4b features, labels and date rule."""

import numpy as np
import pandas as pd
import pytest

from zafra import features as FT
from zafra.detect import model as M

DATES = pd.date_range("2025-11-23", periods=12, freq="12D")


def test_baseline_is_median_of_first_three_nov_dec():
    # First three Nov-Dec values are -12, -14, -13 -> median -13; later values are ignored.
    hv = [-12.0, -14.0, -13.0, -20.0, -25.0] + [-30.0] * 7
    assert FT.season_baseline(DATES, hv) == pytest.approx(-13.0)
    d = FT.hv_dynamics(DATES, hv)
    assert d["delta_hv_vs_baseline"].iloc[4] == pytest.approx(-12.0)


def test_baseline_skips_nan_and_falls_back_when_no_early_dates():
    hv = [np.nan, -12.0, -14.0, -13.0] + [-20.0] * 8
    assert FT.season_baseline(DATES, hv) == pytest.approx(-13.0)
    late = pd.date_range("2026-02-01", periods=4, freq="12D")
    assert FT.season_baseline(late, [-10.0, -11.0, -12.0, -40.0]) == pytest.approx(-11.0)


def test_sharpness_step_vs_ramp():
    ramp = -12.0 - 0.5 * np.arange(len(DATES))  # steady drying: -0.5 dB per scene
    step = np.r_[[-12.0] * 6, [-17.0] * 6]  # cut between scene 5 and 6
    r = FT.hv_dynamics(DATES, ramp)
    s = FT.hv_dynamics(DATES, step)
    # A ramp is fully predicted by its own trend: sharpness ~0 everywhere it is defined.
    assert np.allclose(r["sharpness"].dropna(), 0.0, atol=1e-9)
    assert r["slope_hv_3"].iloc[5] == pytest.approx(-0.5 / 12)
    # The step is 5 dB below a flat trend at the cut, and flat (0) elsewhere.
    assert s["sharpness"].iloc[6] == pytest.approx(-5.0)
    assert np.allclose(s["sharpness"].drop(index=[6, 7, 8]).dropna(), 0.0)
    # The trend uses only *previous* scenes, so the first SLOPE_N rows are NaN.
    assert s["slope_hv_3"].iloc[: FT.SLOPE_N].isna().all()
    assert s["gap_days"].iloc[1] == pytest.approx(12.0)


def test_label_states_around_window():
    feats = pd.DataFrame(
        {
            "field_id": ["a"] * 5 + ["b"],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-09", "2026-01-10", "2026-01-15", "2026-01-16", "2026-01-01"]
            ),
        }
    )
    windows = pd.DataFrame(
        {
            "field_id": ["a", "b"],
            "harvest_window_start": pd.to_datetime(["2026-01-10", None]),
            "harvest_window_end": pd.to_datetime(["2026-01-15", None]),
        }
    )
    lab = M.label_states(feats, windows)
    # Strictly before -> 0, strictly after -> 1, the two boundary dates are dropped,
    # and field b (no window) is not labelled at all.
    assert lab["date"].dt.strftime("%m-%d").tolist() == ["01-01", "01-09", "01-16"]
    assert lab["state"].tolist() == [0, 0, 1]
    kept = M.label_states(feats, windows, keep_window=True)
    assert len(kept) == 5 and kept["state"].isna().sum() == 2


def test_sustained_date_rule():
    d = pd.date_range("2026-01-01", periods=8, freq="6D")
    # Isolated 1 at index 2 is ignored; the first sustained 1 is at index 5.
    r = M.sustained_date(d, [0, 0, 1, 0, 0, 1, 1, 1])
    assert r["detected"]
    assert r["date_before"] == d[4] and r["date_after"] == d[5]
    assert r["date"] == d[4] + pd.Timedelta(days=3)
    # A final lone 1 cannot be confirmed.
    assert not M.sustained_date(d, [0] * 7 + [1])["detected"]
    # Input order does not matter (asc and desc rows arrive interleaved).
    perm = [3, 0, 7, 5, 1, 6, 2, 4]
    states = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    assert M.sustained_date(d[perm], states[perm])["date"] == r["date"]


def test_fire_window_counts_previous_seven_days_only():
    acq = pd.to_datetime(["2026-02-10 12:05"]).tz_localize("UTC")
    fires = pd.to_datetime(
        ["2026-02-03 11:00", "2026-02-03 13:00", "2026-02-10 12:00", "2026-02-10 13:00"]
    ).tz_localize("UTC")
    n, frp = FT.fire_window_stats(fires, [1.0, 2.0, 5.0, 9.0], acq)
    # 02-03 11:00 is > 7 days before; 02-10 13:00 is after the acquisition.
    assert n.tolist() == [2] and frp.tolist() == [5.0]
