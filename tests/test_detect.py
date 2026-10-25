"""Synthetic checks for the exercise-4 detectors.

Field means are set exactly (no noise) so each test pins down the decision logic; per-pixel
values for the ENL estimate are drawn from a gamma distribution with a known number of looks.
"""

import numpy as np
import pandas as pd
import pytest

from zafra.detect import detect_ratio, detect_rule
from zafra.detect import ratio_test as RT

N_PX = 200
LOOKS = 4.0
DATES = pd.date_range("2025-11-23", periods=15, freq="12D")


def make_series(db_values, field_id="f1") -> pd.DataFrame:
    db = np.asarray(db_values, dtype="float64")
    return pd.DataFrame(
        {
            "field_id": field_id,
            "date": DATES[: len(db)],
            "pol": "HVHV",
            "valid_px": N_PX,
            "mean_linear": 10 ** (db / 10),
            "mean_db": db,
            "mean_looks": 4.6,
        }
    )


def gamma_provider(series: pd.DataFrame, looks: float = LOOKS, seed: int = 0):
    """Per-pixel gamma(looks) intensities whose mean is the series' field mean."""
    rng = np.random.default_rng(seed)
    means = series.set_index(["field_id", "date"])["mean_linear"]

    def provider(field_id, date):
        mu = means[(field_id, date)]
        return rng.gamma(shape=looks, scale=mu / looks, size=N_PX)

    return provider


def run(db_values):
    s = make_series(db_values)
    det, scenes = detect_ratio(s, gamma_provider(s), return_scenes=True)
    return det.iloc[0], scenes, detect_rule(s).iloc[0]


def test_step_detected_at_right_scene():
    db = [-12.0] * 7 + [-17.0] * 8  # cut between scene 6 and scene 7
    det, _, rule = run(db)
    assert det["detected"]
    assert det["date_detected"] == DATES[7]
    assert det["date_before"] == DATES[6]
    assert det["p_prev"] < RT.ALPHA_PREV and det["p_base"] < RT.ALPHA_BASE
    assert det["drop_db"] == pytest.approx(-5.0)
    assert det["confidence"] == "high"
    assert rule["detected"] and rule["date_detected"] == DATES[7]


def test_one_scene_dip_not_detected():
    db = [-12.0] * 15
    db[7] = -17.0  # one bad scene, then back to normal
    det, scenes, rule = run(db)
    assert not det["detected"]
    assert not rule["detected"]
    # The dip itself is highly "significant" vs speckle; only the sustained criterion stops it.
    assert scenes.loc[7, "p_prev"] < 1e-6
    assert not scenes.loc[7, "sustained"]


def test_gradual_decline_detected_via_baseline():
    # 0.3 dB per scene after the baseline: no single step is significant at 0.01, but the
    # accumulated decline is.
    db = [-12.0] * 3 + [-12.0 - 0.3 * k for k in range(1, 13)]
    det, _, _ = run(db)
    assert det["detected"]
    assert det["trigger"] == "baseline"
    assert det["p_prev"] >= RT.ALPHA_PREV
    assert det["p_base"] < RT.ALPHA_BASE
    # First scene >= 2 dB below baseline is k = 7 (-2.1 dB) -> index 3 + 6 = 9.
    assert det["date_detected"] == DATES[9]


def test_no_change_nothing_detected():
    det, scenes, rule = run([-12.0] * 15)
    assert not det["detected"]
    assert not rule["detected"]
    assert not scenes["candidate"].any()


def test_drop_after_wet_soil_rise_not_counted_on_step_test():
    # Like field_06's 04-04 -> 04-16: a wet-soil rise followed by the fall back.
    db = [-14.0] * 3 + [-16.5] * 12
    db[8] = -14.6  # rise of 1.9 dB vs previous scene -> flagged wet
    _, scenes, _ = run(db)
    assert scenes.loc[8, "wet_flag"]
    assert scenes.loc[9, "after_wet"] and not scenes.loc[9, "sig_prev"]


def test_enl_estimate_recovers_known_looks():
    rng = np.random.default_rng(1)
    px = [rng.gamma(LOOKS, 0.05 / LOOKS, size=20000) for _ in range(3)]
    enl, per_scene = RT.estimate_enl(px)
    assert enl == pytest.approx(LOOKS, rel=0.05)
    assert len(per_scene) == 3


def test_ratio_pvalue_is_calibrated_under_no_change():
    # Under H0 the one-sided p-value should be uniform: ~1% of null ratios below 0.01.
    rng = np.random.default_rng(2)
    k = N_PX * LOOKS
    a = rng.gamma(k, 1 / k, size=20000)
    b = rng.gamma(k, 1 / k, size=20000)
    p = np.array([RT.p_decrease(x / y, k, k) for x, y in zip(a, b, strict=True)])
    assert np.mean(p < 0.01) == pytest.approx(0.01, abs=0.003)
