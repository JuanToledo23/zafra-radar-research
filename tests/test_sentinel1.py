"""Tests for the exercise-9 follow-up: Sentinel-1 series cleaning (zafra.sentinel1.clean).
Synthetic table only; no Earth Engine."""

import numpy as np
import pandas as pd
import pytest


def test_s1_clean_keeps_best_slice_and_drops_partial_cover():
    from zafra import sentinel1 as S1

    t = pd.Timestamp("2026-01-10").value // 10**6
    df = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "time_ms": [t, t + 20_000, t + 6 * 86_400_000, t + 12 * 86_400_000],
        "rel_orbit": [143.0, 143.0, 143.0, 143.0],
        "platform": ["A", "A", "C", "A"],
        "vh_linear": [0.02, 0.03, 0.02, 0.02],
        "vv_linear": [0.1, 0.1, 0.1, 0.1],
        "valid_px": [120, 200, 200, 150],
    })  # fmt: skip
    out = S1.clean(df)
    assert list(out["image_id"]) == ["b", "c"]  # b beats a on the same date; d < 90 % cover
    assert out["vh_db"].iloc[0] == pytest.approx(10 * np.log10(0.03))
