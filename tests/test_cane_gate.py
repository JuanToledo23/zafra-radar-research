"""Synthetic checks for the exercise-4c cane features, rule baselines and gate API."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from zafra import cane_features as CF
from zafra.detect import cane_gate as G

ASC = pd.date_range("2025-11-23", periods=14, freq="12D")
DESC = ASC - pd.Timedelta(days=5)
CUT = 6  # first post-cut acquisition index


def radar_rows(hv_asc, hv_desc, fid="f"):
    rows = []
    for orbit, dates, hv in (("asc", ASC, hv_asc), ("desc", DESC, hv_desc)):
        for d, v in zip(dates, hv, strict=True):
            rows.append(
                {
                    "field_id": fid,
                    "date": d,
                    "orbit": orbit,
                    "hv_db": v,
                    "hh_db": v + 7,
                    "cross_ratio_db": -7.0,
                    "enl": 3.0,
                }
            )
    return pd.DataFrame(rows)


def coh_rows(values, fid="f"):
    return pd.DataFrame({"field_id": fid, "sec_date": ASC, "coh_mean": values})


def cane_like():
    hv = np.r_[[-13.0] * CUT, [-18.0] * (len(ASC) - CUT)]
    coh = np.full(len(ASC), 0.45)
    coh[CUT] = 0.18  # the pair spanning the cut decorrelates
    return radar_rows(hv, hv - 0.5), coh_rows(coh)


def flat():
    hv = -18.0 - 0.1 * np.arange(len(ASC))  # slow drift, no step
    return radar_rows(hv, hv), coh_rows(np.full(len(ASC), 0.55))


def test_radar_summary_cane_vs_flat():
    a, c = cane_like()
    cane = CF.radar_summary(a, c)
    a, c = flat()
    nc = CF.radar_summary(a, c)
    assert cane["has_radar"] and nc["has_radar"]
    assert cane["hv_max_single_drop_db"] == pytest.approx(-5.0)
    assert cane["hv_drop_date"] in (ASC[CUT], DESC[CUT])
    assert cane["hv_range_db"] == pytest.approx(5.0)
    assert cane["coh_min"] == pytest.approx(0.18)
    assert cane["coh_range"] == pytest.approx(0.27)
    assert cane["joint_collapse"] == 1
    assert cane["hv_pre_level_db"] == pytest.approx(-13.25)  # mean of asc -13 and desc -13.5
    assert nc["hv_max_single_drop_db"] == pytest.approx(-0.1)
    assert nc["coh_range"] == pytest.approx(0.0)
    assert nc["hv_range_db"] < 1.5


def test_joint_collapse_zero_when_drop_and_coherence_minimum_are_apart():
    a, _ = cane_like()
    coh = np.full(len(ASC), 0.45)
    coh[1] = 0.2  # minimum two months before the HV step
    assert CF.radar_summary(a, coh_rows(coh))["joint_collapse"] == 0


def test_no_radar_rows_gives_nan():
    a, c = flat()
    out = CF.radar_summary(a.iloc[0:0], c.iloc[0:0])
    assert not out["has_radar"] and np.isnan(out["hv_max_single_drop_db"])


def test_optical_summary_cane_vs_annual_crop():
    dates = pd.date_range("2025-06-05", "2026-05-25", freq="10D")
    cut = pd.Timestamp("2026-02-10")
    cane = np.where(dates < cut, 0.8, 0.2)
    clear = pd.DataFrame(
        {"date": dates, "ndvi_mean": cane, "image_id": [f"i{k}" for k in range(len(dates))]}
    )
    o = CF.optical_summary(clear, "2025-26")
    assert o["ndvi_n_collapses"] == 1
    assert o["ndvi_collapse_month"] == 2
    assert o["ndvi_longest_high_run_days"] >= 150
    assert o["collapse_low_image_id"] == clear.loc[dates >= cut, "image_id"].iloc[0]

    # Maize-like: green Jul-Sep only, harvested in October (rainy season), then bare.
    crop = np.where((dates >= "2025-07-01") & (dates < "2025-10-01"), 0.8, 0.2)
    o = CF.optical_summary(clear.assign(ndvi_mean=crop), "2025-26")
    assert o["ndvi_longest_high_run_days"] < 150
    assert o["ndvi_collapse_month"] == 10


def test_rule_baselines():
    df = pd.DataFrame(
        {
            "ndvi_longest_high_run_days": [200, 200, 100],
            "ndvi_n_collapses": [1, 2, 1],
            "hv_max_single_drop_db": [-3.0, -1.0, -3.0],
            "joint_collapse": [1, 1, 0],
        }
    )
    assert G.rule_ndvi(df).tolist() == [1, 0, 0]
    assert G.rule_radar(df).tolist() == [1, 0, 0]


def synthetic_training(n=10, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    cols = G.RADAR_FEATURES + G.OPTICAL_FEATURES + G.FIRE_FEATURES
    df = pd.DataFrame({c: rng.normal(size=2 * n) for c in cols})
    df["hv_max_single_drop_db"] = np.where(y == 1, -4.0, -1.0) + rng.normal(0, 0.3, 2 * n)
    df["coh_min"] = np.where(y == 1, 0.2, 0.5) + rng.normal(0, 0.03, 2 * n)
    df["joint_collapse"] = y
    df["ndvi_longest_high_run_days"] = np.where(y == 1, 220.0, 60.0) + rng.normal(0, 10, 2 * n)
    df["ndvi_collapse_month"] = np.where(y == 1, 2, 9)
    df = df.drop(columns="collapse_in_zafra")
    df["field_id"] = [f"t{k}" for k in range(2 * n)]
    return df, y


def gate_inputs():
    train = synthetic_training()
    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )
    feats = pd.concat([train[0].iloc[[0, 15]], train[0].iloc[[1]]], ignore_index=True)
    feats["field_id"] = ["a", "b", "c"]
    feats["zafra"] = "2025-26"
    feats["has_radar"] = [True, True, False]
    feats["has_optical"] = True
    return gdf, feats, train


def test_classify_fields_schema_default_is_optical():
    gdf, feats, train = gate_inputs()
    out = G.classify_fields(gdf, "2025/26", features=feats, train=train)
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 3 and out.crs == gdf.crs
    for col in ("p_cane", "cane_pred", "confidence", "gate_variant", "p_cane_radar"):
        assert col in out
    assert (out["gate_variant"] == "B").all()
    assert out.loc[0, "p_cane"] > 0.5 and bool(out.loc[0, "cane_pred"])
    assert out.loc[1, "p_cane"] < 0.5 and not bool(out.loc[1, "cane_pred"])
    # Field c has optical data but no radar: the gate answers, the radar evidence does not.
    assert not np.isnan(out.loc[2, "p_cane"]) and np.isnan(out.loc[2, "p_cane_radar"])
    assert out.loc[0, "p_cane_radar"] > 0.5 > out.loc[1, "p_cane_radar"]
    assert set(out["confidence"]) <= {"high", "uncertain", "no_data"}


def test_classify_fields_radar_variant():
    gdf, feats, train = gate_inputs()
    out = G.classify_fields(gdf, "2025-26", "radar", features=feats, train=train)
    assert (out["gate_variant"] == "A").all()
    assert np.allclose(out["p_cane"], out["p_cane_radar"], equal_nan=True)
    assert np.isnan(out.loc[2, "p_cane"]) and out.loc[2, "confidence"] == "no_data"
    with pytest.raises(ValueError):
        G.classify_fields(gdf, "2025-26", "sonar", features=feats, train=train)


def test_confidence_tiers():
    assert G.confidence([0.1, 0.5, 0.9, np.nan]).tolist() == [
        "high",
        "uncertain",
        "high",
        "no_data",
    ]
