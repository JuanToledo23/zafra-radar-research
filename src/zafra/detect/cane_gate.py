"""Cane / non-cane gate (exercise 4c): P(field is sugarcane during a zafra).

The harvest detector of exercise 4b fires on almost every non-cane field (the dry-season HV
decline looks like a slow harvest). This gate runs first, so the detector only sees cane.

Circularity
-----------
The cane labels of fields_v2 come from Sentinel-2 NDVI phenology (script 07) plus the
author's visual tracing. A classifier fed NDVI features is therefore partly re-learning the
labelling rule, and its score is optimistic. So three variants are kept apart:
  A  radar only  (NISAR GCOV + GUNW coherence)  -- the honest test: no NDVI in the inputs
  B  optical only (S2 NDVI phenology)           -- how close a model gets to the label rule
  C  combined (radar + optical + VIIRS fire)    -- what optical adds on top of radar
plus "A+F" (radar + VIIRS fire), which also uses no optical input.

Operational default: B ("optical"). Optical selects the fields, radar dates the harvest.
A is reported alongside every prediction (p_cane_radar) as the independent check.

Model: logistic regression with median imputation + missing indicators, standardization
and class_weight="balanced" (zafra.detect.model.make_logistic). One row per field and
zafra; evaluated leave-one-field-out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut

from zafra.detect.model import logistic_coefficients, make_logistic

RADAR_FEATURES = [
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
OPTICAL_FEATURES = [
    "ndvi_max",
    "ndvi_min",
    "ndvi_days_above_0_6",
    "ndvi_n_collapses",
    "ndvi_longest_high_run_days",
    "collapse_in_zafra",
    "bare_share_after_collapse",
]
FIRE_FEATURES = ["fire_count_in_zafra"]
VARIANTS = {
    "A": RADAR_FEATURES,
    "A+F": RADAR_FEATURES + FIRE_FEATURES,
    "B": OPTICAL_FEATURES,
    "C": RADAR_FEATURES + OPTICAL_FEATURES + FIRE_FEATURES,
}
VARIANT_NAMES = {
    "A": "A: radar only (p_cane_radar)",
    "A+F": "A+F: radar + VIIRS fire",
    "B": "B: optical only (default gate)",
    "C": "C: radar + optical + fire (not recommended)",
}
USES_OPTICAL = {"A": False, "A+F": False, "B": True, "C": True}
USES_RADAR = {"A": True, "A+F": True, "B": False, "C": True}
# Readable aliases for the `variant` argument of classify_fields.
VARIANT_ALIASES = {"radar": "A", "radar+fire": "A+F", "optical": "B", "combined": "C"}
# Operational gate: optical (B). It drops no cane field in the leave-one-field-out run,
# while radar-only (A) drops 3, among them p36, the only grower-confirmed field. A stays as
# the independent check and is always returned as p_cane_radar (docs/METHODS.md, 4c).
DEFAULT_VARIANT = "optical"
RADAR_VARIANT = "A"
TRAIN_ZAFRA = "2025-26"  # the only zafra with NISAR data
THRESHOLD = 0.5
HIGH_CONF = (0.2, 0.8)  # p < 0.2 or p > 0.8 -> "high", otherwise "uncertain"

# Transparent rules for comparison
NDVI_RULE_RUN_DAYS = 150  # script 07 test (a)
RADAR_RULE_DROP_DB = -2.0


def design_matrix(df: pd.DataFrame, features) -> pd.DataFrame:
    """Model inputs. collapse_in_zafra: 1 if the main NDVI collapse is in Nov-May, 0 if in
    Jun-Oct, NaN if none (a month number is not a usable linear feature)."""
    X = df.copy()
    m = X["ndvi_collapse_month"] if "ndvi_collapse_month" in X else pd.Series(np.nan, X.index)
    X["collapse_in_zafra"] = np.where(m.isna(), np.nan, m.isin([11, 12, 1, 2, 3, 4, 5]))
    return X[list(features)].astype("float64")


# --- rules --------------------------------------------------------------------------------


def rule_ndvi(df: pd.DataFrame) -> pd.Series:
    """Script-07-style rule: one long canopy plateau and exactly one collapse."""
    return (
        (df["ndvi_longest_high_run_days"] >= NDVI_RULE_RUN_DAYS) & (df["ndvi_n_collapses"] == 1)
    ).astype(int)


def rule_radar(df: pd.DataFrame) -> pd.Series:
    """One HV step of at least 2 dB that coincides with the coherence minimum."""
    return (
        (df["hv_max_single_drop_db"] <= RADAR_RULE_DROP_DB) & (df["joint_collapse"] == 1)
    ).astype(int)


# --- model --------------------------------------------------------------------------------


def fit(df: pd.DataFrame, y, features):
    return make_logistic().fit(design_matrix(df, features), np.asarray(y, dtype=int))


def predict(model, df: pd.DataFrame, features) -> np.ndarray:
    return model.predict_proba(design_matrix(df, features))[:, 1]


def loo_probabilities(df: pd.DataFrame, y, features) -> np.ndarray:
    """Leave-one-out P(cane): one row per field, so this is leave-one-field-out."""
    y = np.asarray(y, dtype=int)
    p = np.full(len(df), np.nan)
    for tr, te in LeaveOneOut().split(df):
        p[te] = predict(fit(df.iloc[tr], y[tr], features), df.iloc[te], features)
    return p


def coefficients(model, features) -> pd.Series:
    return logistic_coefficients(model, list(features))


def evaluate(y, pred) -> dict:
    """Accuracy, precision / recall of the cane class and the confusion matrix counts."""
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "accuracy": (tp + tn) / len(y),
        "precision_cane": tp / (tp + fp) if tp + fp else np.nan,
        "recall_cane": tp / (tp + fn) if tp + fn else np.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def confidence(p) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    lo, hi = HIGH_CONF
    return np.where(np.isnan(p), "no_data", np.where((p < lo) | (p > hi), "high", "uncertain"))


# --- public API ---------------------------------------------------------------------------


def load_training(features_path=None, fields_path=None, zafra: str = TRAIN_ZAFRA):
    """(features of the labelled fields for `zafra`, y) from the fields_v2 tables."""
    import geopandas as gpd

    from zafra.paths import FEATURES_DIR, FIELDS_DIR

    feats = pd.read_parquet(features_path or FEATURES_DIR / "fields_v2_cane_features.parquet")
    labels = gpd.read_file(fields_path or FIELDS_DIR / "fields_v2.geojson")[["id", "label"]]
    df = feats[feats["zafra"] == zafra].merge(labels, left_on="field_id", right_on="id")
    return df.reset_index(drop=True), (df["label"] == "cana").astype(int).to_numpy()


def resolve_variant(variant: str) -> str:
    """'optical' / 'radar' / 'radar+fire' / 'combined' or a key of VARIANTS -> that key."""
    v = VARIANT_ALIASES.get(variant, variant)
    if v not in VARIANTS:
        raise ValueError(f"unknown gate variant {variant!r}")
    return v


def _gate_probability(f: pd.DataFrame, variant: str, train) -> np.ndarray:
    cols = VARIANTS[variant]
    p = predict(fit(train[0], train[1], cols), f, cols)
    if USES_RADAR[variant] and "has_radar" in f:
        p = np.where(f["has_radar"].fillna(False).astype(bool), p, np.nan)
    if USES_OPTICAL[variant] and "has_optical" in f:
        p = np.where(f["has_optical"].fillna(False).astype(bool), p, np.nan)
    return p


def classify_fields(gdf, zafra: str, variant: str = DEFAULT_VARIANT, features=None, train=None):
    """P(sugarcane during `zafra`) for every polygon of `gdf` (needs a unique `id`).

    variant:  "optical" (default, B), "radar" (A), "radar+fire" (A+F), "combined" (C), or
              the letter itself.
    features: a cane_features table holding these ids for this zafra; computed from NISAR /
              Sentinel-2 / FIRMS on disk if None (Earth Engine only if the variant uses
              optical).
    train:    (training features, y); default: the 38 labelled fields_v2 polygons, 2025-26.
    Returns a copy of gdf with p_cane (chosen variant), cane_pred (p >= 0.5), confidence
    ("high" if p < 0.2 or p > 0.8, "uncertain" otherwise, "no_data" if the zafra has no
    data for this variant), gate_variant, and p_cane_radar: P from the radar-only variant A,
    always, so a map can show the independent radar evidence per field.
    """
    zafra = zafra.replace("/", "-")
    variant = resolve_variant(variant)
    if features is None:
        features = compute_features(gdf, zafra, optical=USES_OPTICAL[variant])
    f = features[features["zafra"] == zafra].set_index("field_id").reindex(gdf["id"])
    f = f.reset_index().rename(columns={"index": "field_id"})
    train = train if train is not None else load_training()
    p = _gate_probability(f, variant, train)
    out = gdf.copy()
    out["p_cane"] = p
    out["cane_pred"] = pd.array(np.where(np.isnan(p), pd.NA, p >= THRESHOLD), dtype="boolean")
    out["confidence"] = confidence(p)
    out["gate_variant"] = variant
    out["p_cane_radar"] = (
        p if variant == RADAR_VARIANT else _gate_probability(f, RADAR_VARIANT, train)
    )
    return out


def compute_features(gdf, zafra: str, optical: bool = True, cache_dir=None) -> pd.DataFrame:
    """cane_features table for arbitrary polygons (radar from NISAR on disk; NDVI and bare
    share from Earth Engine if `optical`; fires from the raw FIRMS CSVs)."""
    import tempfile
    from pathlib import Path

    from zafra import cane_features as CF
    from zafra.paths import NISAR_DIR, OUTPUTS_DIR

    cache_dir = Path(cache_dir or OUTPUTS_DIR / "cane_gate" / "adhoc")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fields.geojson"
        gdf.to_file(path, driver="GeoJSON")
        acq, coh = CF.load_radar_inputs(
            path,
            cache_dir,
            NISAR_DIR,
            NISAR_DIR / "GCOV_DESCENDING_t113f79",
            NISAR_DIR / "GUNW_ASCENDING_t19f11",
        )
    ndvi, bare = None, None
    if optical:
        a, b = CF.crop_year_bounds(zafra)
        ndvi = CF.load_ndvi(gdf, cache_dir / "ndvi_series.csv", f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}")
        t = CF.build_table(gdf["id"], [zafra], None, None, ndvi, None)
        req = t.dropna(subset=["collapse_low_image_id"])
        bare = CF.load_bare_shares(
            gdf,
            list(zip(req["field_id"], req["collapse_low_image_id"], strict=True)),
            cache_dir / "bare_share.csv",
        )
    return CF.build_table(gdf["id"], [zafra], acq, coh, ndvi, CF.fire_table_for(gdf), bare)
