"""Trained harvest detector (exercise 4b): per-acquisition state classifier + date rule.

Task
----
Each acquisition row (field, date, orbit; features from zafra.features) is classified as
state 0 = standing cane or 1 = harvested. Training labels come from the Sentinel-2 NDVI
harvest window of each cane field (exercise 7): rows dated before the window are 0, rows
after it are 1, rows inside it are dropped (the cut happened at an unknown time in there).
Non-cane fields are never used for training; they only count false alarms.

Two models, both small on purpose (a few hundred labelled rows from ~25 fields):
* logistic regression on standardized features, class_weight="balanced"; NaN coherence
  are median-imputed and get a 0/1 missing indicator;
* xgboost, max_depth 3, up to N_TREES trees with early stopping on an inner group split
  (NaN handled natively).

Evaluation is out-of-fold: GroupKFold by field_id, so a field is never in both training and
test. The harvest date is then read from each field's out-of-fold predictions with
sustained_date (the same rule is applied to every method).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Model inputs: the feature table minus identifiers, the absolute hv_db / hh_db (the model
# must be field-relative) and DROPPED_FEATURES. is_asc encodes the orbit.
FEATURES = [
    "is_asc",
    "cross_ratio_db",
    "valid_px",
    "n_looks",
    "enl",
    "delta_hv_vs_baseline",
    "delta_hv_vs_prev",
    "coherence_mean",
    "coherence_min",
    "delta_coh_vs_baseline",
    "fire_7d",
    "fire_frp_max_7d",
]
# Kept in the feature table, left out of the models after the first exercise-4b run (ablation
# in docs/VALIDATION.md): rain and gap_days mark the late season in this data (every storm and
# the only 24-day gap come after every harvest), and sharpness / slope did not improve the
# dates (xgboost got better without them).
DROPPED_FEATURES = ["rain_3d_mm", "gap_days", "sharpness", "slope_hv_3"]
N_SPLITS = 5
N_TREES = 300
EARLY_STOP = 30
THRESHOLD = 0.5
SEED = 0


# --- labels -------------------------------------------------------------------------------


def label_states(features: pd.DataFrame, windows: pd.DataFrame, keep_window: bool = False):
    """Add `state` (0 before the window, 1 after) to the rows of fields that have a window.

    windows: field_id, harvest_window_start, harvest_window_end. Rows of fields without a
    window are dropped. Rows inside the window (start <= date <= end) are dropped too, or,
    with keep_window=True, kept with state NaN (needed to date a field from all its rows).
    """
    w = windows.dropna(subset=["harvest_window_start", "harvest_window_end"])
    df = features.merge(w, on="field_id", how="inner")
    start = pd.to_datetime(df["harvest_window_start"])
    end = pd.to_datetime(df["harvest_window_end"])
    state = np.where(df["date"] < start, 0.0, np.where(df["date"] > end, 1.0, np.nan))
    df = df.assign(state=state)
    if not keep_window:
        df = df[df["state"].notna()]
    return df.reset_index(drop=True)


def design_matrix(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    X = df.assign(is_asc=(df["orbit"] == "asc").astype(int))
    return X[list(features)].astype("float64")


# --- models -------------------------------------------------------------------------------


def make_logistic():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )


def make_xgb(n_trees: int = N_TREES, scale_pos_weight: float = 1.0, early_stop: bool = True):
    return XGBClassifier(
        n_estimators=n_trees,
        max_depth=3,
        learning_rate=0.05,
        # No row or column sampling: with 0.8 the dates moved by a few days when only the
        # feature column order changed. With 1.0 the fit is deterministic.
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=2,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP if early_stop else None,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=1,
    )


def _pos_weight(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def fit_xgb_early_stop(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """xgboost with early stopping on ~20 % of the training *fields* (never rows)."""
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, va = next(inner.split(X, y, groups))
    m = make_xgb(scale_pos_weight=_pos_weight(y.iloc[tr]))
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
    return m


def cross_validate(df: pd.DataFrame, features=FEATURES) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold P(state = 1) for both models.

    `df` = every row of the training fields, with `state` NaN inside the harvest window
    (label_states(..., keep_window=True)). Folds split *fields*; each model is trained on the
    labelled rows of the training fields and predicts every row of the held-out fields.
    Returns (df with p_logistic, p_xgb, fold), info (best xgboost iterations per fold).
    """
    groups = df["field_id"]
    out = df.copy()
    out["p_logistic"] = np.nan
    out["p_xgb"] = np.nan
    out["fold"] = -1
    best_iters = []
    X_all = design_matrix(df, features)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X_all, groups=groups)):
        train = df.iloc[tr]
        train = train[train["state"].notna()]
        X, y = design_matrix(train, features), train["state"].astype(int)
        lr = make_logistic().fit(X, y)
        xg = fit_xgb_early_stop(X, y, train["field_id"])
        best_iters.append(int(xg.best_iteration) + 1)
        idx = out.index[te]
        out.loc[idx, "p_logistic"] = lr.predict_proba(X_all.iloc[te])[:, 1]
        out.loc[idx, "p_xgb"] = xg.predict_proba(X_all.iloc[te])[:, 1]
        out.loc[idx, "fold"] = k
    return out, {"best_iterations": best_iters}


def fit_full(df: pd.DataFrame, n_trees: int, features=FEATURES):
    """Both models on every labelled row (for non-cane predictions and for reporting
    coefficients / importances). xgboost uses `n_trees` (median of the CV best iterations),
    without early stopping."""
    df = df[df["state"].notna()]
    X, y = design_matrix(df, features), df["state"].astype(int)
    lr = make_logistic().fit(X, y)
    xg = make_xgb(n_trees=n_trees, scale_pos_weight=_pos_weight(y), early_stop=False).fit(X, y)
    return lr, xg


def predict_rows(models, df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    lr, xg = models
    X = design_matrix(df, features)
    return df.assign(p_logistic=lr.predict_proba(X)[:, 1], p_xgb=xg.predict_proba(X)[:, 1])


def logistic_coefficients(lr, features=FEATURES) -> pd.Series:
    """Coefficients on the standardized scale, named (missing indicators as '<feat>_missing')."""
    imp = lr.named_steps["simpleimputer"]
    names = list(features)
    if imp.indicator_ is not None:
        names += [f"{features[i]}_missing" for i in imp.indicator_.features_]
    coef = lr.named_steps["logisticregression"].coef_[0]
    return pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False)


def xgb_importances(xg, features=FEATURES) -> pd.Series:
    """Total gain per feature, normalized to sum to 1 (features never split on: 0)."""
    gain = xg.get_booster().get_score(importance_type="total_gain")
    s = pd.Series({f: gain.get(f, 0.0) for f in features})
    return (s / s.sum()).sort_values(ascending=False)


# --- harvest date -------------------------------------------------------------------------


def sustained_date(dates, states) -> dict:
    """Harvest date from a chronological sequence of 0/1 states.

    The first acquisition classified 1 that is followed by at least one more 1 (the next
    acquisition) is the first "harvested" look; the harvest happened between it and the
    acquisition before it (classified 0, or nothing if it is the first one). The estimate is
    the midpoint. Returns date_before, date_after, date (midpoint), detected.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    s = np.asarray(states, dtype=int)
    order = np.argsort(dates.to_numpy(), kind="stable")
    dates, s = dates.iloc[order].reset_index(drop=True), s[order]
    for i in range(len(s) - 1):
        if s[i] == 1 and s[i + 1] == 1:
            after = dates.iloc[i]
            before = dates.iloc[i - 1] if i > 0 else pd.NaT
            mid = after if pd.isna(before) else before + (after - before) / 2
            return {"date_before": before, "date_after": after, "date": mid, "detected": True}
    return {"date_before": pd.NaT, "date_after": pd.NaT, "date": pd.NaT, "detected": False}


def dates_from_probabilities(df: pd.DataFrame, col: str, threshold: float = THRESHOLD):
    """One sustained_date per field from P(state = 1) in `col`, asc and desc merged.

    Rows are ordered by exact acquisition time (acq_utc), so the two orbits interleave; the
    dates reported are the UTC acquisition days (`date`), as for the rule and ratio test.
    """
    rows = []
    for fid, d in df.groupby("field_id", sort=True):
        d = d.sort_values("acq_utc")
        r = sustained_date(d["date"], (d[col] >= threshold).astype(int))
        rows.append({"field_id": fid, **r})
    return pd.DataFrame(rows)
