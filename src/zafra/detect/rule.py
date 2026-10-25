"""Rule baseline harvest detector (exercise 4, method A).

A deliberately simple, threshold-only detector kept as a point of comparison for the
statistical test in ratio_test.py. It knows nothing about speckle or how many pixels a field
has: a 3 dB drop counts the same on an 84-pixel field as on a 700-pixel one.

Rule
----
* Baseline = median HV (dB) of the first BASELINE_SCENES scenes. The median (not the mean)
  so that one odd early scene (e.g. wet soil) does not drag the baseline.
* Harvest scene = first scene i with HV_i <= baseline - DROP_DB and HV_{i+1} <= baseline -
  SUSTAIN_DB. The second condition rejects one-scene dips.
* Harvest window = (date_{i-1}, date_i]: the cut happened after the radar's previous look
  and no later than the look that first saw the field cut.

-3 dB means HV power halved. Cutting cane removes the canopy that produces most of the
cross-pol (volume) scattering, so a halving relative to the standing-cane level is a
plausible, if arbitrary, harvest threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POL = "HVHV"
BASELINE_SCENES = 3
DROP_DB = 3.0
SUSTAIN_DB = 2.5

RULE_COLUMNS = [
    "field_id",
    "baseline_db",
    "detected",
    "date_before",
    "date_detected",
    "hv_db",
    "next_db",
]


def _detect_one(hv: pd.DataFrame) -> dict:
    """`hv` = one field's HVHV rows sorted by date."""
    db = hv["mean_db"].to_numpy()
    dates = hv["date"].to_numpy()
    baseline = float(np.median(db[:BASELINE_SCENES]))
    out = {
        "baseline_db": baseline,
        "detected": False,
        "date_before": pd.NaT,
        "date_detected": pd.NaT,
        "hv_db": float("nan"),
        "next_db": float("nan"),
    }
    # Start after the baseline scenes: a "drop" inside the period that defines "before" is
    # not a harvest we can date. The last scene has no successor, so it cannot be confirmed.
    for i in range(BASELINE_SCENES, len(db) - 1):
        if db[i] <= baseline - DROP_DB and db[i + 1] <= baseline - SUSTAIN_DB:
            out.update(
                detected=True,
                date_before=pd.Timestamp(dates[i - 1]),
                date_detected=pd.Timestamp(dates[i]),
                hv_db=float(db[i]),
                next_db=float(db[i + 1]),
            )
            break
    return out


def detect_rule(series_df: pd.DataFrame) -> pd.DataFrame:
    """Run the rule on every field. `series_df` is the exercise-3 series table (all pols).

    Returns one row per field (RULE_COLUMNS). detected=False means "no harvest detected".
    """
    hv_all = series_df[series_df["pol"] == POL]
    rows = []
    for fid, hv in hv_all.groupby("field_id", sort=True):
        hv = hv.sort_values("date").reset_index(drop=True)
        rows.append({"field_id": fid, **_detect_one(hv)})
    return pd.DataFrame(rows, columns=RULE_COLUMNS)
