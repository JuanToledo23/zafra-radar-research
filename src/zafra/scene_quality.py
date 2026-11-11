"""Scene-quality check: flag GCOV acquisitions that look unlike their neighbours.

    from zafra import scene_quality as SQ
    qc = SQ.scene_quality(gc)          # gc: zonal.radar_timeseries GCOV table, all scenes
    qc[qc["flagged"]]

A scene is summarised over the whole AOI by two numbers taken over the polygons (segments)
of the zonal table: the 95th percentile of the field-mean HV (dB) and the median ENL of the
HV pixels. A harvest season moves both slowly: cuts lower the HV of some fields per scene,
and the ENL depends on the processing (looks) and on the scene's texture. A corrupt or
defocused scene moves them at once (the Brazil 2026-07-18 ascending scene: HV p95 -2.6 dB,
ENL +44 % against its neighbours). Each scene is compared with the median of up to
`k` acquisitions before and `k` after it in the same orbit; it is flagged when

* |HV p95 - neighbours| > `p95_db` (default 2 dB), or
* |ENL / neighbours - 1| > `enl_rel` (default 30 %).

The check only flags. Excluding a scene is a decision recorded in the site config
(nisar.exclude_dates, zafra.zonal.drop_excluded), after looking at the image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

P95_DB = 2.0
ENL_REL = 0.30


def scene_summary(gc: pd.DataFrame, hv_col: str = "HVHV_db", enl_col: str = "enl"):
    """One row per (orbit, date): n polygons, HV p95 (dB) and median ENL over polygons."""
    df = gc.assign(date=pd.to_datetime(gc["date"]).dt.normalize())
    return (
        df.groupby(["orbit", "date"])
        .agg(
            n=(hv_col, lambda s: int(s.notna().sum())),
            hv_p95_db=(hv_col, lambda s: float(np.nanpercentile(s, 95))),
            enl_median=(enl_col, "median"),
        )
        .reset_index()
    )


def _neighbour_median(values: np.ndarray, i: int, k: int) -> float:
    nb = np.r_[values[max(i - k, 0) : i], values[i + 1 : i + 1 + k]]
    nb = nb[np.isfinite(nb)]
    return float(np.median(nb)) if nb.size else np.nan


def scene_quality(
    gc: pd.DataFrame,
    hv_col: str = "HVHV_db",
    enl_col: str = "enl",
    k: int = 2,
    p95_db: float = P95_DB,
    enl_rel: float = ENL_REL,
) -> pd.DataFrame:
    """scene_summary plus neighbour medians, deviations and the flag with its reason."""
    rows = []
    for _, s in scene_summary(gc, hv_col, enl_col).groupby("orbit", sort=False):
        s = s.sort_values("date").reset_index(drop=True)
        p95, enl = s["hv_p95_db"].to_numpy(), s["enl_median"].to_numpy()
        for i, r in s.iterrows():
            nb_p95, nb_enl = _neighbour_median(p95, i, k), _neighbour_median(enl, i, k)
            d_p95 = r["hv_p95_db"] - nb_p95
            d_enl = r["enl_median"] / nb_enl - 1 if nb_enl else np.nan
            why = []
            if abs(d_p95) > p95_db:
                why.append(f"HV p95 {d_p95:+.1f} dB")
            if abs(d_enl) > enl_rel:
                why.append(f"ENL {d_enl:+.0%}")
            rows.append(
                {
                    **r.to_dict(),
                    "nb_hv_p95_db": nb_p95,
                    "d_hv_p95_db": d_p95,
                    "nb_enl_median": nb_enl,
                    "d_enl_rel": d_enl,
                    "flagged": bool(why),
                    "reason": "; ".join(why),
                }
            )
    return pd.DataFrame(rows)
