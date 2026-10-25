"""Intensity-ratio (likelihood-ratio) harvest detector (exercise 4, method B, the primary one).

Background for developers new to SAR statistics
-----------------------------------------------
* **Speckle is multiplicative noise with a known distribution.** A radar pixel's intensity is
  the sum of echoes from many scatterers inside it; their random phases interfere, so even a
  perfectly uniform field gives pixels that scatter around the true backscatter sigma. For one
  "look" the intensity is exponentially distributed with mean sigma. Averaging L independent
  looks gives a gamma distribution with shape L and mean sigma (Oliver & Quegan, ch. 4).
  "Multiplicative" means the noise scales with sigma: a field twice as bright is twice as
  noisy in linear units, which is why we test *ratios*, not differences.

* **A field mean is just more looks.** If a field has N independent pixels of L looks each,
  its mean intensity I is the average of N*L independent exponentials: gamma with shape N*L.
  The relative noise of I is 1/sqrt(N*L), so a 400-pixel field is far more precise than one
  pixel. That is why we always average in *linear* power (the gamma result needs it) and only
  convert to dB for display and for human-readable magnitudes.

* **Why the F distribution.** A gamma(k) variable divided by its mean is chi^2 with 2k degrees
  of freedom divided by 2k. The ratio of two independent such variables is, by definition,
  F-distributed. So, *if nothing changed on the ground* (same sigma on both dates), the ratio
  R = I_2 / I_1 of two field means follows F(2*N_2*L, 2*N_1*L), whatever sigma is. This is
  the classic SAR ratio detector (Rignot & van Zyl 1993; Oliver & Quegan ch. 12), and it is
  equivalent to the likelihood-ratio test for a change in the gamma mean. The p-value is the
  probability of a ratio at least this small under "no change".

* **Why one-sided.** We are looking for harvest, and cutting cane removes the canopy that
  produces cross-pol (HV) volume scattering: HV can only go *down* at harvest. A two-sided
  test would also fire on increases (wet soil, regrowth), which are not what we want, and it
  would halve our power for the direction we care about. So p = P(F <= R) = F.cdf(R).

* **Why ENL is not the product's numberOfLooks.** The GCOV layer `numberOfLooks` (~4.6 here)
  is the *nominal* number of radar samples averaged per output pixel. The *equivalent* number
  of looks (ENL) is what the data actually behave like: ENL = mean^2 / variance over a
  region that should be homogeneous. They differ because (a) the averaged samples are
  correlated (oversampling, processing windows), so nominal looks overstate independence,
  and (b) a real field is not perfectly uniform (rows, gaps, soil patches); that texture adds
  variance and lowers the ENL. (b) makes our ENL *smaller* than the pure-speckle value, which
  makes the test conservative (wider null distribution). We estimate ENL per field from the
  first scenes, when the cane is standing and the field is as uniform as it gets, and report
  both numbers.

* **What the test does NOT model (read before trusting a p-value).**
  1. Spatial correlation between neighbouring pixels: N pixels are fewer than N independent
     samples, so N*ENL overstates the information and p-values come out too small. We report
     a diagnostic effective N from the lag-1 autocorrelation (see `effective_n`).
  2. Real temporal variability of an unchanged field (moisture, growth, wind) is not
     speckle. With hundreds of pixels the test detects changes of a few tenths of a dB, which
     happen without any harvest. That is why the decision also needs a *magnitude* that
     persists (the sustained criterion, below); the p-value alone only says "this is not
     speckle".

Decision rule
-------------
Baseline = pixel-weighted mean intensity of the first BASELINE_SCENES scenes, excluding any
flagged as a possible wet-soil event (a weighted mean of gamma variables is gamma with the
summed shape, so the F test still applies). At each later scene i:

    p_prev = P(F <= I_i / I_{i-1})   dof (2 N_i ENL, 2 N_{i-1} ENL)
    p_base = P(F <= I_i / I_base)    dof (2 N_i ENL, 2 ENL sum(N_baseline))

Harvest = first scene i with (p_prev < ALPHA_PREV or p_base < ALPHA_BASE) AND HV at i and at
i+1 both at least SUSTAIN_DB below the baseline. The baseline test lets a gradual decline
(no single big step) be declared once it has accumulated. The sustained criterion rejects
one-scene dips (and the dip-then-recover pattern of a field like field_04), and it is what
protects against p-values that are "significant" for sub-dB natural changes.

Harvest window = (t_{i-1}, t_i]; we report its midpoint +/- HALF_WINDOW_DAYS (half the 12-day
revisit).

Rain guard: wet soil raises backscatter, and the drop back to dry looks like a harvest. A
scene whose HV rises more than WET_RISE_DB above the previous one (or, when GPM rain is
supplied, had rain before the acquisition) is flagged as a possible wet-soil event; the next
scene cannot be declared on the step test (its "previous" is the inflated wet scene), only
on the baseline test, and wet scenes are left out of the baseline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

POL = "HVHV"
BASELINE_SCENES = 3
ALPHA_PREV = 0.01
ALPHA_BASE = 0.001
SUSTAIN_DB = 2.0
WET_RISE_DB = 1.5
RAIN_MM = 5.0  # GPM total in the 2 days before an acquisition that counts as "wet"
HALF_WINDOW_DAYS = 6
HIGH_DROP_DB = 4.0

# 10*log10(x) = DB_PER_NEPER * ln(x). For a gamma variable with shape k, var(ln I) ~= 1/k
# (trigamma(k) ~ 1/k for large k), so the standard error of a field mean in dB is
# DB_PER_NEPER / sqrt(N * ENL).
DB_PER_NEPER = 10 / np.log(10)

PixelProvider = Callable[[str, pd.Timestamp], np.ndarray]

DETECTION_COLUMNS = [
    "field_id",
    "enl",
    "enl_per_scene",
    "product_looks",
    "n_px",
    "baseline_dates",
    "baseline_db",
    "baseline_se_db",
    "detected",
    "date_before",
    "date_detected",
    "window_mid",
    "half_window_days",
    "trigger",
    "p_prev",
    "p_base",
    "step_db",
    "step_se_db",
    "drop_db",
    "drop_se_db",
    "confidence",
    "wet_dates",
]


# --- statistics ------------------------------------------------------------------------------


def estimate_enl(pixel_arrays) -> tuple[float, list[float]]:
    """ENL = mean^2 / variance of linear intensities, per scene, then averaged.

    ENL is scale-free (doubling every pixel leaves it unchanged), so a uniformly wetter or
    drier baseline scene does not bias it; only within-field contrast does.
    """
    per_scene = []
    for px in pixel_arrays:
        px = np.asarray(px, dtype="float64")
        px = px[np.isfinite(px) & (px > 0)]
        if px.size < 2:
            raise ValueError("need at least 2 valid pixels to estimate ENL")
        per_scene.append(float(px.mean() ** 2 / px.var(ddof=1)))
    return float(np.mean(per_scene)), per_scene


def p_decrease(ratio: float, looks_num: float, looks_den: float) -> float:
    """One-sided p-value for a decrease: P(F <= ratio) with dof (2*looks_num, 2*looks_den).

    looks_* = total independent looks behind each intensity (N pixels * ENL). Under "no
    change" ratio ~ F(2*looks_num, 2*looks_den). Very small values may underflow to 0.0.
    """
    return float(stats.f.cdf(ratio, 2 * looks_num, 2 * looks_den))


def ratio_se_db(looks_a: float, looks_b: float) -> float:
    """Standard error (dB) of 10*log10(I_a / I_b) for independent gamma means."""
    return float(DB_PER_NEPER * np.sqrt(1 / looks_a + 1 / looks_b))


def spatial_autocorr(arr2d, valid2d) -> tuple[float, float]:
    """Lag-1 correlation of linear intensity between horizontal and vertical neighbours.

    Uses only pairs where both pixels are valid. Real within-field structure also correlates
    neighbours, so this overstates speckle correlation (errs toward a smaller effective N).
    """
    a = np.asarray(arr2d, dtype="float64")
    v = np.asarray(valid2d, dtype=bool)

    def corr(x, y, ok):
        if ok.sum() < 3:
            return 0.0
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    rho_x = corr(a[:, :-1], a[:, 1:], v[:, :-1] & v[:, 1:])
    rho_y = corr(a[:-1, :], a[1:, :], v[:-1, :] & v[1:, :])
    return rho_x, rho_y


def effective_n(n: int, rho_x: float, rho_y: float) -> float:
    """Independent-sample equivalent of n correlated pixels (separable AR(1) approximation).

    For an AR(1) process with lag-1 correlation rho, the variance of a long-run mean is
    inflated by (1 + rho) / (1 - rho); in 2-D with separable correlation the factors multiply.
    """
    factor = 1.0
    for rho in (rho_x, rho_y):
        rho = min(max(rho, 0.0), 0.95)
        factor *= (1 - rho) / (1 + rho)
    return n * factor


# --- per-scene tests -------------------------------------------------------------------------


def wet_soil_flags(db: np.ndarray, rain_mm: np.ndarray | None = None) -> np.ndarray:
    """True for scenes that may be wet: HV rose > WET_RISE_DB vs the previous scene, or GPM
    reported > RAIN_MM in the 2 days before the acquisition (when rain is supplied)."""
    rise = np.r_[False, np.diff(db) > WET_RISE_DB]
    if rain_mm is not None:
        rise |= np.nan_to_num(np.asarray(rain_mm, dtype="float64")) > RAIN_MM
    return rise


def scene_tests(
    hv: pd.DataFrame, enl: float, n_scale: float = 1.0, rain_mm: np.ndarray | None = None
) -> tuple[pd.DataFrame, dict]:
    """Per-scene p-values and criteria for one field.

    `hv` = that field's HVHV rows sorted by date (needs date, mean_linear, valid_px).
    `n_scale` multiplies the pixel count (1 = treat pixels as independent; use
    effective_n / n for the spatial-correlation sensitivity check).
    Returns (per-scene table, baseline info).
    """
    hv = hv.sort_values("date").reset_index(drop=True)
    intensity = hv["mean_linear"].to_numpy(dtype="float64")
    looks = hv["valid_px"].to_numpy(dtype="float64") * n_scale * enl
    db = DB_PER_NEPER * np.log(intensity)
    wet = wet_soil_flags(db, rain_mm)

    base_idx = [j for j in range(min(BASELINE_SCENES, len(hv))) if not wet[j]] or [0]
    base_looks = float(looks[base_idx].sum())
    # Weighting by looks keeps the baseline an exact gamma variable (shape = summed looks).
    base_int = float((intensity[base_idx] * looks[base_idx]).sum() / base_looks)
    base_db = float(DB_PER_NEPER * np.log(base_int))

    n = len(hv)
    p_prev = np.full(n, np.nan)
    p_base = np.full(n, np.nan)
    for i in range(1, n):
        p_prev[i] = p_decrease(intensity[i] / intensity[i - 1], looks[i], looks[i - 1])
        if i >= BASELINE_SCENES:
            p_base[i] = p_decrease(intensity[i] / base_int, looks[i], base_looks)

    below = db - base_db
    after_wet = np.r_[False, wet[:-1]]
    # Sustained: this scene AND the next are >= SUSTAIN_DB below baseline. The last scene has
    # no successor, so it can never be confirmed.
    sustained = np.r_[(below[:-1] <= -SUSTAIN_DB) & (below[1:] <= -SUSTAIN_DB), False]
    sig_prev = (p_prev < ALPHA_PREV) & ~after_wet
    sig_base = p_base < ALPHA_BASE
    eligible = np.arange(n) >= BASELINE_SCENES

    scenes = pd.DataFrame(
        {
            "field_id": hv["field_id"].to_numpy(),
            "date": hv["date"].to_numpy(),
            "hv_db": db,
            "below_base_db": below,
            "p_prev": p_prev,
            "p_base": p_base,
            "wet_flag": wet,
            "after_wet": after_wet,
            "sustained": sustained,
            "candidate": eligible & sustained & (sig_prev | sig_base),
            "sig_prev": sig_prev,
            "sig_base": sig_base,
            "looks": looks,
        }
    )
    base = {
        "idx": base_idx,
        "dates": [hv["date"].iloc[j] for j in base_idx],
        "intensity": base_int,
        "db": base_db,
        "looks": base_looks,
        "se_db": float(DB_PER_NEPER / np.sqrt(base_looks)),
    }
    return scenes, base


def confidence_label(p: float, drop_db: float) -> str:
    """high: p < 0.001 and a drop > HIGH_DROP_DB; medium: p < 0.01; low otherwise."""
    if p < 0.001 and -drop_db > HIGH_DROP_DB:
        return "high"
    if p < 0.01:
        return "medium"
    return "low"


def _detect_one(hv: pd.DataFrame, pixels, rain_mm, n_scale) -> tuple[dict, pd.DataFrame]:
    enl, enl_per_scene = estimate_enl(pixels)
    scenes, base = scene_tests(hv, enl, n_scale=n_scale, rain_mm=rain_mm)
    row = {
        "enl": enl,
        "enl_per_scene": [round(e, 3) for e in enl_per_scene],
        "product_looks": float(hv["mean_looks"].mean()) if "mean_looks" in hv else np.nan,
        "n_px": int(hv["valid_px"].median()),
        "baseline_dates": [f"{d:%Y-%m-%d}" for d in base["dates"]],
        "baseline_db": base["db"],
        "baseline_se_db": base["se_db"],
        "detected": False,
        "date_before": pd.NaT,
        "date_detected": pd.NaT,
        "window_mid": pd.NaT,
        "half_window_days": HALF_WINDOW_DAYS,
        "trigger": "",
        "p_prev": np.nan,
        "p_base": np.nan,
        "step_db": np.nan,
        "step_se_db": np.nan,
        "drop_db": np.nan,
        "drop_se_db": np.nan,
        "confidence": "",
        "wet_dates": [f"{d:%Y-%m-%d}" for d in scenes.loc[scenes["wet_flag"], "date"]],
    }
    hits = np.flatnonzero(scenes["candidate"].to_numpy())
    if hits.size:
        i = int(hits[0])
        s, prev = scenes.iloc[i], scenes.iloc[i - 1]
        t0, t1 = pd.Timestamp(prev["date"]), pd.Timestamp(s["date"])
        trigger = "+".join(
            name for name, ok in (("prev", s["sig_prev"]), ("baseline", s["sig_base"])) if ok
        )
        # The step test is not valid right after a wet scene; report its p but do not use it.
        p_used = min(s["p_base"], s["p_prev"]) if s["sig_prev"] else s["p_base"]
        drop = float(s["below_base_db"])
        row.update(
            detected=True,
            date_before=t0,
            date_detected=t1,
            window_mid=t0 + (t1 - t0) / 2,
            trigger=trigger,
            p_prev=float(s["p_prev"]),
            p_base=float(s["p_base"]),
            step_db=float(s["hv_db"] - prev["hv_db"]),
            step_se_db=ratio_se_db(s["looks"], prev["looks"]),
            drop_db=drop,
            drop_se_db=ratio_se_db(s["looks"], base["looks"]),
            confidence=confidence_label(p_used, drop),
        )
    return row, scenes


def detect_ratio(
    series_df: pd.DataFrame,
    pixel_provider: PixelProvider,
    *,
    rain_mm: dict[str, np.ndarray] | None = None,
    n_scale: dict[str, float] | None = None,
    return_scenes: bool = False,
):
    """Run the ratio-test detector on every field of the exercise-3 series table.

    pixel_provider(field_id, date) -> 1-D array of linear HVHV values of the field's valid
        pixels on that date; called only for the first BASELINE_SCENES dates (ENL).
    rain_mm: optional {field_id: array aligned with that field's sorted dates}, GPM mm in
        the 2 days before each acquisition. None = use only the HV-rise wet-soil flag.
    n_scale: optional {field_id: effective_n / n} for the spatial-correlation sensitivity run.
    Returns a DataFrame (DETECTION_COLUMNS), or (detections, per-scene table) if
    return_scenes.
    """
    hv_all = series_df[series_df["pol"] == POL]
    rows, all_scenes = [], []
    for fid, hv in hv_all.groupby("field_id", sort=True):
        hv = hv.sort_values("date").reset_index(drop=True)
        dates = hv["date"].iloc[:BASELINE_SCENES]
        pixels = [pixel_provider(fid, pd.Timestamp(d)) for d in dates]
        row, scenes = _detect_one(
            hv,
            pixels,
            None if rain_mm is None else rain_mm.get(fid),
            1.0 if n_scale is None else n_scale.get(fid, 1.0),
        )
        rows.append({"field_id": fid, **row})
        all_scenes.append(scenes)
    det = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    if return_scenes:
        return det, pd.concat(all_scenes, ignore_index=True)
    return det


# --- inputs from disk ------------------------------------------------------------------------


class GcovPixelProvider:
    """pixel_provider backed by the GCOV files: re-reads per-pixel windows for a field.

    Uses the same pixel selection as the exercise-3 series (pixel centre inside the polygon,
    GCOV mask valid, finite and > 0), so N matches `valid_px`. Each scene is opened once and
    all fields are read from it.
    """

    def __init__(self, nisar_dir, fields_geojson, freq: str = "A", pol: str = POL):
        from zafra import timeseries as ts

        self.paths = {s["date"]: s["path"] for s in ts.list_scenes(nisar_dir)}
        self.fields = ts.load_fields(fields_geojson)
        self.freq, self.pol = freq, pol
        self._cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def _load(self, date: str) -> None:
        from rasterio.features import geometry_mask

        from zafra import gcov

        if date not in self.paths:
            raise KeyError(f"no GCOV scene for {date}")
        out = {}
        with gcov.open_gcov(self.paths[date]) as f:
            grid = gcov.read_grid(f, self.freq)
            minfo = gcov.mask_info(f, self.freq) if grid["has_mask"] else None
            for _, field in self.fields.to_crs(epsg=grid["epsg"]).iterrows():
                rows, cols, transform = gcov.window_for_geometry(grid, field.geometry)
                shape = (rows.stop - rows.start, cols.stop - cols.start)
                valid = geometry_mask(
                    [field.geometry], out_shape=shape, transform=transform, invert=True
                )
                if minfo:
                    m = gcov.read_window(f, self.freq, "mask", rows, cols)
                    valid &= gcov.valid_from_mask(m, minfo)
                arr = gcov.read_window(f, self.freq, self.pol, rows, cols).astype("float64")
                valid &= np.isfinite(arr) & (arr > 0)
                out[field["id"]] = (arr, valid)
        self._cache[date] = out

    def window(self, field_id: str, date) -> tuple[np.ndarray, np.ndarray]:
        """(2-D linear window, 2-D valid mask) for one field and date."""
        key = f"{pd.Timestamp(date):%Y-%m-%d}"
        if key not in self._cache:
            self._load(key)
        return self._cache[key][field_id]

    def __call__(self, field_id: str, date) -> np.ndarray:
        arr, valid = self.window(field_id, date)
        return arr[valid]


def gpm_rain_before(dates, bounds, days: int = 2, project: str | None = None):
    """GPM IMERG V07 rain (mm) over `bounds` (lon/lat minx, miny, maxx, maxy) in the `days`
    before each acquisition date (~12:05 UTC). Returns an array aligned with `dates`, or None
    if Earth Engine is not available/authenticated.

    TODO(rain guard): written but NOT yet run against Earth Engine (the account was not
    authenticated when this was written). Before relying on it: run `earthengine
    authenticate`, check a known rainy date, and confirm the units (IMERG `precipitation` is
    mm/hr per 30-min step, hence the 0.5 factor). Dry season (Nov-May) rain in Morelos is
    rare, so the HV-rise flag is the main guard until then.
    """
    try:
        import ee

        from zafra import config

        ee.Initialize(project=project or config.ee_project())
    except Exception:  # noqa: BLE001 (any auth/network/config failure -> fall back)
        return None

    region = ee.Geometry.Rectangle(list(bounds))
    out = []
    for d in dates:
        end = pd.Timestamp(d).tz_localize(None).normalize() + pd.Timedelta(hours=12)
        start = end - pd.Timedelta(days=days)
        total = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
            .filterDate(start.isoformat(), end.isoformat())
            .select("precipitation")
            .sum()
            .multiply(0.5)
        )
        val = total.reduceRegion(ee.Reducer.mean(), region, scale=11132).get("precipitation")
        out.append(val.getInfo())
    return np.array([np.nan if v is None else v for v in out], dtype="float64")
