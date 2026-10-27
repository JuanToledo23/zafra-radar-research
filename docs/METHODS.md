# Methods: per-field harvest date from NISAR L-band HV backscatter

Code: `src/zafra/detect/` (`rule.py`, `ratio_test.py`), driver `scripts/04_detect_harvest.py`,
tests `tests/test_detect.py`. The trained multi-signal classifier (exercise 4b) is described
in its own section below.

## Data

* **NISAR L2 GCOV** (CRID P05023, provisional), track 19, frame 11, ascending, 15 scenes
  2025-11-23 .. 2026-05-10 at the 12-day revisit, ~12:05 UTC (06:05 local). Same orbit and
  map grid every date, so a field polygon samples the same pixels with the same viewing
  geometry. Backscatter is gamma-0 (radiometrically terrain corrected).
* **Polarization:** HV (`HVHV`, cross-pol). HV comes mostly from volume scattering in the
  canopy; cutting the cane removes that volume, so HV drops. HH is kept in the series but
  not used for detection.
* **Fields:** 6 hand-drawn polygons in Morelos (84-718 valid pixels each).
* **Context data:** VIIRS 375 m active fires (NASA FIRMS, SNPP/NOAA-20/NOAA-21), used only to
  cross-check; GPM IMERG V07 rain (Earth Engine), used only as a rain guard when available.

## Preprocessing

For each field and date, pixels whose centre falls inside the polygon and that the GCOV
`mask` marks valid are averaged **in linear power**. The mean intensity is the physical
quantity (and the one whose statistics are known); dB = 10 log10 is used for display and to
express magnitudes. Averaging per-pixel dB would compute a geometric mean, biased low by
speckle.

## Statistical model

Speckle makes a pixel's intensity gamma-distributed with mean sigma (the true backscatter)
and shape L (the number of looks). The mean of N independent pixels of L looks is gamma with
shape N L. Two consequences follow (Oliver & Quegan 2004, ch. 4 and 12):

1. **Precision.** The standard error of a field mean in dB is (10/ln 10) / sqrt(N L), about
   0.06-0.15 dB here.
2. **Ratio test.** If sigma did not change between dates 1 and 2, the ratio of field means
   R = I2 / I1 follows an F distribution with (2 N2 L, 2 N1 L) degrees of freedom,
   independent of sigma. This is the SAR ratio detector (Rignot & van Zyl 1993) and the
   likelihood-ratio test for equal gamma means. We use the one-sided p-value
   p = P(F <= R), because harvest can only lower HV; a two-sided test would also fire on
   wet-soil increases and lose power in the direction of interest.

**Equivalent number of looks (ENL).** L in the formulas is how many *independent* looks the
data behave like, not the nominal `numberOfLooks` stored in the product (~4.6). We estimate
it per field as ENL = mean^2 / variance of the pixel intensities, averaged over the first
three scenes (standing cane, most uniform). ENL is scale-free, so a uniformly wetter scene
does not bias it. Estimated ENL is 3.2-4.3, below the nominal looks. That is expected:
correlated samples count for less than their number, and real within-field texture adds
variance. The texture part makes the test conservative. Both values are reported.

## Decision rule (primary: intensity-ratio test)

* **Baseline:** look-weighted mean intensity of the first three scenes, leaving out any
  scene flagged as possibly wet (below). A look-weighted mean of gamma variables is gamma
  with the summed shape, so the F test against it is exact: dof (2 N_i ENL, 2 ENL Sum N_b).
* **Tests at each later scene i:** p_prev (vs scene i-1) and p_base (vs baseline).
* **Detection:** first scene i (after the baseline period) with
  `p_prev < 0.01` or `p_base < 0.001`, **and** HV at i and at i+1 both at least 2 dB
  below the baseline.
* **Output:** harvest window (t_{i-1}, t_i], reported as its midpoint +/- 6 days (half the
  revisit); p_prev, p_base; drop vs baseline in dB with standard error
  (10/ln 10) sqrt(1/(N_i ENL) + 1/(ENL Sum N_b)); confidence = high if p < 0.001 and drop
  > 4 dB, medium if p < 0.01, low otherwise.

**Why the sustained criterion carries the decision.** With N ENL of 300-2500 looks, the test
flags changes of a few tenths of a dB. Its null hypothesis is "only speckle differs", but an
uncut field also changes from date to date (soil and canopy moisture, growth, wind). Over
this season, standing-cane scene-to-scene changes of ~1 dB give p < 1e-10. So the p-value
answers "is this more than speckle?" and the 2 dB, two-scene criterion answers "is this a
lasting loss of canopy?". A one-scene dip (field_04's sawtooth) or the fall after a wet-soil
spike fails the second question.

The baseline test lets a gradual decline (field_05, partial or progressive harvest) be
declared once it accumulates, even when no single step is significant.

## Rule baseline (method A, for comparison)

Baseline = median HV (dB) of the first three scenes. Harvest = first scene with HV <=
baseline - 3 dB and the next scene <= baseline - 2.5 dB. No noise model: the same
thresholds apply to an 84-pixel and a 718-pixel field. It is fragile near the threshold
(field_03 misses by 0.02 dB).

## Confounders and how they are handled

| Confounder | Effect on HV | Handling |
|---|---|---|
| Rain / wet soil | Rise, then a fall back that mimics harvest | A scene with HV rise > 1.5 dB vs the previous one (or GPM > 5 mm in the 2 days before, when Earth Engine is available) is flagged; the next scene cannot be declared on the step test, only vs baseline; flagged scenes are dropped from the baseline. GPM is a hook (`gpm_rain_before`), not yet validated. |
| Dry / senescent cane before harvest | Gradual decline, can cross 2 dB before the cut | Not separable with HV alone; disclosed. Probably why field_06 is detected ~1 month before its reference (FIRMS burn 02-23/24). HH/HV ratio and polarimetric tests are future work. |
| Partial or staggered harvest | Gradual or two-step decline | Baseline test declares it once accumulated; the date is then the first scene 2 dB down, not the cut of the last row. |
| Burning before cutting | Burn alone changes HV little; the cut does | Burn date from FIRMS is reported next to the window, not used in the decision. |
| Small fields | Few pixels, wide null, mixed edge pixels | Handled statistically (smaller N gives larger p and se); field_04 (84 px) is flagged uncertain. Edge pixels from neighbours are limited by the pixel-centre rule. |
| Spatial correlation of pixels | N overstates independent samples; p too small | Diagnostic only: effective N from lag-1 autocorrelation (separable AR(1)); the detections do not change with it (see script output). |

## Limitations

* The F test assumes independent pixels and a pure-speckle null; neither holds exactly. The
  p-values are best read as a ranking of evidence, not literal error rates.
* Thresholds (2 dB, 1.5 dB, alpha) are chosen a priori, not fitted. With six fields and only
  four usable reference windows there is no independent validation set.
* Dating resolution is the 12-day revisit.
* GCOV product is provisional (P05023); calibration may change between releases.
* GPM rain is ~11 km and is not yet connected (Earth Engine not authenticated).

## Multi-signal features and the trained classifier (exercise 4b)

Code: `src/zafra/features.py`, `src/zafra/detect/model.py`, driver
`scripts/10_harvest_classifier.py`, tests `tests/test_features.py`. Results:
`results/4b/comparison.md`, `docs/VALIDATION.md`.

**Scope.** This detector dates the harvest on fields already known to be sugarcane. Deciding
whether a field is cane is a separate upstream step, the next exercise. On non-cane fields
the detector fires almost every time; the false-alarm table in `docs/VALIDATION.md` is the
evidence for why that step is needed.

### Feature table

There is one row per (field, acquisition date, orbit) for the 38 polygons of
`fields_v2.geojson`: 15 ascending GCOV scenes (t19 f11, ~06:05 local) and 15 descending
scenes (t113 f79, ~19:13 local the day before), giving 1140 rows. The two orbits stay as
separate rows with an `orbit` column and are never averaged. Table:
`data/features/fields_v2_features.{parquet,csv}`.

| feature | definition |
|---|---|
| hv_db, hh_db, cross_ratio_db | field mean in linear power, then dB (`gcov.field_stats`, through `multisignal.extract_gcov_series`); ratio = HV − HH |
| valid_px, n_looks, enl | pixel count; product `numberOfLooks`; ENL = mean² / var of the field's linear HV pixels on that date |
| delta_hv_vs_baseline | HV − median HV of the first 3 acquisitions of that orbit in Nov-Dec |
| delta_hv_vs_prev | HV − previous acquisition of the same orbit |
| slope_hv_3 | least-squares slope (dB/day) over the previous 3 acquisitions of the same orbit, not including this one |
| sharpness | delta_hv_vs_prev − slope_hv_3 × gap_days: how far the last step departs from the local trend (a ramp gives ~0, a cut a large negative value) |
| coherence_mean, coherence_min | HH coherence of the 12-day ascending GUNW pair whose second date is this acquisition (20 m wrapped-interferogram grid, pixel centre in polygon); NaN on descending rows |
| delta_coh_vs_baseline | coherence_mean − median of the field's first 3 pairs |
| fire_7d, fire_frp_max_7d | VIIRS detections with pixel centre ≤ 200 m from the polygon in the 7 days before the acquisition time; their maximum FRP (from `data/fires/fields_v2_fires.csv`, written by `scripts/05_fires_per_field.py`) |
| rain_3d_mm | GPM IMERG V07 (Earth Engine) total over the 3 days before the acquisition time, at the field centroid |
| gap_days | days since the previous acquisition of the same orbit (12, or 24 across the missing descending scene of ~03-30) |

All features are causal: a row uses only its own acquisition and earlier ones.
`coherence_min` is the lowest single-pixel value. With hundreds of pixels it is mostly
noise, and the models give it almost no weight.

### Labels

Training uses the 26 fields with `label == cana` and an NDVI harvest window (exercise 7;
p36's window comes from exercise 9). A row is state 0 if its date is before
`harvest_window_start`, state 1 if it is after `harvest_window_end`, and is dropped from
training if it falls inside the window. That gives 746 labelled rows (343 standing, 403
harvested) and drops 34 window rows. The other 12 fields (11 `dudoso`, 1 `no_cana`) are
never used for training. They only count false alarms. Of these, p5 is `dudoso` but its
NDVI hypothesis is cane, so it is reported apart.

### Models

* Logistic regression: median imputation with a missing-value indicator, standardization,
  `class_weight="balanced"`.
* xgboost: max_depth 3, learning rate 0.05, up to 300 trees, early stopping (30 rounds)
  on an inner split of 20 % of the training *fields*. `scale_pos_weight` = n0/n1. NaN is
  handled natively. No row or column sampling (subsample = colsample_bytree = 1.0) and a
  fixed seed, so the fit is deterministic and does not depend on feature column order.
* Inputs (`model.FEATURES`): every feature above plus `is_asc`, except:
  * the absolute `hv_db` / `hh_db` (the model must be field-relative);
  * `rain_3d_mm`, `gap_days`, `sharpness` and `slope_hv_3`, which stay in the feature
    table but were dropped from the models after the first run.
    * Rain and gap_days are season markers in this data: every storm (Apr 23-28) and the
      only 24-day gap (March-April) come after every harvest. In the first run logistic
      gave rain a positive weight.
    * Sharpness and slope did not improve the dates. The ablation in `docs/VALIDATION.md`
      shows both effects.
* Validation: GroupKFold by field_id, 5 folds. Each fold model is trained on the
  labelled rows of its training fields and predicts *every* row of the held-out fields,
  including rows inside the window.
* Non-cane fields are scored by a model refit on all 26 cane fields. xgboost then uses the
  median of the per-fold best iterations, without early stopping.

### Date rule (applied to every method)

* Each field's rows, both orbits together, are ordered by acquisition time and thresholded
  at P = 0.5.
* The harvest date is the midpoint between the first acquisition classified 1 that is
  followed by another 1, and the acquisition before it.
* **Ratio test on both orbits.** The exercise-4 test runs on each orbit separately, with
  its own baseline, ENL and F tests. A scene counts as harvested if it is after the
  baseline scenes, ≥ 2 dB below baseline, and significant (step or baseline test). Those
  per-scene decisions, both orbits merged in time, go through the same rule as the models,
  so the comparison is at equal (~6-day) sampling.
* **Rule.** Unchanged from exercise 4, ascending HV only: its date is the midpoint of
  (t_{i-1}, t_i]. The ascending-only ratio test of exercise 4 is reported as a reference row.

### Limitations of the classifier

* **26 fields, one season, one region.** The folds hold 5-6 fields each, so one field
  moving by one acquisition shifts the median error by several days. Differences of 2-3
  days between variants are within that noise.
* **No negatives with a slow, cut-free decline.** Non-cane and "dudoso" fields lose 1.5-3 dB
  of HV from December to February with no coherence collapse. The models never saw such
  rows labelled 0, and they fire on 10-11 of the 11 strictly non-cane fields. Hence the
  scope above: cane/no-cane classification has to come first.
* **Feature selection used these fields.** Dropping the four features was decided after
  looking at out-of-fold results on the same 26 fields, so the reported errors are
  somewhat optimistic.
* **Labels are NDVI windows** of 2-10 days, and drying cane lowers NDVI before the cut.
  The reference is itself uncertain by a few days.

## References

* Oliver, C. & Quegan, S. (1998; 2004 reprint). *Understanding Synthetic Aperture Radar
  Images.* SciTech. Ch. 4 (speckle statistics, ENL), ch. 12 (change detection, ratio
  statistics).
* Rignot, E. J. M. & van Zyl, J. J. (1993). Change detection techniques for ERS-1 SAR data.
  *IEEE Trans. Geosci. Remote Sens.* 31(4), 896-906.
* Conradsen, K., Nielsen, A. A., Schou, J. & Skriver, H. (2003). A test statistic in the
  complex Wishart distribution and its application to change detection in polarimetric SAR
  data. *IEEE Trans. Geosci. Remote Sens.* 41(1), 4-19. Future work: a joint HH+HV test with
  the GCOV covariance terms instead of HV alone.
* Huffman, G. J. et al. GPM IMERG Final Precipitation L3 Half Hourly 0.1 x 0.1 degree V07.
  GES DISC.
