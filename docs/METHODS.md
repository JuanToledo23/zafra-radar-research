# Methods: per-field harvest date from NISAR L-band HV backscatter

Code: `src/zafra/detect/` (`rule.py`, `ratio_test.py`), driver `scripts/04_detect_harvest.py`,
tests `tests/test_detect.py`.

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
