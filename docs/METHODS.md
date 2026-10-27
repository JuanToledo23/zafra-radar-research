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
whether a field is cane is a separate upstream step: the cane gate of exercise 4c. On non-cane fields
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

## Field selection (cane gate, exercise 4c)

Code: `src/zafra/cane_features.py`, `src/zafra/detect/cane_gate.py`, driver
`scripts/11_cane_gate.py`, tests `tests/test_cane_gate.py`. Results:
`results/4c/comparison.md`, `docs/VALIDATION.md`.

**Purpose.** The 4b detector fires on almost every non-cane field. The gate runs first: for
each polygon and zafra it gives P(sugarcane), and the detector only runs where the gate
says cane. Public API: `cane_gate.classify_fields(gdf, zafra, variant="optical")` returns the polygons with
`p_cane`, `cane_pred` (P ≥ 0.5) and `confidence` ("high" if P < 0.2 or > 0.8, "uncertain"
otherwise, "no_data" if the zafra has no data for the variant), plus `p_cane_radar`, the
radar-only probability, always.

### The circularity caveat

The cane labels of `fields_v2` come from Sentinel-2 NDVI phenology (script 07: a long
plateau ≥ 0.6, at most one collapse per season, a high peak) plus the author's visual
tracing of the polygons. Any classifier fed NDVI features is partly re-learning the rule
that produced its own labels, so its score is optimistic by construction. The script-07
rule, scored against its own labels, is 38/38 by definition. So three variants are kept
apart and all are reported:

* **A, radar only** (NISAR GCOV asc + desc, GUNW coherence). No NDVI in the inputs: this
  is the honest test of whether the field can be recognised as cane independently.
* **B, optical only** (NDVI phenology + bare-pixel share). It shows how close a model gets
  to the labelling rule. It is not independent evidence.
* **C, combined** (radar + optical + VIIRS fire). It shows what optical adds on top of
  radar.
* **A+F** (radar + VIIRS fire) is also reported. It uses no optical input either.

**Operational gate: variant B (optical).** The product cannot drop p36, the only field
with a grower-confirmed harvest. The radar-only variant A drops it (P = 0.20) together with
p2a and p16; B drops no cane field.
* The circularity caveat still holds: B's 37/38 is agreement with the rule that made the
  labels, not independent accuracy.
* The division of labour is: **optical selects the fields, radar dates the harvest** (4b).
* **A is kept as the independent check.** `classify_fields` always returns `p_cane_radar`
  (variant A) next to `p_cane` (the chosen variant), so a map can show the radar evidence
  per field.
* A is also the evidence that L-band alone separates cane from these non-cane fields: 34/38
  (89 %) leave-one-field-out, with no optical input.
* `classify_fields(..., variant="radar")` makes A the gate when optical is unavailable.

Not recommended:
* **C** (radar + optical + fire) is no better than B and drops p8.
* **A+F** scores one field above A, which is within the noise of 38 fields. Its fire
  coefficient is negative, the opposite of the physics (cane is burned before it is cut),
  and depends on one field (p8, 8 detections). Fire counts do not differ between the
  classes here (median 1 and 1): VIIRS pixels are 375 m and pick up burns in neighbouring
  fields.

### Features (one row per field × zafra)

A zafra is Nov 1 .. May 31. Radar exists only for 2025-26 (every NISAR scene is between
2025-11-18 and 2026-05-22). `has_radar` / `has_optical` flag which rows have each source.

| feature | definition |
|---|---|
| hv_range_db, hv_std_db, hh_std_db | max − min / std of the field-mean HV (HH) over the zafra, per orbit, then averaged over the two orbits |
| cross_ratio_mean_db, cross_ratio_std_db | mean / std of HV − HH, per orbit, averaged |
| hv_max_single_drop_db | most negative HV step between consecutive acquisitions of the same orbit (both orbits searched) |
| coh_min, coh_mean, coh_range | min / mean / max − min of the **field-mean** HH coherence of the 12-day ascending pairs ending in the zafra (the per-pixel minimum is noise) |
| joint_collapse | 1 if the date after the largest HV drop and the secondary date of the lowest-coherence pair are ≤ 12 days apart |
| hv_pre_level_db | median HV in Nov-Dec, per orbit, averaged (tall cane is a strong volume scatterer) |
| enl_median | median field ENL over all acquisitions |
| ndvi_max, ndvi_min | over the clear scenes (≤ 10 % of the field masked) of the **crop year**, Jun 1 .. May 31 |
| ndvi_days_above_0_6 | days covered by consecutive clear scenes both ≥ 0.6 |
| ndvi_longest_high_run_days | longest run of clear scenes ≥ 0.6, first → last (script-07 rule) |
| ndvi_n_collapses | clear scenes > 0.3 below the highest clear scene of the previous 30 days; re-armed only once NDVI is back ≥ 0.6 |
| ndvi_collapse_month | month of the largest collapse. The model uses `collapse_in_zafra` (1 Nov-May, 0 Jun-Oct, NaN if none) |
| bare_share_after_collapse | share of the field's clear 10 m pixels with NDVI ≤ 0.35 in the lowest clear scene within 30 days after the largest collapse (`optical.s2_bare_share`) |
| fire_count_in_zafra | VIIRS detections, pixel centre ≤ 200 m from the polygon, in the zafra |

Optical uses the crop year rather than the zafra because the plateau that marks cane lies
mostly in the rainy season before the cut, and an annual crop's rainy-season harvest is
what separates it from cane.

### Models and evaluation

* Logistic regression, the same pipeline as 4b (`model.make_logistic`): median imputation
  with missing indicators, standardization, `class_weight="balanced"`, C = 1.
* Training: the 38 `fields_v2` polygons in 2025-26. Positive = `label == cana` (26).
  Negative = everything else (11 `dudoso`, 1 `no_cana`). p5 is `dudoso` with a cane
  hypothesis and may in fact be cane.
* Evaluation: leave-one-field-out (38 folds). Rules have no parameters fitted to the data
  and are scored directly.
* Rule baselines: NDVI (`longest_high_run ≥ 150 d and n_collapses == 1`) and radar
  (`hv_max_single_drop ≤ −2 dB and joint_collapse == 1`).
* Gate + detector: the harvest dates of every 4b method (`results/4b/dates.csv`) count
  only on fields the gate's leave-one-out decision calls cane.

### Limitations of the gate

* **38 fields, one zafra, one area.** One field is 2.6 percentage points of accuracy.
  Differences of 1-2 fields between variants are not significant.
* **Labels are NDVI-derived** (see above). Variant A's errors are measured against an
  optical reference, not against field visits. Only p36 has a grower-confirmed date.
* **The "largest HV drop" is often not the cut.** On 8 of the 26 cane fields it falls
  after Apr 20 (the late-April storms and the drying after them), so `joint_collapse` is
  0 on 11 of the 26. The radar rule therefore recalls only 12/26. Coherence minimum and
  pre-season HV level carry most of variant A.
* **The non-cane fields are mostly small or pasture-like `dudoso` polygons** traced near
  cane, not a sample of everything a user might draw (orchards, towns, irrigated
  vegetables). Applied to other land covers, the gate is untested.

## Field delineation (exercise 4d)

Code: `src/zafra/segments.py` (segmentation, cleaning, IoU matching), `src/zafra/zonal.py`
(per-segment statistics), `src/zafra/plots.py` (area maps), drivers
`scripts/12_s2_images.py`, `scripts/13_field_segmentation.py`,
`scripts/14_area_harvest_map.py`, tests `tests/test_segments.py`. Results:
`results/4d/segmentation.md`, `results/4d/area_summary.json`, `docs/VALIDATION.md`.

**Purpose.** Replace hand-traced polygons with candidate field polygons for the whole
Ingenio Emiliano Zapata supply area (AOI = the FIRMS bbox, lon −99.35..−99.05, lat
18.50..18.75, ~760 km² of land after cleaning), so the gate (4c) and the detector (4b) can run
area-wide.

### Input images (Sentinel-2 L2A, Earth Engine)

All on one 10 m UTM 14N grid (EPSG:32614, 3168 × 2771 px), downloaded with
`getDownloadURL` in 1000 × 1000 px tiles (400 × 400 for the stack), each tile on an explicit
`crs_transform` of the same grid, so tiles abut exactly (`optical.download_tiled`).

* **Clearest scene** of 2025-12-01 .. 2026-01-31: 2026-01-10, 99.9 % of the AOI SCL-clear.
  B4, B3, B2, B8, NDVI; same-date granules mosaicked; SCL classes 4-7 kept per pixel.
* **Temporal composite**, crop year 2025-06-01 .. 2026-05-31: per-pixel NDVI median,
  standard deviation and minimum over 64-78 SCL-clear looks (tile cloud ≤ 60 %).
  Neighbouring cane fields are cut on different dates, so std and min differ across a
  boundary even when one image shows two identical green blocks.
* **NDVI stack**: one int16 band per acquisition date of the crop year (77 dates, tile
  cloud ≤ 50 %, the same pre-filter as `load_ndvi`), with codes for masked and off-swath
  pixels. Used to compute per-segment NDVI series locally (below).
* **ESA WorldCover 2021 (v200)**, for removing built-up and water segments.

### Segmentation variants

1. **SAM on the true-colour scene.** Segment Anything (Meta AI, ViT-B checkpoint
   `sam_vit_b_01ec64.pth`, Apache 2.0) through samgeo 1.4.2, automatic mask generation,
   CPU. The image is stretched once for the whole AOI (2-98 %) to 8 bit; samgeo tiles it
   into 384 px cores with 64 px overlap, each 512 px tile upsampled 2× to SAM's native
   1024 px (a 2-3 ha field becomes ~30-40 px). Generator settings: 48 points per side,
   predicted IoU ≥ 0.80, stability ≥ 0.85, minimum region 50 px. samgeo's batch mode writes
   one binary raster: the union of masks minus each mask's 1 px eroded border. Its connected
   components are the segments, grown back by 1 px. 72 tiles, 31 min.
2. **SAM on the NDVI stack**: the same, with NDVI median, std and min as the three
   channels. 72 tiles, 30 min.
3. **Classical (Felzenszwalb)**: `skimage.segmentation.felzenszwalb` on the same three
   NDVI-stack layers, each stretched to 0-1 (2-98 %); scale 100, sigma 0.5, minimum 30 px.
   25 s. A watershed of the Sobel edge strength (seeds = low-edge interiors) was tried first
   and under-segmented 27-33 of the 38 reference fields.

**Cleaning (all variants, `clean_segments`).** make_valid (keep polygonal parts) →
topology-preserving simplify at 5 m → make_valid again → keep 0.5-60 ha → drop segments
whose WorldCover majority class is built-up (50) or water (80).

### Evaluation and choice

For each of the 38 `fields_v2` polygons: the segment with the largest intersection and its
IoU; over-segmented if ≥ 2 segments each cover > 20 % of the field; under-segmented if that
segment is > 1.5× the field.

| variant | segments | median IoU | IoU ≥ 0.5 | over | under | missed |
|---|---|---|---|---|---|---|
| SAM, true colour | 5509 | 0.10 | 9/38 | 0 | 25 | 8 |
| SAM, NDVI stack | 2762 | 0.02 | 2/38 | 0 | 18 | 18 |
| **Felzenszwalb, NDVI stack (chosen)** | 39542 | **0.58** | **24/38** | 4 | 10 | 0 |

**Path taken: the classical fallback (step 2b).** SAM's output is merged blocks: adjacent
cane fields with the same colour come out as one mask, and the 1 px borders of the
binary-union output do not survive where masks overlap. SAM does well on fields that
contrast with their surroundings (pasture `dudoso` fields 0.54-0.83, p36 0.65), and badly on
blocks of contiguous cane, which is what matters here. Felzenszwalb on the NDVI history
separates neighbours cut on different dates. Its three settings (scale 50/100/200) were
compared on the same 38 fields, so 0.58 is somewhat optimistic.

**SAM is a pretrained foundation model** (Meta AI, Apache 2.0). It was used as-is, with no
fine-tuning, as an in-product AI component candidate. In this run it is evaluated and **not
used** for the delivered polygons. The installation, checkpoint and SAM masks stay out of
git.

### Per-segment features at scale (`zonal.py`)

The per-field extractors loop over polygons: one window read or one Earth Engine request
each. For ~40 000 segments every source is instead read **once** over the AOI, the
segments are burnt into a label raster on that product's grid (pixel centre inside), and
each statistic is a `np.bincount`. The rules are those of the per-field code: GCOV mean in
linear power over mask-valid, finite, > 0 pixels; valid_px the smaller of the HH and HV
counts; ENL = mean² / var of linear HV; GUNW coherence mean and minimum over finite pixels;
NDVI mean over SCL-clear pixels with cloud_pct = 1 − clear / covered. Per-scene results are
cached (`outputs/segments_ts/`), gathered into `data/features/segments_timeseries.parquet`
(GCOV, both orbits) and `segments_coherence.parquet`. Fires: one spatial join
(`fires.fires_per_field_fast`, same output as `fires_per_field`).

**Check.** Run on the 38 `fields_v2` polygons, this path reproduces the exercise 4b/4c
tables:
* GCOV: identical on 37 fields; on p5 one pixel moves to p7, because the hand-traced p5
  and p7 overlap and a label raster gives a shared pixel to one polygon (max 0.05 dB).
* Coherence: identical.
* NDVI max/min within 0.006, high-run days and collapse counts identical.
* Gate: all 38 decisions identical; P(cane) within 0.001.

### Gate and detector over the segments

* Gate: `classify_fields(..., variant="optical")` (B), with `p_cane_radar` (A).
* Detector: the 4b models refit exactly as in script 10 (26 cane fields, xgboost with the
  median CV iteration count). Applied to every gated cane segment; the harvest date is the
  sustained-rule date on xgboost (`harvest_date_logistic` kept as the alternative).
* Output `data/results/harvest_dates_2025_26.geojson`, one row per segment:
  * p_cane, p_cane_radar, cane_pred, gate_confidence;
  * harvest_date, harvest_after (first acquisition seen harvested), method;
  * hv_drop_db (largest single HV step, gate feature), coh_min (lowest field-mean
    coherence, gate feature);
  * fires_7d_before: VIIRS detections ≤ 200 m in the 7 days before `harvest_after`.
* `confidence` of a date:
  * high: the gate is sure (P > 0.8) and the radar-only gate agrees (p_cane_radar ≥ 0.5);
  * low: the gate is unsure and radar disagrees;
  * medium: otherwise;
  * none: no sustained detection.

### Limitations of the area-wide run

* The segments are **not fields**. The median cane segment is 1.2 ha against 2.6 ha for the
  hand-traced fields. Count hectares, not fields.
* Felzenszwalb over-segments the whole AOI. Hills, scrub and orchards are segmented too and
  left to the gate.
* The gate was trained on 38 polygons, mostly cane and pasture next to cane. 2036 segments
  that WorldCover maps as tree cover are called cane: tall cane is often mapped as trees,
  but some will be orchards or riparian trees. Untested.
* The detector's dates are only as good as on the 26 training fields (median error ~3.5 d),
  and it has never seen segments that mix two fields.

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
* Kirillov, A. et al. (2023). Segment Anything. *Proc. ICCV*, 4015-4026. Code and
  checkpoints: github.com/facebookresearch/segment-anything (Apache 2.0).
* Wu, Q. & Osco, L. (2023). samgeo: A Python package for segmenting geospatial data with
  the Segment Anything Model. *J. Open Source Software* 8(89), 5663.
* Felzenszwalb, P. F. & Huttenlocher, D. P. (2004). Efficient graph-based image
  segmentation. *Int. J. Computer Vision* 59(2), 167-181.
