# Methods

This document follows the pipeline in the order it runs:

1. [Data and sites](#1-data-and-sites)
2. [Field polygons: reference fields and segmentation](#2-field-polygons-reference-fields-and-segmentation)
3. [Cane gate](#3-cane-gate)
4. [Radar series and features](#4-radar-series-and-features)
5. [Harvest detector](#5-harvest-detector)
6. [Baseline rule, status and flags](#6-baseline-rule-status-and-flags)
7. [Alert rules](#7-alert-rules)
8. [Field summaries](#8-field-summaries)
9. [Outputs and product](#9-outputs-and-product)
10. [What we tried and rejected](#10-what-we-tried-and-rejected)

Results and their limits are in [VALIDATION.md](VALIDATION.md); the AI components in
[AI_COMPONENTS.md](AI_COMPONENTS.md); every source in [DATA_SOURCES.md](DATA_SOURCES.md);
terms in [GLOSSARY.md](GLOSSARY.md); citations [n] in [REFERENCES.md](REFERENCES.md). Prior
work on satellite monitoring of sugarcane: [1-9].

### Exercise and script map

The project was built as a series of numbered exercises. The early steps were later
renumbered, so the commit history has two "exercise 5" and two "exercise 6" commits. In
these documents the early steps are named by their script, the later ones by exercise.

| step | script(s) | what it produces |
|---|---|---|
| exercises 1-3 | `01`-`03` | GCOV inspection, per-field backscatter, full-season series |
| exercise 4 | `04_detect_harvest.py` | rule and intensity-ratio harvest detectors (reference detectors) |
| script 05 | `05_fires_per_field.py` | VIIRS fires per field (its docstring calls it exercise 5) |
| script 06 | `06_optical_validation.py` | NDVI check of the exercise-4 windows (docstring: exercise 6) |
| script 07 | `07_label_fields.py` | NDVI phenology labels and harvest windows of the 38 reference fields |
| script 08 | `08_multisignal.py` | NISAR series: GCOV ascending, descending and GUNW coherence |
| script 09 | `09_p36_ground_truth.py` | ground-truth check of p36 and its neighbours |
| exercise 4b | `10_harvest_classifier.py` | trained harvest detector |
| exercise 4c | `11_cane_gate.py` | cane / non-cane gate |
| exercise 4d | `12`-`14` | Sentinel-2 inputs, segmentation, first area-wide map |
| exercise 4e | `14`, `15_merge_fragments.py` | baseline rule, flags, fragment merging, slim export |
| exercise 5 | `16_transfer_validation.py`, `--site` on 00, 12-16 | transfer to Ribeirão Preto, Brazil |
| exercise 6 | `17_field_summaries.py` | alert rules and per-field summaries |
| exercise 7 | `product/` | product skeleton (PostGIS, FastAPI, React + MapLibre) |
| exercise 9 | `18_web_rasters.py --site` | NISAR backscatter and change rasters and XYZ tiles for the web map |
| exercise 9 | `14 --scene-quality-only`, `19_l_vs_c.py` | scene-quality check; Sentinel-1 VH/VV series and the L- vs C-band drop at harvest |

## 1. Data and sites

Full entries, licences and access paths: [DATA_SOURCES.md](DATA_SOURCES.md).

| source | role | Morelos 2025-26 | Ribeirão Preto 2026-27 |
|---|---|---|---|
| NISAR L2 GCOV, HH + HV [16, 17] | radar backscatter series | asc t19 f11, 15 scenes 2025-11-23 .. 2026-05-10; desc t113 f79, 15 scenes 2025-11-18 .. 2026-05-17 | asc t146 f167, 10 scenes; desc t125 f99, 8 scenes; 2026-06-22 .. 10-22 |
| NISAR L2 GUNW, HH coherence [17] | change between 12-day pairs | asc t19 f11, 16 pairs 2025-11-11/11-23 .. 2026-05-10/05-22 | asc t146 f167, 8 pairs |
| Sentinel-2 L2A [20] | segmentation, gate B, labels | crop year 2025-06-01 .. 2026-05-31 | 2025-12-01 .. 2026-10-20 |
| VIIRS active fires (FIRMS) [18] | features, alerts, fire check | 2024-11-01 .. 2026-05-31 | 2026-04-01 .. 2026-10-25 |
| ESA WorldCover 2021 [23] | segment cleaning, flags | yes | yes |
| GPM IMERG V07 [22] | rain context (not a model input) | 38 reference fields | – |
| MapBiomas Collection 10, 2024 [24] | independent gate check | – | yes |

Both NISAR passes are kept as separate series: their viewing geometry and local time
differ (ascending ~06:05 local, descending ~19:13 local the day before, Morelos). The GCOV
products are provisional (CRID P05023); calibration may change between releases [17].

**Sites.** Everything site-specific lives in `configs/<site>.yaml`, read by `zafra.site`;
scripts 00 and 12-16 take `--site <name>` (default `morelos`). A site holds the AOI bbox
(also the FIRMS bbox), its UTM CRS, the three NISAR sources, the season, the FIRMS dates,
the reference polygons (or `null`), every input and output path, the delivered gate
(`gate_variant`), and the alert and summary settings. The models are not part of a site:
the gate is always fitted on the 38 Morelos polygons and the detector on the 26 Morelos
cane fields, with the same thresholds, Felzenszwalb parameters and cleaning rules.

| | Morelos 2025-26 | Ribeirão Preto 2026-27 |
|---|---|---|
| AOI | lon −99.35..−99.05, lat 18.50..18.75 (Ingenio Emiliano Zapata supply area), EPSG:32614 | lon −48.04..−47.76, lat −21.04..−20.76 (~30 × 31 km), EPSG:32723 |
| season (zafra / safra) window | 2025-11-01 .. 2026-06-01 (exclusive) | 2026-04-01 .. 2026-12-01 (exclusive) |
| crop year (NDVI phenology) | 2025-06-01 .. 2026-06-01 | 2025-12-01 .. 2026-12-01 |
| radar baseline months | Nov, Dec (cane standing) | Jun, Jul (first radar months; no NISAR before 2026-06-22) |
| delivered gate | B, optical | B, optical (A until the 07-18 exclusion) |

Four rules that assumed the Mexican Nov-May season now read the season from the config
and give the old values on Morelos: the radar baseline (`features.season_baseline`), the
pre-harvest HV level (`cane_features.radar_summary`), the crop year for NDVI phenology and
the gate input `collapse_in_zafra`. After the refactor, scripts 14 and 15 reproduce the
Morelos `results/4e/weekly.csv`, the slim and full GeoJSON and both 4e figures byte for
byte; `tests/test_site.py` freezes the SHA-256 of `weekly.csv`.

**Adding a site.** Copy a config and set the AOI, UTM EPSG, frames, season and paths (paths
must not collide with another site's; tested). Then run `00_download_nisar.py` and
`00_download_firms.py`, scripts 12-15, and `16_transfer_validation.py`, all with
`--site <name>`.

## 2. Field polygons: reference fields and segmentation

### Reference fields (`fields_v2`)

38 hand-traced polygons around Zacatepec / Jojutla, Morelos (median 2.6 ha). Script 07
labels each from its Sentinel-2 NDVI phenology (a plateau ≥ 0.6, at most one collapse per
season, a high peak) plus the author's visual tracing: 26 `cana`, 11 `dudoso` (no call),
1 `no_cana`. For each cane field it also writes the NDVI harvest window
(`harvest_window_start/end`): the last clear scene ≥ 0.5 and the first ≤ 0.35. p36's
window comes from script 09. p36 (La Marometa, Tlaquiltenango, 21.4 ha) is the only field
with a grower-reported harvest. None of the labels comes from a field visit.

### Segmentation of the AOI

Code: `src/zafra/segments.py`, `src/zafra/zonal.py`, drivers `scripts/12_s2_images.py`,
`scripts/13_field_segmentation.py`; tests `tests/test_segments.py`.

**Inputs** (Sentinel-2 L2A through Earth Engine [20, 28]), all on one 10 m UTM grid
(Morelos: EPSG:32614, 3168 × 2771 px), downloaded with `getDownloadURL` in 1000 × 1000 px
tiles (400 × 400 for the stack), each on an explicit `crs_transform` of the same grid:
* the clearest scene of the full-canopy window (Morelos 2026-01-10, 99.9 % of the AOI
  clear): B4, B3, B2, B8, NDVI;
* a crop-year temporal composite: per-pixel NDVI median, standard deviation and minimum
  over 64-78 clear looks. Neighbouring fields cut on different dates differ in std and
  min even when one image shows two identical green blocks;
* a per-date NDVI stack (77 dates on Morelos, int16), for per-segment NDVI series;
* ESA WorldCover 2021 [23].

The SCL classes 4-7 are kept per pixel; same-date granules are mosaicked.

**Delivered method: Felzenszwalb** [13, 37] on the three NDVI-history layers, each
stretched to 0-1 (2-98 %); scale 100, sigma 0.5, minimum 30 px; 25 s on Morelos.

**Cleaning** (`clean_segments`): make_valid → topology-preserving simplify at 5 m →
make_valid → keep 0.5-60 ha → drop segments whose WorldCover majority is built-up (50) or
water (80). Morelos: 57 444 raw → 39 542 segments (75 684 ha); Ribeirão Preto: 32 277
segments (87 395 ha).

**Scoring** against the 38 reference fields: the segment with the largest intersection and
its IoU; over-segmented if ≥ 2 segments each cover > 20 % of the field; under-segmented if
that segment is > 1.5 × the field. Felzenszwalb: median IoU 0.58, 24/38 with IoU ≥ 0.5
(`results/4d/segmentation.md`). The two SAM variants scored far lower (section 10).

**Per-segment statistics at scale** (`zonal.py`). Each source is read once over the AOI,
the segments are burnt into a label raster on that product's grid (pixel centre inside),
and every statistic is a `np.bincount`, with the same rules as the per-field code. Run on
the 38 reference polygons, this path reproduces the per-field tables: GCOV identical on
37 fields (p5 and p7 overlap and share one pixel, max 0.05 dB), coherence identical, NDVI
max/min within 0.006, and all 38 gate decisions identical (`results/4d/area_summary.json`,
`check_vs_fields_v2`).

## 3. Cane gate

Code: `src/zafra/cane_features.py`, `src/zafra/detect/cane_gate.py`, driver
`scripts/11_cane_gate.py`, tests `tests/test_cane_gate.py`. Results:
`results/4c/comparison.md`.

**Purpose.** The harvest detector fires on almost every non-cane field (section 5). The
gate runs first: for each polygon and season it gives P(sugarcane), and the detector runs
only where the gate says cane. `cane_gate.classify_fields(gdf, zafra, variant=...)` returns
`p_cane`, `cane_pred` (P ≥ 0.5), `confidence` ("high" if P < 0.2 or > 0.8, "uncertain"
otherwise, "no_data" without data), and always both `p_cane_radar` (variant A) and
`p_cane_optical` (variant B).

**Variants** (one row per field and season; logistic regression with median imputation,
missing indicators, standardization, `class_weight="balanced"`, C = 1 [36]):

| variant | inputs | role |
|---|---|---|
| **B, optical** | NDVI peak and minimum over the crop year, days ≥ 0.6, longest run ≥ 0.6, number of collapses (> 0.3 below the previous 30-day high), `collapse_in_zafra`, share of bare pixels (NDVI ≤ 0.35) after the main collapse | delivered on both sites (the default) |
| **A, radar** | HV range and std, HH std, HV − HH mean and std, largest single HV drop, pre-season HV level, GUNW coherence min / mean / range (field means), `joint_collapse` (HV drop and coherence minimum ≤ 12 days apart), median ENL | radar evidence: always returned as `p_cane_radar`, sets the date confidence; was delivered on Ribeirão Preto until the 07-18 exclusion |
| C, combined | A + B + VIIRS fire count | evaluated only |
| A+F | A + VIIRS fire count | evaluated only |

**Training.** The 38 reference fields, season 2025-26: positive = `cana` (26), negative =
the 12 others. Leave-one-field-out evaluation. Rule baselines: NDVI (`longest_high_run ≥
150 d` and one collapse) and radar (`hv_max_single_drop ≤ −2 dB` and `joint_collapse`).

**Why B on Morelos.** The product cannot drop p36, the only grower-reported field. A drops
it (P = 0.20) with p2a and p16; B drops no cane field. The labels come from NDVI, so B's
score is partly circular; A is the independent check, shown next to B
([VALIDATION](VALIDATION.md#exercise-4c-cane-gate)). **Both sites now use B by default.**
Ribeirão Preto delivered A while the corrupt 2026-07-18 scene was in the radar tables:
judged against MapBiomas, A then agreed better (64 % recall by area against B's 51 %).
Without that scene A drops to 74.9 % precision / 46.9 % recall against B's 77.8 % / 51.2 %,
and B's dates have fewer false alarms, so the site delivers B
([VALIDATION](VALIDATION.md#exercise-5-transfer-to-brazil)). A is still computed and
reported as radar evidence on both sites.

## 4. Radar series and features

Code: `src/zafra/gcov.py`, `src/zafra/multisignal.py`, `src/zafra/features.py`; drivers
`scripts/08_multisignal.py`, `scripts/10_harvest_classifier.py`.

**Field means in linear power.** For each polygon and date, pixels whose centre falls inside
the polygon and that the GCOV `mask` marks valid are averaged in linear power, then
converted to dB. Averaging per-pixel dB would give a geometric mean, biased low by speckle.

**Speckle statistics** [10]. A pixel's intensity is gamma-distributed with mean σ and shape
L (the looks); the mean of N independent pixels is gamma with shape N L. The standard error
of a field mean in dB is (10/ln 10)/√(N L), about 0.06-0.15 dB here. L is the equivalent
number of looks, ENL = mean² / variance of the field's pixel intensities, estimated per
field (3.2-4.3 on the six exercise-4 fields, below the nominal ~4.6 looks, because
neighbouring pixels are correlated; `outputs/detect/detections.csv`).

**Coherence.** HH coherence magnitude of each 12-day ascending GUNW pair, from the 20 m
wrapped-interferogram grid, mean and minimum over pixels whose centre is in the polygon.
Low coherence across a pair means the field changed (cut, burned or trafficked). No
deformation processing is done.

**Feature table** (`data/features/fields_v2_features.parquet`): one row per (field,
acquisition, orbit), 1 140 rows for the 38 fields; the orbits are never averaged. All
features are causal (they use only that acquisition and earlier ones).

| feature | definition | in the model |
|---|---|---|
| delta_hv_vs_baseline | HV − median HV of the first 3 acquisitions of that orbit in the baseline months | yes |
| delta_hv_vs_prev | HV − previous acquisition of the same orbit | yes |
| cross_ratio_db | HV − HH | yes |
| valid_px, n_looks, enl | pixel count; product `numberOfLooks`; ENL of the field's linear HV pixels | yes |
| coherence_mean, coherence_min | coherence of the ascending pair ending on this date; NaN on descending rows | yes |
| delta_coh_vs_baseline | coherence_mean − median of the field's first 3 pairs | yes |
| fire_7d, fire_frp_max_7d | VIIRS detections ≤ 200 m from the polygon in the 7 days before; their maximum FRP | yes |
| is_asc | orbit flag | yes |
| hv_db, hh_db | absolute levels | no (the model must be field-relative) |
| slope_hv_3, sharpness | HV slope over the previous 3 looks; last step minus that trend | no (section 10) |
| rain_3d_mm, gap_days | GPM rain in the 3 days before; days since the previous look | no (section 10) |

**Scene quality and excluded acquisitions** (`zafra.scene_quality`, script 14). Every
GCOV acquisition is summarised over the AOI by two numbers: the 95th percentile of the
segments' HV field means and the median ENL. Each is compared with the median of up to
two acquisitions before and two after it in the same orbit. A scene is flagged when HV p95
moves more than 2 dB or ENL more than 30 %. The harvest moves both slowly; a corrupt scene
moves them at once. The check only flags (`<results_dir>/scene_quality.csv`). Exclusion is
a decision written in the site config (`nisar.exclude_dates`: date, sources, reason) after
looking at the image. `zafra.zonal.drop_excluded` then removes the GCOV rows and every GUNW
pair that has the date as reference or secondary. One scene is flagged on either site:
Ribeirão Preto ascending 2026-07-18 (HV p95 −2.6 dB, ENL +46 %), which is excluded.
Morelos has none (30 acquisitions; the largest deviations are 0.7 dB and 24 %).

## 5. Harvest detector

Code: `src/zafra/detect/model.py`, driver `scripts/10_harvest_classifier.py`, tests
`tests/test_features.py`. Results: `results/4b/comparison.md`.

**Scope.** It dates the harvest on polygons the gate has already called cane. It does not
decide whether a polygon is cane.

**Labels.** The 26 `cana` fields with an NDVI harvest window. A row is 0 (standing) before
`harvest_window_start`, 1 (harvested) after `harvest_window_end`, and dropped inside the
window: 746 labelled rows (343 standing, 403 harvested), 34 dropped. The 12 non-cane fields
are only used to count false alarms.

**Models** [35, 36].
* **xgboost (delivered):** max_depth 3, learning rate 0.05, up to 300 trees, early stopping
  (30 rounds) on an inner split of 20 % of the training fields, `scale_pos_weight` = n0/n1,
  no row or column sampling, fixed seed (deterministic). The model applied to new polygons
  is refit on all 26 fields with the median of the per-fold best iterations: 249 trees.
* **Logistic regression (alternative, kept as `harvest_date_logistic`):** median
  imputation with missing indicators, standardization, `class_weight="balanced"`.
* Validation: GroupKFold by field, 5 folds; each fold model predicts every row of its
  held-out fields.

**Date rule.** A field's rows, both orbits together, are ordered in time and thresholded at
P = 0.5. The candidate is the first acquisition classified 1 that is followed by another 1.
The harvest date is the midpoint between it and the acquisition before it, and
`date_uncertainty_days` is half that gap.

**Reference detectors** (exercise 4, `src/zafra/detect/rule.py`, `ratio_test.py`), scored on
the same fields:
* **Intensity-ratio test** [10, 11]. If σ did not change between two dates, the ratio of
  field means follows an F distribution with (2 N₂ L, 2 N₁ L) degrees of freedom. One-sided
  p-values against the previous scene and against a look-weighted baseline of the first
  three scenes. Detection: first scene with p_prev < 0.01 or p_base < 0.001 **and** HV at
  that scene and the next both ≥ 2 dB below baseline. The 2 dB, two-scene condition carries
  the decision: with 300-2 500 looks per field, standing-cane changes of ~1 dB already give
  p < 1e-10, so the p-value answers "more than speckle?" and the sustained drop answers
  "lasting loss of canopy?". Scenes with an HV rise > 1.5 dB are flagged as possibly wet.
* **Rule:** first ascending scene with HV ≤ baseline − 3 dB and the next ≤ baseline −
  2.5 dB, baseline = median of the first three scenes.

## 6. Baseline rule, status and flags

Code: `detect/model.py` (`baseline_date`, `dates_with_baseline`), driver
`scripts/14_area_harvest_map.py`, tests `tests/test_area_fixes.py`. Results:
`results/4e/area_summary.json`.

**Baseline requirement.** A candidate date is kept only if the polygon has, before it,
**≥ 3 acquisitions of the candidate's own orbit classified standing**. The features
measure the drop against that orbit's baseline, so standing looks of the other orbit say
nothing about it. A count is used rather than "every earlier look standing", because on
several training fields one orbit sees the cut one look before the other; the strict form
would drop well-founded March cuts (p8, p21). Only the first candidate is considered.

On Morelos the earliest datable cut is the 4th look of an orbit (descending Dec 24,
ascending Dec 29), i.e. dates from Dec 20 on.

**Status** of each gated cane polygon: `harvested` (dated); `no_baseline` (some standing
looks, but < 3 in the candidate's orbit); `harvested_before_coverage` (no standing look
before the candidate); `not_detected` (no candidate); non-cane polygons are `not_cane`.

**Date confidence** (`confidence`, script 14):
* high: the delivered gate is sure (P > 0.8) and the other gate agrees (P ≥ 0.5);
* low: the delivered gate is unsure and the other gate disagrees;
* medium: otherwise; none: no date;
* uncertain: an undated cane polygon that WorldCover maps as tree cover (10) or grassland
  (30).

**WorldCover flag.** `wc_disagrees` is true when the gate says cane and WorldCover says
tree cover or grassland. Such polygons are outlined on the map, not removed.

**Fires near cane.** For the area-wide fire check, VIIRS detections are kept if they lie
within 200 m of any gated cane polygon.

## 7. Alert rules

Code: `src/zafra/alerts.py`; parameters under `alerts` in `configs/<site>.yaml`. The rules
are deterministic code and run only on gated cane polygons. Fires are VIIRS detections
≤ 200 m from the polygon edge inside the season, with local time and overpass (FIRMS
`daynight`).

| code | fires when | Morelos | Ribeirão Preto | severity |
|---|---|---|---|---|
| FIRE_ACTIVE_OUTSIDE_HOURS | a detection's local time is outside the burn window | 09:00–17:00 | off (no window known) | warning |
| BURN_WITHOUT_HARVEST | a detection with no harvested look ≤ 14 days after it, or no date at all | on | on | warning |
| HARVEST_WITHOUT_FIRE | a dated harvest with no detection in the 7 days up to the first harvested look | off | off | info |
| HIGH_DATE_UNCERTAINTY | `date_uncertainty_days` > 9 | on | on | info |
| RADAR_OPTICAL_DISAGREE | the delivered gate says cane and the other gate's P < 0.3 | on | on | warning |
| NO_BASELINE | status `no_baseline` | on | on | info |

**Burn window.** 09:00–17:00 for Tlaquiltenango, given by the author; the municipal source
document is still to be cited. Other municipalities use the same window as an assumption
and carry `window_confirmed: false`. Regulatory context for agricultural burning in Mexico:
[31, 32].

**FIRE_ACTIVE_OUTSIDE_HOURS reports when a fire was seen, not when it was lit.** VIIRS
passes over Morelos at about 01:30 and 13:30 local. A night detection shows a fire still
burning at that hour, not its ignition time. The message names the detection time and the
overpass.

**HARVEST_WITHOUT_FIRE** is in the code but disabled on both sites (section 10).

## 8. Field summaries

Code: `src/zafra/summary.py`, prompt `src/zafra/prompts/field_summary.md`, driver
`scripts/17_field_summaries.py`, tests `tests/test_summary.py`. Design and evaluation:
[AI_COMPONENTS.md](AI_COMPONENTS.md#3-summary-generator-with-verifier).

* **Facts.** One dict per polygon copied from the outputs above; nothing is re-estimated.
  The 38 reference fields use their field-level outputs (4b date, 4c leave-one-out P,
  field fires) plus `segments_covering` (p36: 23 segments covering 94.5 % of the field).
* **Status in three classes:** `dated`, `no_baseline` (also `harvested_before_coverage`),
  `no_harvest` (`not_detected`, `not_cane`); the original is kept in `status_detail`.
* **Two confidence facts, stated separately:** `cane_confidence` (the confidence rule of
  section 6, on the unrounded P) and `date_precision_days` (= `date_uncertainty_days`).
* **Renderers:** a deterministic template, and an LLM constrained to the facts and checked
  by a verifier, with the template as fallback.

## 9. Outputs and product

**Per-segment layer** `data/results/harvest_dates_2025_26.geojson` (one row per segment):
gate probabilities and decision, harvest date and its uncertainty, status, confidence,
`hv_drop_db` (largest single HV step), `coh_min` (lowest field-mean coherence),
`fires_7d_before`, WorldCover class and flag. The unmerged segments are the delivered
layer.

**Frontend export** `data/results/harvest_map_2025_26_slim.geojson`: the gated cane
polygons, every status; 5 m simplification, coordinates rounded to 1e-6°; 12 columns
(segment_id, area_ha, p_cane, p_cane_radar, p_cane_optical, harvest_date,
date_uncertainty_days, status, confidence, hv_drop_db, coh_min, wc_disagrees). Morelos:
6 014 polygons, 8.5 MB. Ribeirão Preto: 12 210 polygons, 25.6 MB. Non-cane segments are
left out (with them the Morelos file is 36-61 MB). The weekly series for the dashboard is
`results/4e/weekly.csv`.

**Product** (`product/`, exercise 7). Self-contained (it does not import `src/zafra`):
PostGIS 16, a FastAPI service and a React + MapLibre GL web app, started with
`docker compose up --build`. The schema (`product/api/app/schema.sql`) mirrors the outputs:
`segments` (slim GeoJSON columns, GiST index), `segment_series`, `weekly`, `fires`,
`alerts`. The product will serve the map as vector tiles from PostGIS (`ST_AsMVT`), so the
slim file size does not limit it. As built it is a skeleton: empty tables, `/health` and
`/version` only, and a bundled 50-polygon Morelos sample. It starts with no container
egress (`docker-compose.offline.yml`), and the basemap [29] falls back to a plain
background offline. The build plan is `product/HACKATHON_PLAN.md`.

### Web raster layers

These are display layers for the map's before/after swipe and change overlay (exercise 9,
`zafra.rasters`, `scripts/18_web_rasters.py --site <site>`). The analysis does not use
them; field values always come from field means (section 4).

* **Dates.** Six GCOV dates from one orbit per site (`web_rasters` in the site config).
  Morelos, ascending t19: 2025-11-23, 12-29, 2026-01-22, 02-15, 03-11 and 04-28. They
  bracket the harvest front, from first cuts in mid-December through the January peak
  to the last cuts in early May. Ribeirão Preto, ascending t146: 2026-06-24, 07-06,
  08-11, 09-04, 09-28 and 10-22. Ascending has 10 of 11 cycles (descending 9), 10 looks
  (descending 8) and the site's own UTM 23S grid. **The 07-18 ascending scene is corrupt
  over this AOI.** It shows streaks and no field structure. Its segment HV 95th percentile
  is 2.6 dB below its neighbours and its ENL is 4.9 (other dates about 3.4). It is
  replaced by 07-06, and excluded from the radar tables too (section 4, scene quality).
* **Processing.** A 5 × 5 boxcar mean of linear HV and HH power over pixels the GCOV mask
  marks valid; masked pixels stay nodata. Then bilinear resampling to EPSG:3857 at the
  zoom-14 pixel size (9.55 m in Mercator units), on a grid snapped to whole z14 tiles,
  and conversion to dB. Values are clipped to a fixed stretch per kind: HV −25 … −8 dB,
  HH −18 … −2 dB, cross-ratio HV − HH −15 … −3 dB, change −10 … +3 dB. At most 1.4 % of
  pixels are clipped, except 3 % at +3 dB in Brazil's two late-season change layers.
  Change layers are differences of the unclipped dB values:
  `hv_change_<d1>_<d2>` for consecutive dates, and `hv_change_season` = lowest HV over
  the six dates minus the first. The season layer also takes the minimum of six speckled
  values, so stable ground reads about −1 dB (Morelos median) rather than 0.
* **Outputs.** Float32 COGs `data/rasters/<site>/<layer>.tif` (NaN nodata, deflate) and
  palette PNG XYZ tiles `data/tiles/<site>/<layer>/{z}/{x}/{y}.png`, zoom 10-14. Tiles
  use a grey ramp for backscatter and blue-white-red for change (white at 0 dB); nodata is
  transparent and overview zooms average the dB values. `tiles_index.json` lists the
  layers, dates, bounds, ramps, stretch and units. There are 24 layers per site. Morelos
  is 1.18 GB of COGs and 311 MB of tiles (6 360 tiles); Ribeirão Preto is 1.16 GB and
  311 MB (6 288 tiles). Each site takes about 70 s.
* **Check** (`results/9/raster_check.md`). Over the polygons of p36, p16 and p10, the
  raster's mean (in linear power) matches the 4b field means within 0.12 dB on all 18
  field-dates. A single centroid sample is within 1 dB on only 9 of 18, because the
  fields are not uniform (p36 before the cut: +2-3 dB). Map popups should therefore
  show field means, not pixel values.

## 10. What we tried and rejected

| tried | result | decision |
|---|---|---|
| **SAM** (ViT-B, via samgeo [14, 15]) on the true-colour scene and on the NDVI-history composite | median IoU 0.10 / 0.02; IoU ≥ 0.5 on 9/38 / 2/38 fields; merges contiguous cane fields into blocks | not used; Felzenszwalb (0.58, 24/38) delivered. `segments.run_sam` stays in the code |
| Watershed of the Sobel edge strength | under-segmented 27-33 of the 38 fields | replaced by Felzenszwalb |
| **3 dB rule** (exercise 4) | median error 19.2 d, late by > 6 d on 23 of 26 fields: most first post-cut looks are 1.5-3 dB below baseline | reference row only |
| **Ratio test on both orbits merged in time** | 11.5 d median vs 9.0 d ascending only: a shallow descending drop breaks the two-in-a-row run | reference row only; a per-orbit threshold was not tried |
| **sharpness and slope_hv_3** in the detector | xgboost 5.2 d vs 3.5 d without; logistic 4.0 vs 4.5 d (within noise) | dropped from the models |
| **rain_3d_mm and gap_days** in the detector | season markers here: the only storms (Apr 23-28) and the only 24-day gap come after every harvest; logistic gave rain a positive weight | dropped |
| **Fire as a model input.** A+F gate; `fire_7d` in the detector | A+F: one field more than A, but a negative fire coefficient (opposite to the physics) that depends on p8; detector: zero xgboost gain | A+F not adopted; the detector inputs are kept but carry no weight |
| Combined gate C (radar + optical + fire) | 36/38, no better than B, drops p8 | not adopted |
| Baseline counted in either orbit | keeps 496 dates in Dec 1-20 that the own-orbit rule removes | rejected |
| **Fragment merging** (adjacent cane segments with dates ≤ 6 days apart) | fields with IoU ≥ 0.5: 24 → 12 (identical dates only: 14); neighbouring fields are cut within days of each other | merged layer written for reference, unmerged delivered |
| HARVEST_WITHOUT_FIRE alert | fired on 3 618 of 3 976 dated segments (91 %) on Morelos | disabled on both sites |

Details and numbers: [VALIDATION.md](VALIDATION.md). The joint HH + HV change test of [12]
is future work.
