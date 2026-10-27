# Exercise 5: transfer to a second site (Ribeirão Preto, Brazil)

## Read this first: why Brazil is not Morelos

* **Mechanized green harvest.** Centre-South cane is cut by harvesters, unburned. The
  straw is left on the ground as a trash blanket. There is almost no burning. The
  harvest leaves a straw-covered field, not a burned, bare one.
* **Fire is not a harvest signal here.** VIIRS saw 17 fires in the AOI between Apr 1 and
  Oct 25, all before the end of July. The weekly fire-vs-harvest correlation is computed
  below, but **it is not meaningful on this site** and is not used as evidence for or
  against the detector.
* **Opposite season.** The harvest runs April to November (southern dry season). The
  rainy-season green plateau runs December to March. Every season-dependent rule was
  shifted to these dates (`configs/brazil.yaml`, table below). The NISAR record covers
  only the second half of the harvest: 2026-06-22 to 2026-10-22.
* **The Morelos models are applied unchanged.** No retraining, no re-thresholding:
  * both cane gates are fitted on the 38 Morelos `fields_v2` polygons (2025-26);
  * the harvest detector is the script-10 xgboost on the 26 Morelos cane fields;
  * the Felzenszwalb parameters and cleaning rules are those of Morelos.

  The only per-site choice is which of the two Morelos gates is delivered
  (`gate_variant`). Nothing here is an accuracy figure against ground truth: there are no
  reference fields at this site. The checks are against an independent NDVI rule and an
  independent land-cover map (MapBiomas).

## Site choice

The brief asked for Thailand. **No NISAR L2 product (GCOV or GUNW, provisional or beta)
intersects any Thai cane candidate between 2025-11-01 and 2026-05-31** (ASF search):

| candidate | NISAR L2 products, 2025-11-01 .. 2026-05-31 |
|---|---|
| Khon Kaen / Udon Thani | 0 |
| Kanchanaburi / Suphan Buri | 0 |
| Nakhon Sawan / Kamphaeng Phet | 0 |
| all of Thailand | 0 |

Thai coverage starts 2026-06-19, in the growing season. The author chose the Ribeirão
Preto fallback:

* **AOI:** lon −48.04 .. −47.76, lat −21.04 .. −20.76 (~30 × 31 km), UTM 23S (EPSG:32723).
* **Why this window:** of the windows inside all three frames, it has the highest MapBiomas
  2024 sugar-cane share (72 %; WorldCover cropland 75 %).
* **NISAR:** GCOV ascending t146 f167 (10 looks), GCOV descending t125 f99 (8 looks; its grid
  is UTM 22S and is reprojected), and GUNW ascending t146 f167 (8 pairs). Total 26 files,
  142.7 GB, all checked against ASF sizes and opened with h5py. The ASF search over
  2025-11-01 .. 2026-06-21 returns 0 scenes for any product or direction.

### Season configuration

| rule | Morelos | Brazil |
|---|---|---|
| harvest window | Nov 1 – May 31 | **Apr 1 – Nov 30 2026** (November inside the window but not yet observed) |
| crop year (NDVI phenology) | Jun 1 .. end of zafra | **2025-12-01 .. 2026-11-30** |
| `collapse_in_zafra` (gate input) | collapse in Nov–May | **collapse in Apr 1 – Nov 30** (checked at both edges) |
| radar baseline months | Nov–Dec (cane standing) | **Jun–Jul**: see below |
| delivered gate (`gate_variant`) | B, optical | **A, radar** (B kept as `p_cane_optical`) |

**The radar baseline cannot be Jan–Mar 2026.** NISAR has no acquisition here before
2026-06-22. With `baseline_months: [1, 2, 3]`:
* `season_baseline` would fall back to the same first three looks;
* `hv_pre_level_db` (a gate-A input) would be NaN on every segment.

The baseline is therefore the first radar months (Jun–Jul). By then the harvest has been
running for 2–3 months, so **the baseline is not standing cane everywhere**. Fields cut
before or during the baseline cannot be dated (`no_baseline` below).

## The gate: which segments are cane

Checked against MapBiomas Collection 10, 2024, class 20 = sugar cane. Majority class per
segment on the S2 grid. This is an independent land-cover map, not ground truth, and it
is one year older than the season. Area-weighted figures; the AOI is 87 395 ha, of which
MapBiomas maps 64 556 ha (73.9 %) as cane.

| gate | ha gated (% of AOI) | ha both | precision (ha) | recall (ha) | segment agreement |
|---|---|---|---|---|---|
| **A radar (delivered)** | 57 044 (65.3 %) | 41 091 | 72.0 % | 63.7 % | 50.4 % |
| B optical (pure transfer, not delivered) | 42 491 (48.6 %) | 33 042 | 77.8 % | 51.2 % | 46.8 % |
| B optical **with local labels** (not delivered) | 62 075 (71.0 %) | 55 240 | 89.0 % | 85.6 % | 76.9 % |

* **B, the pure transfer result**, is the Morelos optical gate unchanged, as delivered on
  Morelos. It finds about half of the MapBiomas cane area. Its precision is reasonable,
  but its recall is low.
* **A is delivered on this site** (`gate_variant: radar`). It agrees better with MapBiomas
  than B here, which is the reverse of Morelos (4c). It is still the unchanged Morelos
  model.
* **"With local labels"** re-fits variant B's features on the MapBiomas majority label of
  each segment. The probabilities are out of fold (5-fold CV grouped by 3 km spatial
  blocks of segment centroids).
  * It is a reference for what local labels buy. It is not a transfer result, and it is
    never delivered.
  * It is scored against the same map it learned from, so it is optimistic even out of
    fold.
* **Error modes of A:**
  * 2 906 gated segments (6 346 ha) are MapBiomas forest formation, mostly riparian
    corridors;
  * 3 824 segments are MapBiomas "mosaic of uses";
  * A's own confidence tier does not separate well: MapBiomas cane share is 61 % for
    "high" and 62 % for "uncertain".
* **Why the optical gate transfers worst.** Its inputs are phenology rules (days above
  0.6, collapse in the zafra, longest high run) fitted on Morelos cane. Here the same
  features, on a shifted season, separate cane less well.

## Counts (delivered: gate A, unmerged segments)

| | segments | ha |
|---|---|---|
| all segments (after cleaning) | 32 277 | 87 395 |
| gated cane (gate A, radar) | 21 171 | 57 044 (65.3 %) |
| — gate confidence high / uncertain | 11 888 / 9 283 | |
| cane, harvest dated | 4 780 | 16 590 |
| cane, no baseline (cut before or during Jun–Jul) | 6 576 | 16 349 |
| cane, no harvest detected | 9 815 | 24 105 |
| gated cane that WorldCover maps as trees / grassland | 7 910 | 14 414 |

* **Date confidence** (dated segments): high 1 990, medium 1 690, low 1 100. The check
  gate is now B: "high" means gate A is sure and B agrees.
* **Without the 3-look baseline rule** (the 4d rule), 11 356 segments would be dated. With
  "≥ 3 standing looks in any orbit" instead of the candidate's own orbit, 10 113 would be.
* **Merged layer** (script 15): 4 685 cane polygons (median 1.81 ha vs 1.22 ha unmerged),
  1 492 dated. There is no reference to score the merge, so the unmerged layer is
  delivered, as on Morelos.
* **With gate B delivered** (the first run, same detector): 12 210 cane segments
  (42 491 ha), of which 3 490 dated, 4 723 no baseline and 3 997 not detected.

## Detector dates vs independent NDVI windows

Random sample (seed 5) of 40 dated and 20 undated gated-cane segments. For each segment,
the script-07 rule on the local S2 stack: the first clear scene (≤ 10 % of the segment
masked) with NDVI ≤ 0.35 after one ≥ 0.5, inside Apr 1 – Nov 30. The NDVI window is
(previous clear scene, that scene], and the error is the detector date minus the window
midpoint.

| delivered gate + detector | median abs. error (d) | mean error (d) | within ±6 d | within ±12 d | missed | false alarms |
|---|---|---|---|---|---|---|
| **A + 4b xgboost, 3-look baseline (delivered)** | 5.0 | +6.7 | 21/40 | 30/40 | 10/20 | 4/40 |
| B + 4b xgboost, 3-look baseline (first run) | 4.5 | +8.1 | 25/40 | 28/40 | 7/20 | 2/40 |

* **Within ±N d:** dated segments whose absolute error is ≤ N days, out of all 40 dated
  (including those with no NDVI window).
* **False alarm:** dated by radar, but no NDVI harvest in the season.
* **Missed:** undated, but an NDVI harvest lies after the first date the baseline rule can
  emit (2026-08-11).
* **Undated A sample:** 3 of the 20 have an NDVI harvest before that date (not datable by
  construction); 7 have no NDVI harvest in the season.
* **The two rows are different samples.** Each is drawn from that gate's own cane set, so
  they compare the two delivered products, not the detector alone. The detector and its
  dates are the same model in both. A gates in more marginal segments (more `mosaic of
  uses` and forest), which shows up as more misses and false alarms.
* **Mean error is positive** in both (+6.7 / +8.1 d), against medians of 5.0 / 4.5 d:
  there is a tail of late radar dates. For A, 6 of the 36 segments with a window are
  outside ±12 d.
* Morelos 4b (26 labelled fields, out of fold): median error about 3.5 d, against each
  field's Sentinel-2 NDVI harvest window (script 07), so also an optical reference.

## Distribution and uncertainty of the detected dates

4 780 dated segments: **p10 2026-07-30, median 2026-08-28, p90 2026-10-09**. Area-weighted:
p10 07-30, median 08-28, p90 10-03. 100 % of the dates fall inside the Apr 1 – Nov 30
season.

| month | segments | ha |
|---|---|---|
| 2026-07 | 1 466 | 4 691 |
| 2026-08 | 1 093 | 4 233 |
| 2026-09 | 1 142 | 4 003 |
| 2026-10 | 1 079 | 3 664 |

**`date_uncertainty_days`** is half the gap between the last standing and the first
harvested acquisition. Every date is the midpoint of that gap:

| ± days | segments | ha | dates | the gap |
|---|---|---|---|---|
| 1 | 478 | 1 452 | 08-22 .. 10-09 | descending → ascending, 2 days apart |
| 5 | 2 487 | 9 073 | 08-16 .. 10-15 | ascending → descending, 10 days apart |
| 6 | 349 | 1 374 | 09-10 | ascending 09-04 → 09-16 (12 days, the descending 09-14 is missing) |
| **12** | **1 466** | **4 691** | **07-30** | **ascending 07-18 → 08-11 (24 days)** |

* **The July "peak" is 1 466 segments dated 07-30 ± 12 d, from one 24-day gap.** It is not
  a July peak. There is no ascending look on 07-30 (permanent gap 07-27 .. 08-10), so every
  cut that falls between 07-18 and 08-11 is dated at the midpoint, 07-30. With gate B
  delivered, the same artefact holds 1 230 segments. Morelos had the same mechanism in
  early December.
* 69 % of the dated segments are known to within ±6 d. The other 31 % are all from the one
  24-day gap.
* Morelos, for comparison (same column, 3 976 dated segments): ±2.5 d for 2 109, ±3.5 d
  for 1 700, ±6 d for 167. Asc and desc interleave there, and there is no gap longer than
  12 days.
* **The dates cover late July to mid-October only, not the season.** Cuts before about
  Jul 18 are invisible to the radar (no data, then the baseline). November is not observed
  yet.

## HV drop at harvest

Largest single HV drop (`hv_drop_db`), median:

| | dated cane segments | all gated cane |
|---|---|---|
| Brazil, gate A (delivered) | **−5.14 dB** (4 780) | −3.68 dB |
| Brazil, gate B (first run) | −5.41 dB (3 490) | −4.56 dB |
| Morelos area map (same pipeline) | −2.97 dB (3 976) | −2.60 dB |
| Morelos labelled cane fields (4c) | −3.5 dB | |

* **Brazil's drop is larger, not smaller as expected for a green harvest.** The median
  per acquisition date ranges from −3.4 to −5.6 dB with gate A (−3.9 to −5.8 dB with B),
  so this is not an artefact of the 24-day interval.
* **Open hypothesis, not established.** Two candidate causes:
  * larger fields, with less edge mixing of cut and standing cane or other cover within a
    segment;
  * very dry soil in the São Paulo winter (Aug–Sep), which lowers the backscatter of the
    cut field further below the standing canopy than a wetter soil would.
* A selection effect is also possible: the Morelos-trained detector may date only the
  clearest cuts. None of the three has been tested.

## Weekly harvests vs VIIRS fires (not meaningful here)

| fires | r, same week | r, fires +1 week |
|---|---|---|
| within 200 m of gated cane | −0.12 | −0.12 |
| all in the AOI | −0.13 | −0.13 |

There are 17 fires in total, all in late May to July, before the first datable date. This
is consistent with a green-harvest region, and it says nothing about the detector.

## What broke, or needs a decision

1. **No Jan–Mar baseline.** It is impossible with this coverage (see above). Under the
   Jun–Jul baseline, 6 576 cane segments (16 349 ha) are `no_baseline`, and nothing cut
   before about Jul 18 is dated.
2. **The optical gate transfers poorly** (51 % recall by area). The radar gate is delivered
   instead. Local labels would close most of the gap (86 % recall out of fold), but would
   no longer be a transfer test.
3. **The 07-18 → 08-11 gap** puts 1 466 segments at 07-30 ± 12 d. It is exposed as
   `date_uncertainty_days` and is to be shown by the frontend.
4. **The slim GeoJSON is 41.8 MB** with gate A (21 171 cane polygons). With gate B it was
   24.9 MB. The product serves vector tiles from PostGIS (docs/METHODS.md).
5. **What worked:**
   * the UTM 22S descending grid was reprojected by `zonal` with no NaN values (every
     segment valid on every date);
   * the near-empty fire input (17 detections) ran without errors.
6. **Script bug fixed:** a date-formatting error in the area-weighted quantiles of
   `scripts/16_transfer_validation.py`, found on its first run.

Generated outputs: `validation.md` / `validation.json` / `validation_sample.csv`
(script 16), `area_summary.json` / `weekly.csv` / `harvest_map.png` /
`harvest_vs_fires_weekly.png` (script 14), `data/results/brazil/*.geojson` (scripts 14-15).
