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

## Corrupt scene excluded: ascending 2026-07-18 (re-run 2026-10-29)

The 2026-07-18 ascending acquisition (GCOV and both GUNW pairs with it) is corrupt over the
AOI. The image has streaks and no field structure. Over the segments, its HV 95th
percentile is 2.6 dB below its neighbours and its ENL is 46 % above them. The coherence of
07-06/07-18 and 07-18/08-11 is a flat 0.20, against 0.38–0.54 for every other pair. The
scene-quality check (`results/5_brazil/scene_quality.csv`, docs/METHODS.md section 4)
flags it and nothing else on either site. It is listed in `nisar.exclude_dates`
(configs/brazil.yaml), and scripts 14 (`--force`), 15 and 16 were re-run without it.
Every figure below is from that re-run; the figures before it are given where they matter.

* **The July cluster is gone.** Before, 1 466 segments were dated 07-30 ± 12 d from the
  07-18 → 08-11 step. Now no segment is dated in July. The earliest date is 08-16. Cuts
  before mid-August cannot be dated. Without 07-18, the fourth acquisition of an orbit
  (three standing looks, then the cut) is 08-21 descending / 08-23 ascending, so the
  3-look baseline rule emits nothing earlier (`harvest_after` >= 08-21; the dated
  midpoint can be 08-16).
* **The radar gate shrank** from 21 171 segments (57 044 ha) to 13 411 (40 483 ha). The
  corrupt scene had added an HV drop and a coherence collapse to every segment, and gate
  A reads both as cane behaviour. Without it, **gate A no longer agrees better with
  MapBiomas than gate B** (table below). The delivered `gate_variant: radar` was chosen
  on the old figures; switching to B is a pending decision, not made here.
* **The dates are better** on the 40 + 20 sample: median absolute error 5.0 → 2.5 d,
  within ±6 d 21 → 27 of 40, missed 10 → 2 of 20. False alarms went from 4 to 6 of 40. The
  sample is redrawn from the new cane set (same seed), so the two samples are not the same
  segments.

## The gate: which segments are cane

Checked against MapBiomas Collection 10, 2024, class 20 = sugar cane. Majority class per
segment on the S2 grid. This is an independent land-cover map, not ground truth, and it
is one year older than the season. Area-weighted figures; the AOI is 87 395 ha, of which
MapBiomas maps 64 556 ha (73.9 %) as cane.

| gate | ha gated (% of AOI) | ha both | precision (ha) | recall (ha) | segment agreement |
|---|---|---|---|---|---|
| **A radar (delivered)** | 40 483 (46.3 %) | 30 307 | 74.9 % | 46.9 % | 47.3 % |
| A radar, before the 07-18 exclusion | 57 044 (65.3 %) | 41 091 | 72.0 % | 63.7 % | 50.4 % |
| B optical (pure transfer, not delivered) | 42 491 (48.6 %) | 33 042 | 77.8 % | 51.2 % | 46.8 % |
| B optical **with local labels** (not delivered) | 62 075 (71.0 %) | 55 240 | 89.0 % | 85.6 % | 76.9 % |

* **B, the pure transfer result**, is the Morelos optical gate unchanged, as delivered on
  Morelos. It finds about half of the MapBiomas cane area. Its precision is reasonable,
  but its recall is low.
* **A is delivered on this site** (`gate_variant: radar`). It was chosen because, before
  the 07-18 exclusion, it agreed better with MapBiomas than B (63.7 % recall against
  51.2 %). Without the corrupt scene, B is ahead on both precision (77.8 % against 74.9 %)
  and recall (51.2 % against 46.9 %). The choice is to be revisited. A is still the
  unchanged Morelos model.
* **"With local labels"** re-fits variant B's features on the MapBiomas majority label of
  each segment. The probabilities are out of fold (5-fold CV grouped by 3 km spatial
  blocks of segment centroids).
  * It is a reference for what local labels buy. It is not a transfer result, and it is
    never delivered.
  * It is scored against the same map it learned from, so it is optimistic even out of
    fold.
* **Error modes of A:**
  * 2 601 gated segments are MapBiomas forest formation, mostly riparian corridors (2 906
    before the exclusion);
  * 1 515 segments are MapBiomas "mosaic of uses" (3 824 before);
  * A's own confidence tier now separates: MapBiomas cane share is 73 % for "high" and
    58 % for "uncertain" (61 % and 62 % before the exclusion).
* **Why the optical gate transfers worst.** Its inputs are phenology rules (days above
  0.6, collapse in the zafra, longest high run) fitted on Morelos cane. Here the same
  features, on a shifted season, separate cane less well.

## Counts (delivered: gate A, unmerged segments)

| | segments | ha |
|---|---|---|
| all segments (after cleaning) | 32 277 | 87 395 |
| gated cane (gate A, radar) | 13 411 | 40 483 (46.3 %) |
| — gate confidence high / uncertain | 5 331 / 8 080 | |
| cane, harvest dated | 3 028 | 11 634 |
| cane, no baseline (cut before or during Jun–Jul) | 1 766 | 6 338 |
| cane, no harvest detected | 8 617 | 22 511 |
| gated cane that WorldCover maps as trees / grassland | 4 800 | 9 420 |

Before the 07-18 exclusion: 21 171 cane segments (57 044 ha), 4 780 dated, 6 576 no
baseline, 9 815 not detected.

* **Date confidence** (dated segments): high 1 132, medium 1 359, low 537. The check
  gate is now B: "high" means gate A is sure and B agrees.
* **Without the 3-look baseline rule** (the 4d rule), 4 794 segments would be dated. With
  "≥ 3 standing looks in any orbit" instead of the candidate's own orbit, 4 154 would be.
* **Merged layer** (script 15): 2 486 cane polygons (median 2.82 ha vs 1.39 ha unmerged),
  941 dated. There is no reference to score the merge, so the unmerged layer is
  delivered, as on Morelos.
* **With gate B delivered** (the first run, same detector, 07-18 still included):
  12 210 cane segments (42 491 ha), of which 3 490 dated, 4 723 no baseline and 3 997 not
  detected. Not re-run.

## Detector dates vs independent NDVI windows

Random sample (seed 5) of 40 dated and 20 undated gated-cane segments. For each segment,
the script-07 rule on the local S2 stack: the first clear scene (≤ 10 % of the segment
masked) with NDVI ≤ 0.35 after one ≥ 0.5, inside Apr 1 – Nov 30. The NDVI window is
(previous clear scene, that scene], and the error is the detector date minus the window
midpoint.

| delivered gate + detector | median abs. error (d) | mean error (d) | within ±6 d | within ±12 d | missed | false alarms |
|---|---|---|---|---|---|---|
| **A + 4b xgboost, 3-look baseline (delivered)** | 2.5 | +5.3 | 27/40 | 30/40 | 2/20 | 6/40 |
| A, before the 07-18 exclusion | 5.0 | +6.7 | 21/40 | 30/40 | 10/20 | 4/40 |
| B + 4b xgboost, 3-look baseline (first run, 07-18 included) | 4.5 | +8.1 | 25/40 | 28/40 | 7/20 | 2/40 |

* **Within ±N d:** dated segments whose absolute error is ≤ N days, out of all 40 dated
  (including those with no NDVI window).
* **False alarm:** dated by radar, but no NDVI harvest in the season.
* **Missed:** undated, but an NDVI harvest lies after the first date the baseline rule can
  emit (2026-08-21; 2026-08-11 before the 07-18 exclusion).
* **Undated A sample:** 2 of the 20 have an NDVI harvest before that date (not datable by
  construction); 16 have no NDVI harvest in the season. Dated: 34 with an NDVI window, 6
  with none.
* **The rows are different samples.** Each is drawn from that gate's own cane set, so
  they compare the two delivered products, not the detector alone. The detector and its
  dates are the same model in both. A gates in more marginal segments (more `mosaic of
  uses` and forest), which shows up as more misses and false alarms.
* **Mean error is positive** in all rows (+5.3 / +6.7 / +8.1 d), against medians of
  2.5 / 5.0 / 4.5 d: there is a tail of late radar dates. For A, 4 of the 34 segments with
  a window are outside ±12 d.
* Morelos 4b (26 labelled fields, out of fold): median error about 3.5 d, against each
  field's Sentinel-2 NDVI harvest window (script 07), so also an optical reference.

## Distribution and uncertainty of the detected dates

3 028 dated segments: **p10 2026-08-16, median 2026-09-10, p90 2026-10-15**. Area-weighted:
p10 08-16, median 09-10, p90 10-09. 100 % of the dates fall inside the Apr 1 – Nov 30
season. (Before the 07-18 exclusion: 4 780 segments, p10 07-30, median 08-28, p90 10-09.)

| month | segments | ha |
|---|---|---|
| 2026-08 | 1 090 | 4 406 |
| 2026-09 | 1 047 | 4 095 |
| 2026-10 | 891 | 3 133 |

**`date_uncertainty_days`** is half the gap between the last standing and the first
harvested acquisition. Every date is the midpoint of that gap:

| ± days | segments | ha | dates | the gap |
|---|---|---|---|---|
| 1 | 377 | 1 269 | 08-22 .. 10-09 | descending → ascending, 2 days apart |
| 5 | 2 266 | 8 715 | 08-16 .. 10-15 | ascending → descending, 10 days apart |
| 6 | 385 | 1 650 | 09-10 | ascending 09-04 → 09-16 (12 days, the descending 09-14 is missing) |

* **Corrupt scene excluded.** The former July "peak" was 1 466 segments dated 07-30 ± 12 d
  (1 230 with gate B), all from the 07-18 → 08-11 step. 07-18 is corrupt (section above),
  so most of that step was the artefact, not cuts. With 07-18 excluded, no segment is
  dated in July and the ±12 d class is empty.
* Every dated segment is now known to within ±6 d.
* Morelos, for comparison (same column, 3 976 dated segments): ±2.5 d for 2 109, ±3.5 d
  for 1 700, ±6 d for 167. Asc and desc interleave there, and there is no gap longer than
  12 days.
* **The dates cover mid-August to mid-October only, not the season.** Cuts before mid-August
  cannot be dated: there is no radar before 06-22, and the first `harvest_after` the 3-look
  baseline allows is 08-21. November is not observed yet.

## HV drop at harvest

Largest single HV drop (`hv_drop_db`), median:

| | dated cane segments | all gated cane |
|---|---|---|
| Brazil, gate A (delivered) | **−5.44 dB** (3 028) | −1.82 dB |
| Brazil, gate A before the 07-18 exclusion | −5.14 dB (4 780) | −3.68 dB |
| Brazil, gate B (first run, 07-18 included) | −5.41 dB (3 490) | −4.56 dB |
| Morelos area map (same pipeline) | −2.97 dB (3 976) | −2.60 dB |
| Morelos labelled cane fields (4c) | −3.5 dB | |

* **Brazil's drop is larger, not smaller as expected for a green harvest.** The median
  per date ranges from −4.3 to −6.1 dB with gate A (before the exclusion −3.4 to −5.6 dB;
  −3.9 to −5.8 dB with B), so it is not an artefact of one interval.
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
   Jun–Jul baseline, 1 766 cane segments (6 338 ha) are `no_baseline`, and nothing cut
   before mid-August is dated.
2. **Neither gate transfers well.** The optical gate finds 51 % of the MapBiomas cane area
   and the radar gate, without the corrupt scene, 47 %. The radar gate is delivered on the
   old figures; **which gate to deliver is open** (B now leads on precision and recall).
   Local labels would close most of the gap (86 % recall out of fold), but would no longer
   be a transfer test.
3. **Corrupt scene excluded:** ascending 2026-07-18 (`nisar.exclude_dates`), which had put
   1 466 segments at 07-30 ± 12 d.
4. **The slim GeoJSON is 27.1 MB** with gate A (13 411 cane polygons; 41.8 MB and 21 171
   before the exclusion). With gate B it was 24.9 MB. The product serves vector tiles from
   PostGIS (docs/METHODS.md).
5. **What worked:**
   * the UTM 22S descending grid was reprojected by `zonal` with no NaN values (every
     segment valid on every date);
   * the near-empty fire input (17 detections) ran without errors.
6. **Script bug fixed:** a date-formatting error in the area-weighted quantiles of
   `scripts/16_transfer_validation.py`, found on its first run.

Generated outputs: `validation.md` / `validation.json` / `validation_sample.csv`
(script 16), `area_summary.json` / `weekly.csv` / `harvest_map.png` /
`harvest_vs_fires_weekly.png` (script 14), `data/results/brazil/*.geojson` (scripts 14-15).
