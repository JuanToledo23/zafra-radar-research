# In-product AI components: design notes

These are the AI components that run inside Zafra Radar (not the AI used to build it;
see `AI_USAGE.md`). Status as of 2026-10-26: the harvest classifier (exercise 4b) and the
cane / non-cane gate (exercise 4c) are implemented. Field segmentation (exercise 4d) was
tried with SAM and runs with a classical method instead. The LLM summary is not built.

| # | Component | Status | Exercise |
|---|---|---|---|
| 1 | Harvest classifier (dates harvest on known cane fields) | Implemented, evaluated | 4b |
| 1b | Cane / non-cane gate (decides which fields the classifier runs on) | Implemented, evaluated | 4c |
| 2 | Field segmentation | SAM evaluated, not used; classical method delivered | 4d |
| 3 | Per-field LLM summary | Built during the hackathon | - |

---

## 1. Harvest classifier

**Status (2026-10-26).** Implemented in `src/zafra/detect/model.py`, features in
`src/zafra/features.py`, driver `scripts/10_harvest_classifier.py`. Results in
`results/4b/comparison.md` and `docs/VALIDATION.md`. The plan below was changed where noted.

**Purpose.** Estimate when each cane field was harvested, reported next to the exercise-4
rule and ratio-test detectors, not instead of them.

**Scope.** It dates the harvest on fields already known to be sugarcane. Deciding whether a
field is cane is a separate upstream step, the cane gate of exercise 4c; the classifier does not do it.
The false-alarm counts below show why that step is needed.

**What it does.** Classifies each NISAR acquisition of a field as standing (0) or harvested
(1). The harvest date is the midpoint between the first acquisition classified 1 that is
confirmed by the next one, and the acquisition before it.

**Inputs (as built).** One row per field, acquisition date and orbit (ascending and
descending kept separate):
- HV relative to the field's own Nov-Dec baseline and to the previous scene; HV − HH
  ratio; pixel count, product looks and ENL;
- NISAR GUNW HH coherence of the 12-day ascending pair ending on that date, and its change
  from the field's baseline;
- FIRMS VIIRS detections ≤ 200 m in the 7 days before.

Computed but left out of the model after the first run:
- GPM rain in the 3 days before, and the gap since the previous scene. Both mark the late
  season in this data.
- The HV trend over the previous 3 scenes, and "sharpness" (step minus trend). Neither
  improved the dates.

Not used, unlike the plan: Sentinel-1, the field's standard deviation, the absolute HH/HV
levels.

**Training data.** 26 cane fields in Morelos (`label == cana` in
`data/fields/fields_v2.geojson`), one zafra (2025-11-18 .. 2026-05-17). Labels come from the
Sentinel-2 NDVI harvest window of each field (exercise 7; p36 from exercise 9), not from
dated field visits. 746 labelled rows (343 standing, 403 harvested); rows inside each
window are not used for training.

**Models.** Logistic regression (standardized, class-balanced, imputed NaN plus missing
indicators) and xgboost (depth 3, ≤ 300 trees, early stopping on held-out fields). Random
Forest and DTW from the plan were not tried.

**Evaluation.**
- GroupKFold by field (5 folds). Out-of-fold dates are compared with the NDVI window
  midpoint, and with the rule and the ratio test on the same 26 fields.
- The ratio test runs on both orbits through the same date rule, so sampling is equal.
- False alarms are counted on the 12 non-cane fields.

Results on these fields:
- Median absolute date error: 4.5 days (logistic) and 3.5 days (xgboost). The ratio test
  on both orbits gets 11.5 days (9.0 on ascending only) and the rule 19.2 days.
- Within ±12 days on 24 of 26 fields.
- With 26 fields, differences of 2-3 days are within one acquisition and not significant.
- It fires on 10-11 of the 11 strictly non-cane fields.

**Limits, stated plainly.**
- It only dates a harvest on a field already known to be cane. It does not decide whether a
  field is cane or whether it was harvested. On non-cane fields the seasonal 1.5-3 dB HV
  decline triggers it almost every time.
- 26 fields, one season, one area, provisional calibration. Differences of a few days
  between model variants are within the noise of this sample.
- The four dropped features were chosen after looking at out-of-fold results on these same
  fields, so the errors above are somewhat optimistic. There is no untouched test set.
- Results are reproducible: xgboost uses no row or column sampling and a fixed seed. An
  earlier version with 0.8 column sampling moved by a few days between runs.
- Labels are optical windows of 2-10 days, not observed cut dates.
- Area-wide (exercise 4e), a date is only emitted after ≥ 3 standing acquisitions in the
  candidate's orbit; otherwise the field gets a status instead (`no_baseline`, ...). On the
  26 training fields this changes one field (p16, now undated).

**Risks (from the plan, still open).** Few labels, label noise, domain shift (region,
season, NISAR calibration). Leakage is handled: features are causal and folds split fields.

---

## 1b. Cane / non-cane gate

**Status (2026-10-26).** Implemented in `src/zafra/detect/cane_gate.py`, features in
`src/zafra/cane_features.py`, driver `scripts/11_cane_gate.py`. Results in
`results/4c/comparison.md` and `docs/VALIDATION.md`. A second trained model inside the
product, upstream of the harvest classifier.

**Purpose.** For each polygon and zafra, P(the field is sugarcane). The harvest classifier
then runs only where the gate says cane. Without it, the classifier fires on 11-12 of the
12 non-cane control fields.

**Inputs.** One row per field and zafra.
- **Operational gate, variant B (optical):** Sentinel-2 NDVI phenology over the crop year
  (peak, minimum, longest green plateau, days green, collapses and their season) and the
  share of bare pixels after the collapse.
- **Independent check, variant A (radar only), always returned as `p_cane_radar`:**
  - HV/HH variability and range, HV − HH ratio;
  - the largest single HV drop and the pre-season HV level;
  - GUNW coherence minimum, mean and range;
  - whether the HV drop and the coherence minimum coincide;
  - the median ENL.
- Also evaluated: C (all inputs plus VIIRS fire; not recommended, no gain over B and it
  drops p8) and A+F (radar + fire).

**Why B is the operational gate despite the circularity.** The product cannot drop p36, the
only grower-confirmed field. A drops it (and p2a, p16); B drops no cane field.
- Optical selects the fields, radar dates the harvest.
- A is the evidence that L-band alone separates cane here (89 % leave-one-field-out).
- A is shown next to B on the map.

**Training data.** 38 polygons in Morelos, one zafra (2025-26): 26 `cana`, 12 others
(11 `dudoso`, 1 `no_cana`). One row per field, so the training set has **38 rows**. The
labels come from NDVI phenology plus visual tracing (script 07), not from field visits. That
is why the optical variants are partly circular, and why the radar-only variant is always
reported next to the operational optical gate.

**Model.** Logistic regression (standardized, class-balanced, median-imputed with missing
indicators), the same pipeline as the harvest classifier. No tree model: with 38 rows it
would only memorize.

**Evaluation.** Leave-one-field-out:
- Optical only (operational): 37/38, all 26 cane fields kept, 1 false positive (p5, a
  `dudoso` field that is probably cane). This is partly circular.
- Radar only (independent check): 34/38, cane recall 23/26, same false positive.
- Combined: 36/38. The NDVI labelling rule: 38/38 by construction.
- As a gate in front of the 4b detectors, every variant cuts the non-cane false alarms
  from 2-12 of 12 to 1 of 12 (p5).

**Limits, stated plainly.**
- 38 training rows, one zafra, one area, provisional NISAR calibration. Differences of 1-2
  fields between variants are noise.
- The reference labels are optical, so even the radar-only score is a measure of agreement
  with NDVI, not with the ground.
- The operational gate's score is agreement with its own labelling rule.
- The radar check drops 3 of 26 cane fields, including p36, the only field with a
  grower-confirmed harvest. p36, n02 and n09 (La Marometa) show clear NDVI cuts with weak
  radar response; the cause is not known yet.
- Optical misses cuts of cane that was already dry (neighbour n03), which the radar sees.
- Negatives are mostly pasture/scrub polygons traced next to cane. Orchards, towns and
  irrigated vegetables are not in the training set, so P for them is untested.
- Radar exists only for the 2025-26 zafra. For other zafras `p_cane_radar` is NaN.

---

## 2. Field segmentation

**Status (2026-10-26).** Exercise 4d. Code in `src/zafra/segments.py`, drivers
`scripts/12_s2_images.py` and `scripts/13_field_segmentation.py`. Results in
`results/4d/segmentation.md` and `docs/VALIDATION.md`.

**Purpose.** Candidate field polygons for the whole supply area (AOI = the FIRMS bbox), so
the gate and the harvest classifier can run beyond the hand-traced fields.

**The AI component: Segment Anything (SAM).**
- A **pretrained foundation model** from Meta AI (Kirillov et al. 2023), ViT-B checkpoint,
  **Apache 2.0** licence. Run through samgeo 1.4.2, on CPU.
- Not trained or fine-tuned here. In automatic mode it is prompted with a 48 × 48 point
  grid per tile and returns object masks.
- Run on two inputs, each in 72 overlapping tiles of the 10 m Sentinel-2 AOI (~30 min per
  input):
  - the clearest Dec-Jan true-colour scene;
  - a 3-band image of the crop-year NDVI median, std and min.

**Evaluation.** IoU against the 38 hand-traced `fields_v2` polygons:

| variant | median IoU | IoU ≥ 0.5 | under-segmented | missed |
|---|---|---|---|---|
| SAM, true colour | 0.10 | 9/38 | 25 | 8 |
| SAM, NDVI stack | 0.02 | 2/38 | 18 | 18 |
| Felzenszwalb (classical), NDVI stack | **0.58** | **24/38** | 10 | 0 |

**Decision: SAM is not used for the delivered polygons.**
- It merges blocks of contiguous cane fields into single masks.
- It does well only on fields that contrast with their surroundings (pasture fields,
  IoU 0.54-0.83).
- The product uses the classical Felzenszwalb segmentation of the NDVI history instead.
  That is a deterministic algorithm, not an AI model.
- SAM stays in the code (`segments.run_sam`) so it can be re-run, for example with a
  fine-tuned checkpoint or prompted with points.

**Limits, stated plainly.**
- The delivered segments fragment fields: the median cane segment is 1.2 ha against 2.6 ha
  hand-traced. Only 24 of 38 reference fields are matched with IoU ≥ 0.5.
- The Felzenszwalb setting was chosen on the same 38 fields.
- SAM was trained on photos, and at 10 m a cane field is 15-20 pixels wide. Its failure
  here says nothing about SAM on sub-metre imagery.

**Risks (from the plan, confirmed).** At 10 m smallholder plots are a few pixels wide.
Neighbouring fields in the same growth stage look identical in one image. The temporal
stack is what separates them, and SAM did not use it well either.

---

## 3. Per-field natural-language summary (LLM)

**Purpose.** Make the per-field results readable for a non-expert (a grower, a mill
technician, a judge) in one short paragraph.

**Inputs.** Only that field's computed results, as structured data: field ID, area,
estimated harvest interval, probability from the classifier and from the baseline, FIRMS
detections (date, FRP), data gaps (e.g. 2026-07-27 to 2026-08-10, rain flags). Inserted into
a fixed prompt template.

**Outputs.** 2-4 sentences of plain text (Spanish and/or English) shown next to the field in
the product.

**Candidate approach.** One LLM API call per field with a strict template. The prompt tells
the model to use only the given values and to state uncertainty. The output is cached.

**Evaluation.** No quantitative metric; this component is cosmetic, not scientific. The
author checks a sample by hand to make sure every number matches the input.

**Risks.** The model may invent numbers, causes or certainty; to limit this, check that every
number in the output appears in the input. API cost, latency and key handling. Judges might
read the summaries as analysis, so they are labelled as generated text everywhere.
