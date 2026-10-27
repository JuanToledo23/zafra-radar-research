# In-product AI components: design notes

These are the three AI components that run inside Zafra Radar (not the AI used to build it;
see `AI_USAGE.md`). Status as of 2026-10-26: the harvest classifier is implemented
(exercise 4b); the other two are not.

| # | Component | Status | Exercise |
|---|---|---|---|
| 1 | Harvest classifier (dates harvest on known cane fields) | Implemented, evaluated | 4b |
| 2 | Field segmentation | Stretch goal | 7 |
| 3 | Per-field LLM summary | Built during the hackathon | - |

---

## 1. Harvest classifier

**Status (2026-10-26).** Implemented in `src/zafra/detect/model.py`, features in
`src/zafra/features.py`, driver `scripts/10_harvest_classifier.py`. Results in
`results/4b/comparison.md` and `docs/VALIDATION.md`. The plan below was changed where noted.

**Purpose.** Estimate when each cane field was harvested, reported next to the exercise-4
rule and ratio-test detectors, not instead of them.

**Scope.** It dates the harvest on fields already known to be sugarcane. Deciding whether a
field is cane is a separate upstream step, the next exercise; the classifier does not do it.
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

**Risks (from the plan, still open).** Few labels, label noise, domain shift (region,
season, NISAR calibration). Leakage is handled: features are causal and folds split fields.

---

## 2. Field segmentation (stretch)

**Purpose.** Produce sugarcane field boundaries automatically, so the pipeline can scale
beyond the hand-drawn demo fields.

**Inputs.** Sentinel-2 L2A cloud-free images of the demo area (RGB and NIR, 10 m), from
dates when neighbouring fields look different (mid-zafra). Optionally, prompt points or boxes
placed by hand.

**Outputs.** Field polygons (GeoPackage in `data/fields/`) with an area and a confidence
score.

**Candidate approach.** A pretrained Segment Anything Model (SAM) in automatic-mask mode,
with filters on area and shape. If SAM is too slow or too heavy on a laptop, use a smaller
model or classic methods (edge detection + watershed). Needs the
`segmentation` optional dependencies (torch, segment-anything).

**Evaluation.** Intersection over Union (IoU) between predicted and hand-drawn polygons, per
field and as a mean. Also count missed fields and fields wrongly merged or split.

**Risks.**
- At 10 m resolution, smallholder plots can be only a few pixels wide.
- Neighbouring fields at the same growth stage look identical, so boundaries disappear.
- SAM was trained on natural photos, not satellite imagery.
- Runtime and model download size.

It is a stretch goal: the hand-drawn polygons remain the fallback.

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
