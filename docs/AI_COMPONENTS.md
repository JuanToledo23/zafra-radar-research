# In-product AI components: design notes

These are the three AI components that run inside Zafra Radar (not the AI used to build it;
see `AI_USAGE.md`). Status as of 2026-10-23: none implemented.

| # | Component | Status | Exercise |
|---|---|---|---|
| 1 | Harvest classifier | Planned | 4b |
| 2 | Field segmentation | Stretch goal | 7 |
| 3 | Per-field LLM summary | Built during the hackathon | - |

---

## 1. Harvest classifier

**Purpose.** Estimate when each field was harvested more reliably than the rule-based
baseline (exercise 4). The baseline is a fixed threshold on HV drop, which rain, soil
moisture or partial harvest can fool. The classifier is reported next to the baseline, not
instead of it.

**Inputs.** For each field and acquisition date:
- NISAR GCOV HH and HV gamma-0 (field mean in linear units, converted to dB), the HV/HH
  ratio, change from the previous date, and the field's spread (standard deviation).
- Sentinel-1 VV/VH at the nearest date, as a cross-check (optional feature).
- GPM IMERG rain in the 24-72 h before the acquisition.
- FIRMS: number of fire detections and maximum FRP within a buffer around the field since
  the previous acquisition.
- Labels: harvest dates in `data/labels/`, judged by the author from Sentinel-2 images
  (and burn scars where visible).

**Outputs.** For each field and acquisition interval, the probability that harvest happened
in that interval. Per field: the most likely harvest interval and its probability.

**Candidate approach.** Random Forest and XGBoost on hand-built per-date features
(scikit-learn / xgboost), saved with joblib. Alternative: Dynamic Time Warping (DTW), which
compares each field's backscatter curve with template harvested and unharvested curves,
using nearest neighbours.

**Evaluation.**
- Accuracy and F1 on labelled fields, compared with the rule-based baseline on exactly the
  same fields.
- Harvest-date error in days (|predicted − labelled| interval), compared with the baseline.
- **Cross-validation grouped by field** (`GroupKFold`), so dates from one field never end up
  in both training and test sets. Spatially blocked folds if the labelled fields cluster.

**Risks.**
- **Few labels.** A few dozen fields can lead to overfitting, so keep the features few and
  the models shallow.
- **Label noise.** Sentinel-2 can only bound a harvest between two cloud-free dates.
- **Domain shift.** The model is valid only for this region, season and NISAR
  provisional calibration.
- **Leakage.** Normalising features over the whole season leaks future data into earlier
  dates; compute features causally.
- **Too few dates.** The 12-day revisit gives about 15 scenes per season.

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
