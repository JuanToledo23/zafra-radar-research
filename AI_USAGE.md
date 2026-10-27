# AI usage

This project uses AI in two separate ways, kept apart below:

1. **AI as a development tool**: helps write the code. It is not part of the product.
2. **AI inside the product**: components that run when Zafra Radar processes data.

---

## AI as a development tool

Claude Code (Anthropic) is used to speed up project scaffolding and scripting: repository
setup, download scripts, boilerplate and documentation drafts. The scientific method, data
choices, interpretation of results and validation are the author's own, and every
AI-generated change is reviewed before it is committed.

---

## AI inside the product

Design details for each component: `docs/AI_COMPONENTS.md`.

### 1. Harvest classifier: **implemented and evaluated** (exercise 4b)

- **What it does:** For a field already known to be sugarcane, classifies each NISAR
  acquisition as standing or harvested and gives an estimated harvest date. Deciding
  whether a field is cane is a separate upstream step (next exercise). It is reported
  next to the rule-based and ratio-test detectors (exercise 4), on the same fields.
- **Trained on:** Per-acquisition features from NISAR GCOV HV/HH (ascending and descending),
  NISAR GUNW coherence and FIRMS VIIRS fires. GPM IMERG rain is computed but not used: it
  marks the season rather than the harvest in this data. The training set is 26 cane
  fields in Morelos over one zafra (2025-11 to 2026-05): 746 labelled acquisitions. The labels
  are each field's Sentinel-2 NDVI harvest window (exercise 7), not dated field visits.
  The models are logistic regression and a small xgboost (scikit-learn, xgboost), trained
  here from scratch.
- **How good it is:** Validated by holding out whole fields (5-fold GroupKFold). The median
  date error is 3.5-4.5 days, against 11.5 days for the ratio test (both orbits) and 19 for
  the rule, on the same 26 fields. With 26 fields, differences of 2-3 days are not
  significant.
- **What it is NOT:** It is not a pretrained or third-party model. It cannot tell whether a
  field is cane or was harvested: it fires on 10-11 of the 11 non-cane control fields. It is only
  tested on one season in one area. It does not detect burning; fires come from FIRMS.
  Details and failure cases: `docs/AI_COMPONENTS.md`, `docs/VALIDATION.md`.

### 2. Field segmentation: **stretch goal** (exercise 7)

- **What it does:** Draws sugarcane field boundaries automatically from Sentinel-2 imagery,
  so they don't have to be digitised by hand.
- **Model / data:** A pretrained Segment Anything Model (SAM) or a lightweight segmentation
  model, run on Sentinel-2 L2A images of the demo area. It will be scored against
  hand-drawn polygons.
- **What it is NOT:** It does not identify the crop and does not detect harvest. If it falls
  short, the hand-drawn polygons are used instead.

### 3. Per-field natural-language summary: **planned** (built during the hackathon)

- **What it does:** Turns one field's detector output into a short readable paragraph
  (e.g. "Field 12 was probably burned on ~Mar 14 and harvested between Mar 11 and Mar 23").
- **Prompted with:** A fixed template filled only with that field's computed results: dates,
  confidence, fire detections and data gaps.
- **What it is NOT:** It does not detect anything, make decisions or add facts. Every number
  in the summary comes from the detectors. It is cosmetic, not scientific, and is checked by
  hand.

---

## Log

Append one line per session: date, what the AI helped with, what was changed or rejected.

- 2026-10-23: Project bootstrap (layout, environment, NISAR/FIRMS download scripts, docs
  skeleton). The asf_search and FIRMS API details were checked against the installed library
  and the live documentation.
- 2026-10-23: Added in-product AI scope (dependencies, AI_COMPONENTS.md, glossary terms,
  `detect/` and `models/` placeholders). No in-product AI is implemented yet.
- 2026-10-26: Exercise 4b. Claude Code wrote the feature table (`src/zafra/features.py`),
  the classifier and date rule (`src/zafra/detect/model.py`), the driver
  (`scripts/10_harvest_classifier.py`), tests and the doc updates, following the author's
  written brief. It added the ablation (coherence, sharpness, rain/gap removed) to test the
  brief's hypothesis. Results and failure cases (false alarms on every non-cane field) are
  reported as measured. After review, at the author's request: the ratio test was also run
  on both orbits; rain, gap_days, sharpness and slope were dropped from the model; and the
  scope (dates harvest on known cane fields only) was stated in the docs.
