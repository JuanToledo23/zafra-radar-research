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
  whether a field is cane is a separate upstream step (the cane gate, exercise 4c). It is reported
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

### 1b. Cane / non-cane gate: **implemented and evaluated** (exercise 4c)

- **What it does:** For any field polygon and zafra, gives P(sugarcane), so the harvest
  classifier only runs on cane. The output is `p_cane`, a yes/no at 0.5, a confidence
  tier, and `p_cane_radar` (the radar-only probability, for the map).
- **Trained on:** One row per field.
  - The operational gate uses Sentinel-2 NDVI phenology.
  - A radar-only model (NISAR HV/HH statistics, largest HV drop, pre-season HV level, GUNW
    coherence) is always reported next to it as independent evidence.
  - **38 training rows** (26 cane, 12 non-cane) in Morelos, zafra 2025-26.
  - Labels come from NDVI phenology and visual tracing, not field visits.
  - Logistic regression, trained here from scratch.
- **How good it is:** Leave-one-field-out:
  - Optical gate: 37/38 correct, no cane field dropped. This is partly circular: the labels
    come from NDVI.
  - Radar only: 34/38 correct, 23/26 cane fields found, 1 false positive (probably cane in
    fact). It drops the one grower-confirmed field, which is why it is not the gate.
  - As a gate, it cuts the harvest detectors' false alarms on the 12 non-cane fields to 1.
- **What it is NOT:** It is not validated against field visits. The NDVI variants are
  partly circular because the labels come from NDVI. It is untested on land covers absent
  from the 38 fields (orchards, towns). The radar check misses cane with a weak radar cut,
  including the one grower-confirmed field (p36).

### 2. Field segmentation: **SAM evaluated, not used** (exercise 4d)

- **What it does:** Draws candidate field polygons over the whole supply area from
  Sentinel-2, so the gate and the harvest classifier can run area-wide.
- **Model / data:**
  - Segment Anything (SAM), a **pretrained foundation model** from Meta AI.
  - ViT-B checkpoint, Apache 2.0 licence, through samgeo.
  - Used as-is, with no training or fine-tuning.
  - Run on a 10 m true-colour scene and on an NDVI-history composite of the area.
- **How good it is:**
  - SAM matched 9 (true colour) and 2 (NDVI composite) of the 38 hand-traced fields with
    IoU ≥ 0.5. It merges neighbouring cane fields into blocks.
  - A classical, non-AI segmentation (Felzenszwalb on the NDVI history) matched 24 of 38,
    median IoU 0.58.
  - **The delivered polygons come from the classical method.**
- **What it is NOT:** It does not identify the crop or detect harvest. The delivered
  segments are fragments of fields, not cadastral fields.

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
- 2026-10-26: Exercise 4c. Claude Code wrote the per-field cane features
  (`src/zafra/cane_features.py`, plus `optical.s2_bare_share`), the gate
  (`src/zafra/detect/cane_gate.py`), the driver (`scripts/11_cane_gate.py`), tests and the
  doc updates, following the author's written brief.
  - It added an extra "radar + fire" variant.
  - It first proposed radar-only (A) as the default. After review, the author chose optical
    (B) as the operational gate, because A drops p36, the only grower-confirmed field. A is
    always returned as `p_cane_radar`.
  - A+F scored one field more than A but was not adopted: the fire coefficient has the
    wrong sign and depends on one field.
  - Reported as measured, not fixed: the "largest HV drop" feature often lands on the
    late-April storms rather than the cut.
- 2026-10-26: Exercise 4d. Claude Code wrote the Sentinel-2 / WorldCover tiled export
  (`optical.py`), the segmentation module (`segments.py`), the vectorized per-segment
  statistics (`zonal.py`, `fires.fires_per_field_fast`), the area-wide plots (`plots.py`),
  drivers 12-14, tests and the doc updates, following the author's written brief.
  - SAM (ViT-B) was run as asked on both inputs. It produced merged blocks, so the
    fallback of the brief (classical segmentation) is the delivered method.
  - A watershed was tried first for the fallback and replaced by Felzenszwalb after it
    under-segmented most fields. Three Felzenszwalb settings were compared on the same 38
    fields; this is disclosed as optimistic.
  - It added a per-date NDVI stack download and the bincount zonal path, because
    per-polygon Earth Engine calls would not scale to ~40 000 segments. That path was
    checked against the 38-field tables of 4b/4c before use.
  - It defined the date `confidence` tiers (gate certainty × radar agreement); the brief
    did not specify them.
  - Reported as measured, not fixed: the early-December detection spike, the fire peak in
    April without matching detections, cane calls on WorldCover tree cover, and the
    missing official cane-area figure (TODO).
