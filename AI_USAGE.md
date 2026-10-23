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

### 1. Harvest classifier: **planned** (exercise 4b)

- **What it does:** Classifies each field and each NISAR acquisition interval as harvested or
  not harvested, giving an estimated harvest date. It runs alongside the rule-based baseline
  (exercise 4), and both sets of results are reported.
- **Trained on:** Per-field time-series features from NISAR HH/HV backscatter (with
  Sentinel-1 as a cross-check), GPM IMERG rain and FIRMS fire detections. Labels are the
  author's ground-truth harvest dates in `data/labels/`, taken from Sentinel-2 imagery and
  local knowledge. Models are Random Forest / XGBoost (scikit-learn, xgboost), with Dynamic
  Time Warping (DTW) as an alternative.
- **What it is NOT:** It is not a pretrained or third-party model, and it is not accurate
  beyond the labelled demo area. It does not detect burning; burn detection (exercise 5)
  comes from FIRMS.

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
