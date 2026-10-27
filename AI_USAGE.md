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

### 3. Per-field summary: **implemented, LLM measured live** (exercise 6)

- **What it does:** Turns one polygon's computed results into a summary of at most 70
  words, in Spanish and English. Example: "p36, 21.4 ha. Caña: confianza baja. Fecha:
  12 feb 2026 ±2.5 días." Explicit alert rules (deterministic code, not AI)
  flag fires outside the burn window, fires without a harvest, imprecise dates, gate
  disagreement and missing baselines.
- **Two renderers:**
  - A deterministic template, always available.
  - An LLM, Anthropic Claude Haiku 5.5 through the Messages API. It is given only the
    polygon's facts, pre-formatted, with a prompt (`src/zafra/prompts/field_summary.md`)
    that forbids any fact not in the input, advice and speculation.
- **Checked by:** A verifier on every output. The harvest date, uncertainty, area and
  alert codes must appear verbatim, and no other date, number or code may appear. If the
  check fails, or the API key or the API is unavailable, the template text is shown and
  the reason recorded.
- **How good it is:** The template passes the verifier on 116/116 texts (58 polygons ×
  2 languages). With the revised prompt, the LLM passes on 57 of 58 polygons (98.3 %);
  the other one hit the output-token cap and fell back to the template. Measured cost:
  $0.25 per 1,000 summaries.
- **What it is NOT:** It does not detect anything, make decisions or add facts. It is not
  legal evidence: a burn alert means a VIIRS detection near the polygon at that time. The
  verifier checks values, not meaning. The summaries are labelled as generated text.

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
- 2026-10-26: Exercise 4e. Claude Code implemented the author's five fixes to the
  area-wide map: the baseline requirement (`model.baseline_date`), fires restricted to cane,
  the WorldCover flag, fragment merging and the slim frontend export. It also wrote the
  tests and doc updates.
  - It read the baseline rule first as "any orbit". It switched to "≥ 3 standing looks in
    the candidate's own orbit" after seeing that the looser reading kept 280 detections at
    the 3rd ascending look. It reports both.
  - It counted standing looks instead of requiring all earlier looks to be standing,
    because the strict version dropped two well-founded March training fields (p8, p21).
  - Merging lowered IoU (24 → 12 fields ≥ 0.5), so, per the brief, the unmerged layer
    stays delivered. An exact-date merge was also checked (14 fields).
  - The slim file meets the 15 MB target only without the non-cane segments (8.2 MB vs
    36-61 MB). The author approved leaving them out (non-cane is not drawn).
  - After review, at the author's request: undated cane segments that WorldCover maps as
    trees or grassland get confidence "uncertain". The pre-Dec-20 limit is now stated as
    a consequence of coverage starting in Nov 2025.
- 2026-10-27: Exercise 5, transfer to Ribeirão Preto (Brazil). Claude Code did the site
  refactor (`configs/`, `zafra.site`, `--site` on scripts 00 and 12-16). It then ran the
  pipeline on the new site with the Morelos models unchanged, and wrote
  `results/5_brazil/transfer.md` and the doc sections.
  - It searched ASF for the Thai candidates and found no NISAR L2 coverage in 2025-26. The
    author chose the Brazil fallback.
  - The author asked for a Jan–Mar 2026 radar baseline. Claude Code checked ASF and found
    0 scenes before 2026-06-22. It kept Jun–Jul and reported the consequence (4 723
    `no_baseline` segments) instead of silently changing the rule.
  - A VS Code restart lost the session mid-download. It rebuilt the state from disk
    (sizes against ASF, h5py open, flock probes), removed the unheld lock files and
    resumed the one partial file.
  - It added the HV-drop comparison and the in-season share to script 16, and fixed a
    date-formatting bug found on its first run.
  - It reported as measured, not fixed:
    * the optical gate's low MapBiomas recall (51 % by area);
    * an HV drop larger than in Morelos, the opposite of the expectation for a green
      harvest;
    * the 07-30 cadence pile-up;
    * the 24.9 MB slim GeoJSON.
  - Author's decisions after review:
    * B is kept as the pure transfer result, and the radar gate A is delivered on Brazil
      through a new per-site `gate_variant` key;
    * a "with local labels" reference row: B re-fitted on MapBiomas with 5-fold CV by
      3 km spatial blocks (the block size was Claude Code's choice), never delivered;
    * `date_uncertainty_days` on every harvest GeoJSON;
    * the HV-drop difference written as an open hypothesis;
    * the slim-file size accepted, since the product serves vector tiles;
    * the UNICA / CONAB / MapBiomas entries. The UNICA figure was supplied by the author
      and not re-checked.

    Claude Code implemented these and re-ran both sites. With A as the delivered gate,
    the date confidence cross-checks against the other gate, B, instead of comparing A
    with itself. On Morelos, `weekly.csv` and the 4e figures stay byte-identical; the
    slim file grows from 8.2 to 8.5 MB with the two new columns.
- 2026-10-27: Exercise 6, field summaries and alert rules. Claude Code wrote the facts and
  summary module (`src/zafra/summary.py`), the six alert rules (`src/zafra/alerts.py`),
  the prompt, the `alerts` / `summary` site-config sections, `scripts/17_field_summaries.py`,
  `tests/test_summary.py` and the doc updates, following the author's written brief.
  - Choices the brief left open:
    * the three-class status mapping (harvested_before_coverage counts as no_baseline);
    * giving the LLM pre-formatted strings, so the verifier can match exactly;
    * summarising each reference field through its dominant segment;
    * alerts only on gated cane polygons;
    * the LLM's model settings (Haiku 5.5, effort low, JSON output).
  - Off on Brazil: FIRE_ACTIVE_OUTSIDE_HOURS (no burn window is known for São Paulo) and, per the
    brief, HARVEST_WITHOUT_FIRE.
  - The LLM path was first exercised only with a mocked client (no API key).
  - Reported as measured, not fixed:
    * HIGH_DATE_UNCERTAINTY never fires (largest uncertainty 6 days);
    * FIRE_ACTIVE_OUTSIDE_HOURS hits are all night overpasses;
    * HARVEST_WITHOUT_FIRE fires on 91 % of dated harvests;
    * p36's dominant segment covers only 27 % of the field.
  - After review, at the author's request:
    * HARVEST_WITHOUT_FIRE disabled on both sites;
    * BURN_OUTSIDE_HOURS renamed FIRE_ACTIVE_OUTSIDE_HOURS, with the overpass in the
      evidence;
    * the 38 reference fields summarised from the field-level 4b/4c outputs, with
      `segments_covering`.
  - Measured with live calls once the key was in `.env`: 58 calls, 41 passed, 17 too long.
  - Second review, at the author's request:
    * confidence split into `cane_confidence` and `date_precision_days`, stated
      separately;
    * the prompt rewritten (55-word budget, fixed content, no interpretation language);
    * a forbidden-phrase check added to the verifier.

    Re-run live: 57/58 passed. Claude Code also found that rounding P before the 0.5
    threshold changed `cane_confidence` on 2 segments, and fixed it.
