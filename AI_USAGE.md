# AI usage

This project uses AI in two separate ways:

1. **AI as a development tool** (section 1): an assistant that helped write the code and
   documents. It is not part of the product.
2. **AI inside the product** (section 2): components that run when Zafra Radar processes
   data. Full design notes: [docs/AI_COMPONENTS.md](docs/AI_COMPONENTS.md).

This file is the only place in the repository that describes how AI assistance was used.

---

## 1. AI as a development tool

**Assistant.** Claude Code (Anthropic), used from the terminal on the author's machine.

**What it did.** From the author's written brief for each exercise, it wrote code
(download scripts, feature extraction, models, drivers, the product skeleton), tests and
documentation drafts; ran the scripts; and reported results, including failures and
surprises. The log below lists, per session, what it wrote, which choices it made where the
brief left them open, and what was changed or rejected in review.

**What stayed with the author.** The scientific method, the choice of data, sites and
models, the interpretation of results, and every decision recorded below as "author's
decision" or "after review". Every change was reviewed by the author before it was
committed. Where the assistant made a choice the brief left open, the log says so, and the
choice was kept or changed in review.

### Decisions log

One entry per session: date, what the AI helped with, what was changed or rejected. The
log is itemised per exercise from exercise 4b on; earlier sessions (exercises 1-4, scripts
05-09) are not itemised.


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
- 2026-10-27: Exercise 7. Claude Code wrote the product skeleton in `product/` (compose with
  PostGIS 16, FastAPI and React + MapLibre; the empty schema; the map shell with the
  50-polygon sample; `HACKATHON_PLAN.md`), the headless screenshot script, the integration
  smoke test (`tests/test_product_smoke.py`) and the doc updates, following the author's
  written brief. No data loading and no data endpoints, as briefed.
  - Choices the brief left open:
    * Esri World Imagery as the online basemap, shown only after a probe tile loads;
    * the 50 sample polygons are the dated segments nearest the median of all dated
      segments (one compact cluster), with harvest_date and confidence only;
    * the API applies the schema on every start (idempotent);
    * host port 5434 for the database (5432/5433 were taken on the author's machine).
  - Found while proving it runs:
    * `pg_isready` over the socket passed during PostGIS's first-init temporary server, so
      the API started too early; the healthcheck now uses TCP and the seed retries;
    * Docker does not publish ports of containers on an internal network only, so the
      offline override adds a TCP forwarder (`gateway`) for the browser;
    * `postgis/postgis:16-3.4` is amd64 only and runs under emulation on Apple Silicon.
  - Offline was checked with the containers on an internal network and the browser aborting
    every non-localhost request; the Mac wifi itself was not switched off.
- 2026-10-27: Exercise 8, documentation audit. Claude Code checked every numeric claim in
  `README.md`, `AI_USAGE.md` and `docs/` against the committed results and the data on
  disk, fixed the statements those contradict, restructured the documents (README for
  judges, METHODS in pipeline order, VALIDATION as one table per exercise, new
  `docs/REFERENCES.md`) and copied three result figures to `docs/figures/`, following the
  author's written brief. No code, data or results changed.
  - Reference details it could not find in the repository are marked "[URL pending]"
    rather than filled in.
- 2026-10-28: Exercise 9, web raster layers. Following the author's brief, Claude Code
  wrote `src/zafra/rasters.py`, `scripts/18_web_rasters.py` and their tests. They produce
  six-date NISAR HV / HH / cross-ratio and HV-change COGs and XYZ tiles per site, a
  `tiles_index.json`, and a raster sanity check against the 4b field means
  (`results/9/raster_check.md`). The Brazil 2026-07-18 scene was left out of the web dates
  after it was seen to be corrupt.
- 2026-10-29: Exercise 9 follow-ups, from the author's brief:
  - Scene-quality check (`src/zafra/scene_quality.py`) and per-site exclusion
    (`nisar.exclude_dates`, `zonal.drop_excluded`). Excluding the Brazil 07-18 scene was the
    author's decision. Brazil was re-run (scripts 14-16) and the documents were updated.
    The check only flags; the exclusion is written in the site config.
  - Demo API host port moved from 8000 to 8010 (author's decision; 8000 was taken by
    another local container).
  - Sentinel-1 2025-26 VH / VV series for the 38 reference fields (`src/zafra/sentinel1.py`,
    `scripts/19_l_vs_c.py`) and the L- vs C-band drop at harvest (`results/9/l_vs_c.md`).
    Orbits and tracks are kept apart, as for NISAR.
- 2026-11-11: Library-readiness audit and publication pass for v0.1.0 (author's brief).
  Claude Code audited the repository (`git` status, tests, secrets in history, packaging,
  documents) and reported. The author decided every change below:
  - The pending work went into three commits. Claude Code found that one Sentinel-1 test
    sat in the scene-quality commit; at the author's request it was moved, and the three
    local commits were rewritten.
  - The Brazil gate was switched to optical (B), and scripts 14-16 and the documents were
    re-run and updated.
  - The p36 parcel name was replaced by "grower-reported field (Tlaquiltenango)".
  - CC BY 4.0 data licence, README rewrite, packaging (`zafra-radar-research`, extras),
    CI, citation and community files, and housekeeping.
  - While doing this, Claude Code found that `pyyaml`, `pyarrow`, `scipy` and `pillow`
    were imported but not declared, and that `zafra.sentinel1` imported Earth Engine at
    module level. They are now declared, and the import is lazy, so CI runs without the
    `ee` extra.

---

## 2. AI inside the product

Three AI components run in the product; a fourth was tried and rejected. Training data,
size, metrics and limits of each: [docs/AI_COMPONENTS.md](docs/AI_COMPONENTS.md).

| component | what it is | headline result |
|---|---|---|
| Cane gate | logistic regression trained here on 38 Morelos fields | 37/38 leave-one-field-out (optical, partly circular); 34/38 radar only |
| Harvest detector | xgboost trained here on 26 Morelos cane fields | median date error 3.5 d out of fold; needs the gate (fires on 11/12 non-cane fields alone) |
| Summary generator with verifier | Claude Haiku 5.5, used as-is, constrained to the polygon's facts and checked by a deterministic verifier; template fallback | 57/58 LLM summaries pass the verifier; $0.25 per 1 000 |
| Field segmentation with SAM (rejected) | pretrained Segment Anything, used as-is | 9/38 fields matched vs 24/38 for the classical method; not used |

The alert rules, the baseline rule and the delivered segmentation are deterministic code,
not AI.
