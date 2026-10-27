# In-product AI components

The AI components that run inside Zafra Radar when it processes data. The AI used to build
the project is described in [AI_USAGE.md](../AI_USAGE.md). Status as of 2026-10-27. Method
details: [METHODS.md](METHODS.md); numbers: [VALIDATION.md](VALIDATION.md).

| # | component | model | trained here? | in the product |
|---|---|---|---|---|
| 1 | Cane gate | logistic regression | yes, 38 rows | yes |
| 2 | Harvest detector | xgboost (logistic regression as alternative) | yes, 746 rows | yes |
| 3 | Summary generator with verifier | Claude Haiku 5.5 via the Messages API, checked by a deterministic verifier; template fallback | no (pretrained LLM, used as-is) | yes |
| – | Field segmentation with SAM | Segment Anything, ViT-B | no (pretrained, used as-is) | **no, rejected** |

Everything else in the pipeline is deterministic code: the Felzenszwalb segmentation, the
baseline rule, the confidence tiers and the alert rules.

---

## 1. Cane gate

**What it does.** For each polygon and season, gives P(sugarcane), so the harvest detector
runs only on cane. Without it, the detector fires on 11 of the 12 non-cane control fields
(xgboost; 12 of 12 with logistic).

**Code.** `src/zafra/cane_features.py`, `src/zafra/detect/cane_gate.py`, driver
`scripts/11_cane_gate.py`, tests `tests/test_cane_gate.py`.

**Training data.** The 38 hand-traced Morelos polygons, season 2025-26, one row per field:
26 `cana` (positive), 11 `dudoso` and 1 `no_cana` (negative). The labels come from NDVI
phenology (script 07) plus visual tracing, not field visits.

**Size.** Logistic regression [36] (median imputation with missing indicators,
standardization, class-balanced, C = 1). Two variants are delivered: B, optical, 7 inputs
(NDVI phenology over the crop year and the bare-pixel share after the collapse, plus
missing-value indicators); A, radar, 12 inputs (HV/HH statistics, largest HV drop, pre-season HV level, GUNW coherence, ENL).
No tree model: with 38 rows it would memorize. Morelos delivers B and Ribeirão Preto A
(`gate_variant`); both probabilities are always returned (`p_cane_optical`,
`p_cane_radar`).

**Metrics** (leave-one-field-out, `results/4c/comparison.md`):

| | accuracy | cane recall | false positives |
|---|---|---|---|
| B, optical | 37/38 | 26/26 | 1 (p5, probably cane) |
| A, radar | 34/38 | 23/26 | 1 (p5) |
| Detector false alarms on the 12 non-cane fields behind either gate | 1/12 | | |
| Ribeirão Preto, A vs MapBiomas (by area) | precision 72.0 %, recall 63.7 % | | |
| Ribeirão Preto, B vs MapBiomas (by area) | precision 77.8 %, recall 51.2 % | | |

**Limits.**
- 38 rows, one season, one area; one field is 2.6 points of accuracy.
- B's score is agreement with its own NDVI labelling rule; even A's is agreement with NDVI,
  not with the ground.
- A drops p36, the only grower-reported field (P = 0.20), and p2a and p16.
- Negatives are mostly pasture or scrub traced next to cane. Orchards, towns and irrigated
  vegetables were never seen; area-wide, 2 036 Morelos segments that WorldCover maps as
  tree cover are called cane.
- The Morelos fit of B transfers poorly to Brazil (51 % recall).

---

## 2. Harvest detector

**What it does.** For a polygon the gate calls cane, classifies each NISAR acquisition as
standing or harvested and gives a harvest date with its uncertainty (half the gap between
the last standing and first harvested look). A date is only emitted after ≥ 3 standing
looks in the candidate's orbit (the baseline rule, deterministic).

**Code.** `src/zafra/features.py`, `src/zafra/detect/model.py`, driver
`scripts/10_harvest_classifier.py`; applied area-wide by `scripts/14_area_harvest_map.py`.

**Training data.** 26 Morelos cane fields, one season (NISAR acquisitions 2025-11-18 ..
2026-05-17), one row per field, acquisition and orbit: 746 labelled rows (343 standing, 403
harvested). Labels: each field's Sentinel-2 NDVI harvest window (rows inside the window are
not used). Inputs: HV change vs the orbit's baseline and vs the previous look, HV − HH, pixel
count, looks, ENL, GUNW coherence and its change, VIIRS fires ≤ 200 m in the 7 days before,
orbit flag (12 inputs).

**Size.** xgboost [35], max depth 3, 249 trees (median of the per-fold best iterations),
deterministic (no row or column sampling, fixed seed). The alternative logistic regression
uses the same inputs.

**Metrics** (out of fold, GroupKFold by field, 5 folds; `results/4b/comparison.md`):

| | median abs. error | within ±12 d | non-cane false alarms (no gate) |
|---|---|---|---|
| xgboost (delivered) | 3.5 d | 24/26 | 11/12 |
| logistic | 4.5 d | 24/26 | 12/12 |
| ratio test, both orbits (reference) | 11.5 d | 15/26 | 8/12 |
| rule (reference) | 19.2 d | 8/26 | 2/12 |
| Ribeirão Preto, behind gate A, vs NDVI rule (40 dated segments) | 5.0 d | 30/40 | 4/40 dated with no NDVI harvest |

**Limits.**
- It does not decide whether a polygon is cane: on non-cane fields the seasonal 1.5-3 dB HV
  decline triggers it almost every time.
- 26 fields, one season, one area, provisional calibration; 2-3-day differences between
  variants are noise.
- Feature selection used these same fields; there is no untouched test set.
- Drying cane can read as harvested (p6: 37 days early).
- It cannot date cuts before its 4th look of an orbit (Morelos: before Dec 20).

---

## 3. Summary generator with verifier

**What it does.** Turns one polygon's computed results into a summary of at most 70 words,
in Spanish and English, e.g. "p36, 21.4 ha. Caña: confianza baja. Fecha: 12 feb 2026 ±2.5
días." It adds no analysis and makes no decision. The alert rules it reports are
deterministic code ([METHODS, section 7](METHODS.md#7-alert-rules)).

**Code.** `src/zafra/summary.py` (facts, display strings, renderers, verifier), prompt
`src/zafra/prompts/field_summary.md`, driver `scripts/17_field_summaries.py`, tests
`tests/test_summary.py`. Model settings under `summary` in `configs/<site>.yaml`.

**Training data.** None. The LLM is Anthropic Claude Haiku 5.5 (`claude-haiku-5-5`) through
the Messages API [38], used as-is: effort low, JSON output with keys `es` and `en`, up to
1 024 output tokens. It receives only the polygon's facts as pre-formatted strings.

**Size.** One API call per polygon for both languages; 1 378 input and 217 output tokens per
call on average; $0.25 per 1 000 summaries at $0.10 / $0.50 per million tokens.

**How it is constrained.**
1. **Facts** are copied from the outputs; nothing is re-estimated.
2. **Display strings** are formatted once per language (cane line, date line, fire count,
   alert codes and messages). Both renderers draw only from them.
3. **Renderers:** a deterministic template, always available; and the LLM, whose prompt
   asks for a 55-word budget, fixes the order of the content and forbids facts outside the
   input, advice, and confirmation, causal or speculative wording.
4. **Verifier** on every text, template included: the area, cane line, date line and every
   alert code must appear verbatim; no date, time, number or upper-case code absent from the
   display strings may appear; a list of forbidden phrases ("confirmed", "because",
   "probably", ...) is rejected; at most 70 words.
5. **Fallback** to the template, with the reason recorded, when there is no API key, the
   call fails, the stop reason is `refusal` or `max_tokens`, the JSON is malformed, or
   either language fails the verifier.

**Metrics** (58 polygons, live calls, `results/6/alert_counts.json`):

| | value |
|---|---|
| template texts passing the verifier | 116/116 |
| LLM, first prompt | 41/58 polygons passed (17 over 70 words) |
| LLM, revised prompt | **57/58 (98.3 %)**; the other hit the token cap and fell back |
| LLM text length, revised prompt | median 23 words, max 70; 4 of 114 over the 55-word budget |

**Limits.**
- The verifier is lexical: it catches invented or altered numbers, dates and codes, not an
  unsupported claim made without a number. The sample is read by hand.
- The summaries are generated text, labelled as such. A burn alert means a VIIRS detection
  near the polygon at that time; it is not legal evidence.
- Reference fields and segments come from different outputs, so p36's summary differs from
  that of any segment inside it.

---

## Rejected: field segmentation with SAM

**What was tried.** Segment Anything (Kirillov et al. 2023 [14]), a pretrained foundation
model from Meta AI, ViT-B checkpoint (`sam_vit_b_01ec64.pth`, 375 MB, Apache 2.0), through
samgeo 1.4.2 [15], on CPU, to draw field polygons over the Morelos AOI from 10 m
Sentinel-2.

**Training data.** None here: used as-is, automatic mask generation (48 × 48 point grid,
predicted IoU ≥ 0.80, stability ≥ 0.85), 72 overlapping tiles per input.

**Size.** ViT-B; 31 min (true colour) and 30 min (NDVI history) on CPU.

**Metrics** (IoU against the 38 hand-traced fields, `results/4d/segmentation.md`):

| input | median IoU | IoU ≥ 0.5 |
|---|---|---|
| SAM, 2026-01-10 true colour | 0.10 | 9/38 |
| SAM, NDVI median / std / min | 0.02 | 2/38 |
| Felzenszwalb (classical, delivered) | 0.58 | 24/38 |

**Why rejected.** SAM merges contiguous cane fields of one colour into blocks; it does well
only on fields that contrast with their surroundings (pasture fields, IoU 0.54-0.83). The
delivered polygons come from Felzenszwalb, a deterministic algorithm. `segments.run_sam`
stays in the code. At 10 m a cane field is 15-20 pixels wide; this result says nothing
about SAM on sub-metre imagery.
