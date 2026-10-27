# Exercise 5: status (2026-10-27, ~12:30)

**Complete and committed** on branch `features/multisignal`, approved by the author. The reading is in `transfer.md`.

## Done

| step | evidence (mtime 10-27 unless noted) |
|---|---|
| site choice, configs, `--site` refactor, Morelos byte-identity | 10-26; docs/METHODS.md "Site configuration" |
| FIRMS, S2 scene/composite/NDVI stack, WorldCover, segmentation (32 277) | 10-26 21:45–22:20 |
| MapBiomas 2024 raster (Collection 10) on the S2 grid | `data/s2/brazil/mapbiomas_2024.tif` 09:27 |
| NISAR **26/26**, sizes = ASF, all open in h5py | last file 11:33 |
| first run, gate B delivered | superseded; its figures are kept in `transfer.md` as "gate B / first run" |
| `gate_variant` per site (morelos optical, brazil radar), `p_cane_optical` alongside, cross-check against the non-delivered gate | `configs/*.yaml`, `zafra.site`, `cane_gate.classify_fields`, script 14 |
| `date_uncertainty_days` (half the last-standing → first-harvested gap) in every harvest GeoJSON, merged and slim layers | scripts 14, 15 |
| Brazil re-run with gate A: scripts 14 → 15 → 16 | `data/results/brazil/*.geojson`, `results/5_brazil/*` 12:12+ |
| script 16: both gates, B with local labels (5-fold CV by 3 km blocks), uncertainty table | `validation.md` |
| Morelos re-run (scripts 14, 15) for the new columns | `results/4e/weekly.csv` and PNGs byte-identical to HEAD; `area_summary.json` restored to HEAD (only runtimes / check differ); slim 8.2 → 8.5 MB (`merge.md`, `merge_summary.json`) |
| docs: VALIDATION (section rewritten, uncertainty for the frontend), METHODS (season table, columns, PostGIS `ST_AsMVT` serving), DATA_SOURCES (UNICA, CONAB, MapBiomas, OCSB note), AI_USAGE | |
| ruff clean, pytest 60 passed | |

## Open

1. The UNICA figure (490.93 Mt to 1 Oct 2025) was supplied by the author and not
   re-checked.

Resolved: the 41.8 MB Brazil slim GeoJSON (gate A, 21 171 cane polygons) is accepted
by the author. The product serves it as PostGIS vector tiles.

## Runtimes

| step | s |
|---|---|
| NISAR desc remainder download (20.6 GB) | ~1 060 |
| optical gate preview (scratch) | 230 |
| Brazil script 14, first run (radar zonal 34, features + gate 268, detector 48) | 391 |
| Brazil script 14, gate A re-run (features cached) | 130 |
| Brazil script 15 / 16 | 33 / 20 |
| Morelos script 14 / 15 re-run | 75 / 13 |
