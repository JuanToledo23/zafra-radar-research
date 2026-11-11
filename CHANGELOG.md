# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] – 2026-11-12

First public release of the research pipeline, built before NASA Space Apps 2026. The
hackathon application is a separate repository.

### Added

- **Exercise 1** (scripts 01-03): inspect a NISAR GCOV file, read one field's HH / HV, and
  compare two dates.
- **Exercise 3**: full-season NISAR HV / HH time series per field.
- **Exercise 4**: rule baseline and intensity-ratio harvest test.
- **Exercise 5** (fires): VIIRS active fires per field (FIRMS).
- **Exercise 6** (optical): radar harvest windows checked against Sentinel-2 NDVI; NDVI
  labels for the 38 reference fields; p36 ground-truth check and harvest-front test.
- **Exercise 4b**: multi-signal features (GCOV ascending / descending, GUNW coherence) and
  the trained xgboost harvest detector.
- **Exercise 4c**: cane / non-cane gate (optical B, radar A) and the gated detector.
- **Exercise 4d**: Felzenszwalb field segmentation of the AOI (SAM tried and rejected) and
  the area-wide harvest map.
- **Exercise 4e**: 3-look baseline rule, cane-only fire check, fragment merging, slim
  GeoJSON.
- **Exercise 5**: site configuration (`configs/<site>.yaml`, `--site`) and transfer of the
  unchanged Morelos models to Ribeirão Preto, Brazil, validated against MapBiomas and an
  NDVI rule.
- **Exercise 6**: per-field facts, six deterministic alert rules, and Spanish / English
  summaries (optional LLM under a verifier, template fallback).
- **Exercise 7**: demo skeleton in `product/` (compose, empty PostGIS schema, FastAPI
  `/health`, MapLibre map shell).
- **Exercise 9**: NISAR backscatter and change COGs and XYZ tiles for the web map;
  scene-quality check with per-site scene exclusion; Sentinel-1 2025-26 series and the
  L- vs C-band drop at harvest.
- NISAR downloads with retries, resume, size check and lock.
- Packaging (`zafra-radar-research`, extras `download`, `ee`, `llm`, `sam`, `dev`), CI
  (ruff + pytest on Python 3.11 / 3.12), citation metadata and community files.

### Changed

- Brazil: the corrupt ascending scene 2026-07-18 is excluded from every radar table, and
  the delivered gate is now the optical gate B (both sites use B by default; the radar
  gate A is kept as radar evidence).
- The demo API is published on host port 8010 and the web app on 5174.

[0.1.0]: https://github.com/OWNER/zafra-radar-research/releases/tag/v0.1.0
