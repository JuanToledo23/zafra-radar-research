# Zafra Radar

Field-level sugarcane harvest dates from NISAR L-band radar, with the uncertainty of every
date, checked against satellite fire detections. Solo entry for the **NASA Space Apps
Challenge 2026** (Nov 14-15), challenge **"Dancing with the SARs"**. The processing
pipeline below was built and validated before the event; the product (map, API, alerts)
exists as a skeleton and is built during the hackathon.

## The problem

In the sugarcane belt of Morelos, Mexico, the harvest season (*zafra*) runs from November
to May, and cane is usually burned before it is cut. There is no field-level record of when
each field was harvested or whether fire was involved (the public harvest reports are per
mill [27], and harvest news is per region [34]), and field visits do not scale. Optical satellites lose fields to cloud and
smoke and see a drying canopy before the cut. Satellite fire detections (VIIRS) are coarse
and brief: within 200 m and the week before, they catch only 8.7 % of the harvests dated
here ([VALIDATION](docs/VALIDATION.md#known-limitations)).

## What Zafra Radar does

It splits an area into candidate field polygons, decides which polygons are sugarcane, and
dates the harvest of each cane polygon from NISAR L-band backscatter and interferometric
coherence, ascending and descending passes together. Every date carries its uncertainty
(half the gap between the last standing and the first harvested look) and a confidence
tier. Deterministic alert rules flag fires outside permitted hours, fires with no harvest,
and polygons where radar and optical disagree, and a short summary per polygon, in Spanish
and English, states the results without adding to them. The demo site is the supply area
of the Ingenio Emiliano Zapata (Zacatepec / Jojutla, Morelos); the same pipeline, with the
Morelos models unchanged, was run on Ribeirão Preto, Brazil.

## Results at a glance

| site | measure | result | checked against | source |
|---|---|---|---|---|
| Morelos | harvest date error (xgboost, out of fold) | median 3.5 d; 24 of 26 fields within ±12 d (rule: 19.2 d; ratio test, both orbits: 11.5 d) | Sentinel-2 NDVI harvest windows of 26 cane fields | [`results/4b`](results/4b/comparison.md) |
| Morelos | cane gate accuracy (leave-one-field-out) | 37/38 optical (partly circular); 34/38 radar only | NDVI labels of 38 fields | [`results/4c`](results/4c/comparison.md) |
| Morelos | area dated, zafra 2025-26 | 3 976 of 6 014 cane segments; 6 867 of 9 943 ha | – | [`results/4e`](results/4e/area_summary.json) |
| Morelos | weekly harvests vs VIIRS fires within 200 m of cane | r = 0.71 (same week) | VIIRS active fires | [`results/4e`](results/4e/area_summary.json) |
| Brazil | cane gate (radar, unchanged Morelos model) | precision 72.0 %, recall 63.7 % by area | MapBiomas 2024 land cover | [`results/5_brazil`](results/5_brazil/validation.md) |
| Brazil | harvest date error (unchanged Morelos detector) | median 5.0 d; 30 of 40 sampled segments within ±12 d | NDVI rule on the local Sentinel-2 stack | [`results/5_brazil`](results/5_brazil/validation.md) |
| Brazil | area dated, safra 2026-27 (radar from 2026-06-22) | 4 780 of 21 171 cane segments; 16 590 of 57 044 ha | – | [`results/5_brazil`](results/5_brazil/transfer.md) |

None of these is accuracy against field visits: the references are optical or land-cover
maps (see [Validation](#validation)).

![Morelos 2025-26 harvest map](docs/figures/4e_harvest_map.png)

*Estimated harvest date per cane segment, Morelos zafra 2025-26 (darker = later). Outlined
segments are cane by the gate but trees or grassland by ESA WorldCover.*

## How it works

1. **Data.** NISAR L2 GCOV backscatter (HH, HV) and L2 GUNW coherence, Sentinel-2,
   VIIRS fires, ESA WorldCover, per site from `configs/<site>.yaml`
   ([METHODS 1](docs/METHODS.md#1-data-and-sites)).
2. **Field polygons.** Felzenszwalb segmentation of the Sentinel-2 NDVI history
   (39 542 segments on Morelos); SAM was tried and rejected
   ([METHODS 2](docs/METHODS.md#2-field-polygons-reference-fields-and-segmentation)).
3. **Cane gate.** A logistic regression on NDVI phenology (or on radar statistics) gives
   P(cane) per polygon ([METHODS 3](docs/METHODS.md#3-cane-gate)).
4. **Harvest detector.** xgboost on per-look radar features (HV change, coherence) dates the
   cut on gated cane polygons ([METHODS 4-5](docs/METHODS.md#4-radar-series-and-features)).
5. **Baseline rule and flags.** A date is kept only after 3 standing looks in its orbit;
   status, confidence and the WorldCover flag are added
   ([METHODS 6](docs/METHODS.md#6-baseline-rule-status-and-flags)).
6. **Alerts, summaries, product.** Six deterministic alert rules, a verified per-polygon
   summary, and a PostGIS + FastAPI + MapLibre product
   ([METHODS 7-9](docs/METHODS.md#7-alert-rules)).

## Validation

The harvest detector beats the classic radar change detectors on the 26 labelled fields
(3.5 d vs 11.5-19.2 d median error) but fires on almost every non-cane field, which is why
the cane gate runs first (non-cane false alarms 11/12 → 1/12). Area-wide, the weekly
harvest counts and the weekly fires near cane rise and fall together (r = 0.71). The one
grower report for 2025-26 (field p36) disagrees with every satellite signal by 12-17 days;
the 2024-25 report agrees. Full tables, figures and the p36 ground-truth
check: [docs/VALIDATION.md](docs/VALIDATION.md).

## Limitations

One season, 26 labelled cane fields and one region; references are optical, not field
visits, and there is no untouched test set. The delivered Morelos gate is partly circular,
and the radar-only gate drops the one grower-reported field. Coverage sets hard limits: no
Morelos date before Dec 20, nothing before ~Jul 18 in Brazil, and NISAR data are
provisional. Segments are fragments of fields, not fields. Numbered list with figures:
[docs/VALIDATION.md, Known limitations](docs/VALIDATION.md#known-limitations).

## Data sources

NISAR L2 GCOV and GUNW (NASA/JPL-ISRO, ASF DAAC, provisional), Sentinel-2 L2A and
Sentinel-1 GRD (ESA Copernicus, via Earth Engine), NASA FIRMS VIIRS active fires, ESA
WorldCover 2021, GPM IMERG, MapBiomas Collection 10. Resolution, dates, access and licences:
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## AI usage

Two in-product models are trained here (the cane gate and the harvest detector), and an LLM
writes the per-polygon summaries under a deterministic verifier with a template fallback;
SAM was tried for segmentation and rejected. The code and documents were written with an AI
coding assistant from the author's briefs, and every change was reviewed by the author.
Details and the decisions log: [AI_USAGE.md](AI_USAGE.md); component design:
[docs/AI_COMPONENTS.md](docs/AI_COMPONENTS.md).

## Repository layout

```
configs/          one YAML per site (morelos, brazil)
src/zafra/        library: NISAR readers, features, detect/ (gate, detector, reference
                  detectors), segments, zonal statistics, alerts, summaries
scripts/          numbered pipeline scripts, 00 (downloads) to 17 (summaries)
tests/            pytest suite
results/          committed reports per exercise (4b, 4c, 4d, 4e, 5_brazil, 6)
docs/             METHODS, VALIDATION, DATA_SOURCES, AI_COMPONENTS, GLOSSARY, REFERENCES, figures/
product/          product skeleton: PostGIS, FastAPI, React + MapLibre (docker compose)
data/, outputs/   git-ignored inputs and intermediate outputs (except the reference fields)
```

## How to run

**Setup.** Python 3.11+. `rasterio`, `pyproj` and `pyogrio` wheels bundle GDAL/PROJ. On
macOS, `xgboost` needs the OpenMP runtime (`brew install libomp`).

```bash
make setup            # uses uv if installed, otherwise python3.12/3.11 -m venv + pip
cp .env.example .env  # EARTHDATA_USERNAME, EARTHDATA_PASSWORD, FIRMS_MAP_KEY, EE_PROJECT,
                      # optional ANTHROPIC_API_KEY (summaries fall back to the template)
.venv/bin/earthengine authenticate
make lint && make test
```

Credentials: Earthdata Login https://urs.earthdata.nasa.gov (NISAR from ASF); FIRMS MAP_KEY
https://firms.modaps.eosdis.nasa.gov/api/map_key/.

**Pipeline** (Morelos is the default site; add `--site brazil` for the second site).

```bash
P=.venv/bin/python
$P scripts/00_download_nisar.py --download            # GCOV ascending (~7 GB per scene)
$P scripts/00_download_nisar.py --product GUNW --track 19 --frame 11 --download
$P scripts/00_download_nisar.py --direction DESCENDING --track 113 --frame 79 --download
$P scripts/00_download_firms.py
# reference fields (Morelos only): fires, NDVI labels, radar series, detector, gate
$P scripts/05_fires_per_field.py
$P scripts/07_label_fields.py
$P scripts/08_multisignal.py
$P scripts/10_harvest_classifier.py   # -> results/4b
$P scripts/11_cane_gate.py            # -> results/4c
# area-wide, any site
$P scripts/12_s2_images.py            # Sentinel-2 and WorldCover over the AOI
$P scripts/13_field_segmentation.py   # -> results/4d
$P scripts/14_area_harvest_map.py     # gate + detector + baseline rule -> results/4e
$P scripts/15_merge_fragments.py      # merged reference layer, slim GeoJSON
$P scripts/16_transfer_validation.py --site brazil   # -> results/5_brazil
$P scripts/17_field_summaries.py      # alerts and summaries -> results/6
```

Each script's options are in its docstring (`--help`).

**Product skeleton.**

```bash
cd product
docker compose up --build    # web http://localhost:5173, api http://localhost:8000/health
```

It shows a bundled 50-polygon sample; data loading and the real endpoints are planned in
[product/HACKATHON_PLAN.md](product/HACKATHON_PLAN.md). Offline mode and timings:
[product/README.md](product/README.md).

## References

Full numbered list with URLs or DOIs: [docs/REFERENCES.md](docs/REFERENCES.md). Key
sources: NISAR GCOV specification and ASF documentation [16, 17], NASA FIRMS [18], speckle
and ratio-test statistics [10, 11], Felzenszwalb segmentation [13], Segment Anything [14],
prior work on satellite monitoring of sugarcane [1-9].

## License

Code: MIT, see [LICENSE](LICENSE). Data licences and citations:
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
