# Zafra Radar

Zafra Radar detects, field by field, when sugarcane is harvested and whether it was burned
before harvest. It combines NISAR L-band SAR backscatter (L2 GCOV) with Sentinel-1, NASA FIRMS
active fires and Sentinel-2. The harvest detector ships in two versions, a rule-based baseline
and a trained classifier, and the project reports results for both. The demo site is the
sugarcane belt around Zacatepec / Jojutla, Morelos, Mexico.

## Challenge

Solo entry for the **NASA Space Apps Challenge 2026** (Nov 14-15), challenge
**"Dancing with the SARs"**. Before the event the author works through the learning
exercises below; the product itself is built during the hackathon.

Demo acquisition geometry: NISAR track 19, frame 11, ascending, HH+HV, 40+5 MHz.

## Setup

Requires Python 3.11+. No system GDAL/PROJ needed: the `rasterio`, `pyproj` and `pyogrio`
wheels bundle them. On macOS, `xgboost` needs the OpenMP runtime: `brew install libomp`.

**With `uv` (preferred, if installed):**

```bash
make setup            # detects uv and uses it
# or manually:
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

**With `venv` + `pip`:**

```bash
make setup            # falls back to python3.12/3.11 -m venv when uv is absent
# or manually:
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Then add credentials:

```bash
cp .env.example .env  # fill in EARTHDATA_USERNAME, EARTHDATA_PASSWORD, FIRMS_MAP_KEY
```

- Earthdata Login: https://urs.earthdata.nasa.gov (NISAR downloads from ASF)
- FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
- Earth Engine: run `.venv/bin/earthengine authenticate` once; project id goes in `EE_PROJECT`

Other targets: `make lint`, `make test`.

## Downloading data

**NISAR GCOV** (~7 GB per scene):

```bash
.venv/bin/python scripts/00_download_nisar.py                       # list matching scenes only
.venv/bin/python scripts/00_download_nisar.py --download \
    --dates 2026-03-11,2026-05-10                                   # fetch two specific scenes
.venv/bin/python scripts/00_download_nisar.py --download --limit 3  # earliest 3
```

Options: `--start`, `--end` (defaults 2025-11-15 .. 2026-05-21). Files already present with
the correct size are skipped. Equivalent: `make download-nisar ARGS="--download --limit 1"`.

**FIRMS VIIRS active fires** (SNPP, NOAA-20, NOAA-21):

```bash
.venv/bin/python scripts/00_download_firms.py                       # 2024-11-01 .. 2026-05-31
.venv/bin/python scripts/00_download_firms.py --start 2025-11-01 --end 2026-05-31
```

Writes `data/firms/firms_viirs_{snpp,noaa20,noaa21}_<start>_<end>.csv`.

## Layout

```
src/zafra/        package: config.py (.env loading), paths.py (data dirs),
                  detect/ (harvest/burn detectors), models/ (ML training and inference)
scripts/          numbered, runnable scripts (00_ = data download)
data/             gitignored: nisar/, sentinel1/, firms/, fields/, labels/ (ground truth)
outputs/          gitignored: figures, tables
notebooks/        exploration
docs/             DATA_SOURCES.md, GLOSSARY.md, AI_COMPONENTS.md
```

## Learning exercises

- **1.** Inspect a GCOV scene
- **2.** Field backscatter
- **3.** Full-season time series
- **4.** Harvest detector (rule-based baseline)
- **4b.** Harvest detector (trained classifier: Random Forest / XGBoost on time-series
  features + GPM rain + FIRMS fire; DTW as alternative)
- **5.** Burn detection with FIRMS
- **6.** Validation
- **7.** Automatic field segmentation (Sentinel-2, SAM or lightweight model; stretch goal)

## In-product AI

| Component | Status | Exercise |
|---|---|---|
| Harvest classifier (Random Forest / XGBoost on per-field features) | Planned | 4b |
| Automatic field segmentation from Sentinel-2 | Stretch goal | 7 |
| Per-field natural-language summary via LLM, from a template | Built during the hackathon; cosmetic, not scientific | - |

Design notes: `docs/AI_COMPONENTS.md`. How AI is used to build the project versus inside the
product: `AI_USAGE.md`.

## Licence

Code: MIT (see `LICENSE`). Data licences and citations: `docs/DATA_SOURCES.md`.
AI assistance disclosure: `AI_USAGE.md`.
