# Data sources

One entry per source, all with the same fields. "Access" is how this repository gets the
data; everything downloaded goes under `data/` (git-ignored). Check the provider's current
terms before redistributing anything. Citations [n] are in [REFERENCES.md](REFERENCES.md).

---

## NISAR L2 GCOV (L-band backscatter)

- **Product:** NISAR Level-2 Geocoded Polarimetric Covariance (GCOV), PROVISIONAL
  (collection `NISAR_L2_GCOV_PROVISIONAL_V1`, CRID P05023) [16, 17]. Radiometrically
  terrain-corrected gamma-0 on a map grid, HDF5. Datasets used:
  `/science/LSAR/GCOV/grids/frequencyA/HHHH` and `.../HVHV` (linear power).
- **Provider:** NASA/JPL-ISRO NISAR Project, distributed by ASF DAAC.
- **Resolution:** 10 m grid; 12-day repeat per track; dual-pol HH + HV, 40 + 5 MHz. About 7 GB
  per scene.
- **Dates used:**
  - Morelos ascending t19 f11 (~12:05 UTC): 15 scenes, 2025-11-23 .. 2026-05-10.
  - Morelos descending t113 f79 (~01:13 UTC): 15 scenes, 2025-11-18 .. 2026-05-17 (one
    missing scene around 03-30, a 24-day gap).
  - Ribeirão Preto ascending t146 f167 (~08:59 UTC): 10 scenes, 2026-06-24 .. 2026-10-22,
    UTM 23S grid. Descending t125 f99 (~21:23 UTC): 8 scenes, 2026-06-22 .. 2026-10-20, UTM
    22S grid (reprojected). Brazil total with GUNW: 26 files, 142.7 GB.
- **Access:** `scripts/00_download_nisar.py [--site <site>] [--product GCOV|GUNW]
  [--direction ...] [--track N --frame N] --download` (`asf_search`, dataset `NISAR`,
  dataMaturity `PROVISIONAL`), Earthdata Login credentials in `.env`.
- **Licence / terms:** NASA open data, no restrictions. Cite as NASA/JPL-ISRO NISAR Project,
  *NISAR L2 GCOV, Provisional*, ASF DAAC, accessed <date> (DOI to confirm on the ASF
  collection page).
- **Known issues** (provisional products, see the ASF NISAR documentation [17]):
  - calibration and formats may change between releases;
  - the standard observation series starts 2026-06-17; earlier acquisitions, including all
    Morelos scenes, are supplemental frames with no guaranteed cadence;
  - permanent gap 2026-07-27 .. 2026-08-10 (Brazil: ascending 07-30, descending 07-28 and
    08-09 fall in it; descending 09-14 is also missing);
  - no NISAR L2 product covers any Thai cane region in 2025-11 .. 2026-05 (searched
    2026-10-26), which is why the second site is in Brazil; nothing over Ribeirão Preto
    before 2026-06-22.

## NISAR L2 GUNW (L-band interferometric coherence)

- **Product:** NISAR Level-2 Geocoded Unwrapped Interferogram (GUNW), PROVISIONAL
  (`NISAR_L2_GUNW_PROVISIONAL_V1`, CRID P05023) [17]. Only the coherence magnitude is used:
  `/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH/coherenceMagnitude`. No
  phase or deformation processing.
- **Provider:** NASA/JPL-ISRO NISAR Project, ASF DAAC.
- **Resolution:** 20 m grid (the 80 m unwrapped-grid version is not used); one file per
  nearest-neighbour 12-day pair; HH, frequency A. About 2.3 GB per pair.
- **Dates used:** Morelos ascending t19 f11, 16 pairs from 2025-11-11/11-23 to
  2026-05-10/05-22. Ribeirão Preto ascending t146 f167, 8 pairs (7 at 12 days, 1 at 24).
- **Access:** `scripts/00_download_nisar.py --product GUNW ...`; read by
  `src/zafra/multisignal.py` (`extract_coherence_series`). Grid from `xCoordinates` /
  `yCoordinates` (pixel centres) and `projection`; NaN pixels dropped, mask flags counted.
- **Licence / terms:** as GCOV.
- **Known issues:** as GCOV. Older BETA products (CRID X050xx) exist and are not used. A
  missing acquisition breaks the 12-day chain.

## Sentinel-2 L2A (optical)

- **Product:** Copernicus Sentinel-2 MSI Level-2A surface reflectance, Earth Engine
  collection `COPERNICUS/S2_SR_HARMONIZED` [20, 28]. Bands B2, B3, B4, B8 (NDVI), B12 (NBR,
  p36 check only), SCL for per-pixel masking (classes 4-7 kept).
- **Provider:** ESA / European Commission, Copernicus; through Google Earth Engine.
- **Resolution:** 10 m (SCL 20 m); ~5-day revisit.
- **Dates used:** Morelos per-field series 2024-10-12 .. 2026-06-29 (labels, script 07, and
  the p36 check); area-wide crop year 2025-06-01 .. 2026-05-31 (clearest scene 2026-01-10, 64-78
  clear looks, 77-date NDVI stack). Ribeirão Preto 2025-12-01 .. 2026-10-20.
- **Access:** Earth Engine Python API with `EE_PROJECT`; per-field series in
  `src/zafra/optical.py`, area-wide tiled export in `scripts/12_s2_images.py [--site]` into
  `data/s2/`.
- **Licence / terms:** free and open Copernicus data. Credit: "Contains modified Copernicus
  Sentinel data [year]".

## Sentinel-1 GRD (C-band)

- **Product:** Copernicus Sentinel-1 IW GRD, sigma-0 in dB, Earth Engine collection
  `COPERNICUS/S1_GRD` [21]. VH only.
- **Provider:** ESA / European Commission, Copernicus; through Google Earth Engine.
- **Resolution:** 10 m; 6-12-day revisit.
- **Dates used:** p36 ground-truth check of the 2024-25 harvest only (ascending t78, 27
  scenes; descending t143, 17 scenes), script 09. Not used by the pipeline.
- **Access:** Earth Engine Python API.
- **Licence / terms:** as Sentinel-2.

## NASA FIRMS VIIRS active fires

- **Product:** VIIRS 375 m active fire detections from Suomi-NPP, NOAA-20 and NOAA-21, with
  fire radiative power and confidence [18, 19]. Standard Processing (`*_SP`) where available,
  Near Real-Time (`*_NRT`) after.
- **Provider:** NASA FIRMS (ESDIS).
- **Resolution:** 375 m pixels; each satellite passes about twice a day (Morelos ~01:30 and
  ~13:30 local).
- **Dates used:** Morelos 2024-11-01 .. 2026-05-31 (1 097 detections in the AOI in the
  2025-26 zafra). Ribeirão Preto 2026-04-01 .. 2026-10-25 (17 detections).
- **Access:** `scripts/00_download_firms.py [--site <site>]`, FIRMS area API with
  `FIRMS_MAP_KEY` (at most 5 days per request), into `data/firms/`.
- **Licence / terms:** NASA open data. Credit: "We acknowledge the use of data and imagery
  from NASA's Fire Information for Resource Management System (FIRMS)
  (https://earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information
  System (ESDIS)."

## ESA WorldCover 2021

- **Product:** ESA WorldCover 10 m 2021 v200, Earth Engine collection `ESA/WorldCover/v200`,
  band `Map` [23].
- **Provider:** ESA.
- **Resolution:** 10 m; one map for 2021. Stated overall accuracy ~77 %; tall cane is often
  mapped as tree cover or cropland.
- **Dates used:** 2021 map, both sites.
- **Access:** Earth Engine (`optical.worldcover_image`), exported with the Sentinel-2 rasters.
  Used to drop built-up and water segments and to flag cane calls on trees or grassland;
  never a model input.
- **Licence / terms:** CC BY 4.0. Zanaga, D. et al. (2022), doi:10.5281/zenodo.7254221.

## GPM IMERG precipitation

- **Product:** GPM IMERG Final, half-hourly, V07, Earth Engine collection
  `NASA/GPM_L3/IMERG_V07` [22].
- **Provider:** NASA GES DISC.
- **Resolution:** 0.1° (~11 km), 30 min.
- **Dates used:** Morelos 2025-11 .. 2026-05, 3-day totals before each acquisition at the 38
  reference-field centroids (`data/features/gpm_rain_3d.csv`). Computed but not a model
  input; shown as context in figures and in the reference-field summaries.
- **Access:** Earth Engine Python API.
- **Licence / terms:** NASA open data. Cite Huffman, G. J. et al., GPM IMERG Final
  Precipitation L3 Half Hourly 0.1° × 0.1° V07, GES DISC.

## MapBiomas Brazil, Collection 10

- **Product:** annual land use / land cover of Brazil, 30 m from Landsat, class 20 = sugar
  cane; Earth Engine asset
  `projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`,
  band `classification_2024` [24].
- **Provider:** MapBiomas Project.
- **Resolution:** 30 m, resampled to the 10 m S2 grid; majority class per segment.
- **Dates used:** 2024 map (one season before the test season), accessed 2026-10-26.
- **Access:** Earth Engine. Used to choose the Brazil AOI, to check the gate, and as the
  labels of the reference-only "B with local labels" row. Never an input to a delivered
  model.
- **Licence / terms:** CC BY 4.0.

## Reference fields and grower report

- **Product:** 38 hand-traced polygons (`data/fields/fields_v2.geojson`) with NDVI labels and
  harvest windows from script 07, and p36's grower report (burned and cut 2025-03-16 and
  2026-02-26, from a dated video, not yet checked).
- **Provider:** the author (tracing) and one grower (p36).
- **Resolution:** polygons, median 2.6 ha.
- **Dates used:** zafras 2024-25 and 2025-26.
- **Access:** versioned in the repository: `data/fields/fields_v2.geojson`,
  `data/labels/fields_v2_labels.csv`.
- **Licence / terms:** the author's polygons, labels and harvest windows, and everything in
  `results/`, are CC BY 4.0 (`data/fields/LICENSE`, `results/LICENSE`). Third-party layers
  and anything derived from them keep their upstream terms (entries above). The grower's
  report is used with permission. The grower is not named, and the field is identified only
  as a grower-reported field in Tlaquiltenango.

## Data licence of this repository

- Code: MIT ([LICENSE](../LICENSE)).
- Author-made data (reference polygons, labels, harvest windows) and committed results
  (reports, tables, figures): CC BY 4.0, Copyright (c) 2026 Juan Toledo.
- Third-party data (NISAR, Sentinel-1/2, FIRMS, WorldCover, IMERG, MapBiomas) are not
  redistributed here except as derived per-field statistics and figures. They keep their
  upstream terms, with the credit lines given in each entry above.

## Sugar-sector statistics (context only)

None of these is compared with a pipeline output yet.

- **CONADESUCA weekly zafra reports** [27]: cane milled, area harvested, yield and KARBE per
  Mexican mill, weekly; manual download. CONADESUCA is the sector body set up under the
  LDSCA [30]. TODO (author): the harvested area of the Ingenio Emiliano Zapata for zafra
  2025-26, to compare with the 9 943 ha gated as cane.
- **UNICA** [25]: fortnightly Centre-South crushing reports. Cumulative Centre-South crush to
  1 Oct 2025 for safra 2025-26: 490.93 Mt (figure supplied by the author, not re-checked).
- **CONAB** [26]: safra surveys of cane area and production per state. Listed only.
- **Brazil fire policy:** São Paulo phased out pre-harvest cane burning (state law
  11.241/2002 and the 2007 agro-environmental protocol) [33], so fire is not a harvest marker
  at the Brazil site.
