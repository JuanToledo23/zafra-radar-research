# Data sources

One entry per source. "Access" says how this repository gets the data. Check the provider's
current terms before redistributing anything; the citation lines are starting points to
confirm against the provider pages.

---

## NISAR L2 GCOV (L-band)

- **What:** NASA-ISRO SAR mission, Level-2 Geocoded Polarimetric Covariance product.
  Radiometrically terrain-corrected backscatter (gamma-0) on a map grid, delivered as HDF5.
- **Resolution / cadence:** 12-day repeat per track. Demo scenes: track 19, frame 11,
  ascending, dual-pol HH+HV, frequency A 40 MHz + frequency B 5 MHz ("40+5"). About 7 GB
  per scene.
- **Access:** `scripts/00_download_nisar.py` uses `asf_search` (dataset `NISAR`,
  processingLevel `GCOV`, dataMaturity `PROVISIONAL`) and downloads from ASF with Earthdata
  Login credentials into `data/nisar/`.
- **Dataset paths inside the HDF5:**
  `/science/LSAR/GCOV/grids/frequencyA/HHHH` and `/science/LSAR/GCOV/grids/frequencyA/HVHV`
  (the diagonal covariance terms, i.e. HH and HV backscatter power, linear scale).
- **Licence / citation:** NASA data, open with no restrictions. Cite as: NASA/JPL-ISRO NISAR
  Project, *NISAR L2 Geocoded Polarimetric Covariance (GCOV), Provisional*, distributed by
  ASF DAAC, accessed <date>. (Confirm the DOI on the ASF collection page.)
- **Caveats:**
  - Maturity is **PROVISIONAL** (collection `NISAR_L2_GCOV_PROVISIONAL_V1`), so calibration
    and formats may still change. Products seen so far are **CRID P05023**.
  - The **standard observation series starts 2026-06-17**. Earlier acquisitions (including the
    2025-11 to 2026-05 demo scenes) are **supplemental frames**, so cadence and coverage before
    that date are not guaranteed.
  - **Permanent gap from 2026-07-27 to 2026-08-10**: there will never be data for this window.
  - The files are large: subset to the area of interest early and never commit `.h5`.
  - A second pass covers the site: **descending track 113, frame 79** (about 01:13 UTC,
    against about 12:05 UTC for ascending), same 40+5 HH+HV mode. Downloaded with
    `--direction DESCENDING` into `data/nisar/GCOV_DESCENDING_t113f79/`. Keep it as a
    separate series because the viewing geometry is different.

## NISAR L2 GUNW (L-band interferometric coherence)

- **What:** Level-2 Geocoded Unwrapped Interferogram. Each file is one **nearest-neighbour
  12-day pair** (reference date and secondary date) from the same track. It contains wrapped
  and unwrapped interferometric phase, **coherence magnitude**, connected components,
  ionosphere phase screen and pixel offsets, all on map grids. Co-pol only (**HH** for our
  track; VV in other modes), **frequency A** only (40 MHz). About 2.3 GB per pair.
- **Access:** `scripts/00_download_nisar.py --product GUNW --track 19 --frame 11`
  (processingLevel `GUNW`, dataMaturity `PROVISIONAL`, collection
  `NISAR_L2_GUNW_PROVISIONAL_V1`) into `data/nisar/GUNW_ASCENDING_t19f11/`. ASF matches on
  temporal overlap, so pairs that straddle the window edges are included (2025-11-11/11-23
  and 2026-05-10/05-22).
- **Coherence layer** (checked in the `20251111/20251123` file, CRID P05023):
  - `/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH/coherenceMagnitude`:
    **20 m** spacing, 16740 x 17064, float32, NaN fill. **This is the one we use**, because
    it resolves individual fields.
  - `/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude`:
    80 m (multilooked) version, 4185 x 4266.
  - Both are in UTM zone 14N, **EPSG:32614** (scalar `projection` in the same group). x and y
    come from `xCoordinates`/`yCoordinates`, and the spacing from
    `xCoordinateSpacing`/`yCoordinateSpacing` (y is negative, north-up). The valid-pixel mask
    is `mask` one level up.
- **Dataset paths read by `src/zafra/multisignal.py`** (`extract_coherence_series`):
  - coherence: `/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH/coherenceMagnitude`
  - grid: `.../wrappedInterferogram/xCoordinates` and `.../yCoordinates` (pixel centres; the
    `HH` subgroup holds identical copies), `.../wrappedInterferogram/projection` (EPSG:32614).
    Spacing is computed from the coordinates (20 m, -20 m) and the window is cut with
    `gcov.window_for_geometry`, the same as for GCOV. The coordinates carry no
    `pixel_coordinate_convention` attribute; their values (odd multiples of 10 m on a 20 m grid)
    are consistent with pixel centres.
  - QA only: `.../wrappedInterferogram/mask` (uint32; bits 0-7 = water flag and the reference and
    secondary subswath digits, 0 = invalid sample; bits 8-23 = RSLC anomaly flags; 255 = fill).
    Flagged pixels are counted (`mask_flagged_px`) but not removed. Only NaN pixels are dropped.
  - pair dates: `/science/LSAR/identification/referenceZeroDopplerStartTime` and
    `secondaryZeroDopplerStartTime`.
  - Field statistic: mean and median coherence of the pixels whose **centre** lies inside the
    polygon, with NaN excluded.
- **Use in this project:** **coherence only**. Low coherence across a pair is evidence that
  the field changed (cut, burnt or trafficked) between the two dates. We do **no deformation
  processing**: no phase unwrapping checks, no ionosphere or troposphere correction, no
  time-series inversion.
- **Licence / citation:** NASA data, open with no restrictions. Cite as: NASA/JPL-ISRO NISAR
  Project, *NISAR L2 Geocoded Unwrapped Interferogram (GUNW), Provisional*, distributed by
  ASF DAAC, accessed <date>. (Confirm the DOI on the ASF collection page.)
- **Caveats:** PROVISIONAL maturity, the same supplemental-frame caveats as GCOV, and the
  same permanent July/August 2026 gap. Older BETA products (CRID X050xx) also exist and
  include a 24-day 2025-11-23/12-17 pair. They are not downloaded, because PROVISIONAL
  reprocessing covers every date. A missing acquisition breaks the 12-day chain, so check the
  date pairs before treating the series as regular.

## Sentinel-1 GRD (C-band) via Google Earth Engine

- **What:** ESA Copernicus C-band SAR, Ground Range Detected, already calibrated and
  terrain-corrected to sigma-0 in dB by Earth Engine. Collection `COPERNICUS/S1_GRD`.
- **Resolution / cadence:** 10 m pixels (IW mode), VV+VH. Revisit is 6-12 days depending on
  which satellites in the constellation are operating.
- **Access:** Earth Engine Python API (`earthengine-api`, `geemap`) with `EE_PROJECT`. Not
  downloaded yet; exports go to `data/sentinel1/`.
- **Licence / citation:** Copernicus Sentinel data, free and open. Credit:
  "Contains modified Copernicus Sentinel data [year]".
- **Caveats:** C-band reaches less far into the canopy than L-band, so it saturates sooner
  over tall cane. Earth Engine's GRD is sigma-0, not gamma-0, so do not mix it with NISAR
  values without normalising. Ascending and descending passes have different geometry; keep
  them as separate series.

## NASA FIRMS VIIRS active fires (375 m)

- **What:** Active-fire detections (thermal anomalies) from VIIRS on Suomi-NPP, NOAA-20 and
  NOAA-21, with fire radiative power (FRP) and confidence.
- **Resolution / cadence:** 375 m pixels. Each satellite overpasses roughly twice a day (day
  and night).
- **Access:** `scripts/00_download_firms.py` calls the FIRMS area API
  (`/api/area/csv/<key>/<SOURCE>/<bbox>/<1..5 days>/<date>`) with `FIRMS_MAP_KEY`. It uses
  Standard Processing (`*_SP`) where available and Near Real-Time (`*_NRT`) for recent days.
  Output: `data/firms/`.
- **Licence / citation:** NASA open data. Credit: "We acknowledge the use of data and imagery
  from NASA's Fire Information for Resource Management System (FIRMS)
  (https://earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information
  System (ESDIS)." Also cite the VIIRS 375 m active fire product (Schroeder et al., 2014).
- **Caveats:** A detection is a pixel centre, not a field outline. Small or short cane burns
  can be missed, especially under cloud or between overpasses. A 375 m pixel covers several
  fields. NRT records are later replaced by SP, so re-download before the final analysis.
  The API allows at most 5 days per request.

## Sentinel-2 L2A (optical)

- **What:** ESA Copernicus multispectral surface reflectance. Earth Engine collection
  `COPERNICUS/S2_SR_HARMONIZED`.
- **Resolution / cadence:** 10-20 m pixels, about 5-day revisit.
- **Access:** Earth Engine Python API. Used for visual checks (NDVI drop after harvest,
  dark burn scars) and to draw field boundaries into `data/fields/`.
- **Licence / citation:** "Contains modified Copernicus Sentinel data [year]".
- **Caveats:** Clouds are a problem in the rainy season, though the zafra (Nov-May) is mostly
  dry. Use the SCL band or Cloud Score+ to mask clouds. The harmonised collection removes the
  2022 processing-baseline offset.

## GPM IMERG (precipitation)

- **What:** NASA Global Precipitation Measurement, multi-satellite precipitation estimates.
  Earth Engine collection `NASA/GPM_L3/IMERG_V07`.
- **Resolution / cadence:** 0.1 degrees (about 11 km), every 30 minutes.
- **Access:** Earth Engine Python API, aggregated to daily totals over the demo area.
- **Licence / citation:** NASA open data. Huffman, G.J., et al., *GPM IMERG Final
  Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07*, GES DISC.
- **Caveats:** Used only to flag rain on or just before a SAR acquisition. Wet soil and wet
  canopy raise backscatter and can look like a harvest change. Its cells are far coarser than
  a field.

## CONADESUCA weekly harvest reports

- **What:** Mexico's Comité Nacional para el Desarrollo Sustentable de la Caña de Azúcar
  publishes weekly zafra progress reports per sugar mill (cane milled, area harvested, yield,
  KARBE). The relevant local mill is the Zacatepec mill.
- **Resolution / cadence:** Weekly, aggregated per mill (no per-field data).
- **Access:** Manual download from the CONADESUCA / Government of Mexico website. Not
  automated in this repo.
- **Licence / citation:** Government of Mexico public information. Cite the CONADESUCA report
  title and week.
- **Caveats:** Totals are per mill and cannot validate individual fields, only the shape of
  the season (start, peak, end). Report formats change between seasons.
