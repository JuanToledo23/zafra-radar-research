# Exercise 9: web raster sanity check (Morelos)

HV raster (`data/rasters/morelos/hv_<date>.tif`: 5x5 boxcar in linear power, EPSG:3857 at
the z14 pixel size, dB) against the exercise-4b field means
(`data/features/fields_v2_features.parquet`, ascending t19: mean of linear power over the
pixel centres inside the polygon on the native 10 m grid, then dB).

* `centroid_db`: the raster at the field centroid (one ~50 m boxcar footprint);
* `raster_poly_mean_db`: mean of the raster's linear power over the polygon (checks the
  reprojection and the dB/linear round trip; small fields lose edge pixels to the boxcar).

Fields: p36 (ground truth) and the two largest other cane fields, p16 (9.0 ha) and p10
(6.9 ha).

| field | date | field_mean_4b_db | centroid_db | centroid_minus_4b | raster_poly_mean_db | poly_minus_4b |
|---|---|---|---|---|---|---|
| p36 | 2025-11-23 | -14.93 | -12.79 | +2.14 | -14.92 | +0.02 |
| p36 | 2025-12-29 | -13.91 | -11.06 | +2.85 | -13.90 | +0.01 |
| p36 | 2026-01-22 | -14.20 | -11.44 | +2.76 | -14.18 | +0.02 |
| p36 | 2026-02-15 | -15.94 | -16.10 | -0.16 | -15.90 | +0.04 |
| p36 | 2026-03-11 | -17.33 | -18.31 | -0.97 | -17.27 | +0.06 |
| p36 | 2026-04-28 | -17.70 | -19.20 | -1.51 | -17.61 | +0.09 |
| p16 | 2025-11-23 | -14.84 | -14.29 | +0.55 | -14.85 | -0.01 |
| p16 | 2025-12-29 | -16.03 | -16.01 | +0.02 | -16.02 | +0.01 |
| p16 | 2026-01-22 | -17.73 | -16.93 | +0.80 | -17.75 | -0.02 |
| p16 | 2026-02-15 | -18.53 | -19.93 | -1.39 | -18.42 | +0.12 |
| p16 | 2026-03-11 | -19.18 | -19.79 | -0.61 | -19.11 | +0.08 |
| p16 | 2026-04-28 | -17.67 | -18.73 | -1.06 | -17.65 | +0.02 |
| p10 | 2025-11-23 | -14.30 | -13.53 | +0.77 | -14.25 | +0.06 |
| p10 | 2025-12-29 | -14.58 | -13.46 | +1.12 | -14.58 | -0.00 |
| p10 | 2026-01-22 | -12.72 | -11.36 | +1.36 | -12.83 | -0.11 |
| p10 | 2026-02-15 | -16.85 | -15.13 | +1.71 | -16.77 | +0.07 |
| p10 | 2026-03-11 | -17.59 | -17.71 | -0.12 | -17.53 | +0.06 |
| p10 | 2026-04-28 | -16.44 | -16.98 | -0.54 | -16.45 | -0.01 |

* Centroid vs 4b: |diff| median 1.02 dB, max 2.85 dB;
  9 of 18 within 1 dB.
* Polygon mean vs 4b: |diff| median 0.03 dB, max 0.12 dB;
  18 of 18 within 1 dB.

The polygon means reproduce the 4b values, so georeferencing, the mask, the boxcar and the
linear/dB handling are right. The centroid samples miss the ~1 dB target on half the
field-dates. That comes from texture inside the fields, not from the raster: one 50 m
footprint is not the field mean. p36 before the cut is patchy (its centroid is +2-3 dB
above the field mean; see the figure). A web popup should show the field mean (API /
series), not the pixel under the cursor.

`results/9/p36_hv_before_after.png`: HV around p36 on 2026-01-22 (last selected date before the 4b xgb window
2026-02-10 .. 02-15) and 2026-03-11 (first after the 2026-02-26 ground truth); yellow = p36.
