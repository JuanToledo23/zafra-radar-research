# Glossary

Short, plain-language definitions for developers new to remote sensing.

**SAR (Synthetic Aperture Radar)**
A satellite radar that sends its own microwave pulses and records the echoes. It sees through
clouds and works at night. Moving the antenna along the orbit "synthesises" a much longer
antenna, which gives fine resolution.

**Backscatter**
The share of the radar energy that bounces back to the satellite from a pixel. Rough
surfaces and tall, structured vegetation (like standing cane) send back more than smooth bare
soil or calm water.

**Sigma-0 / gamma-0 (σ⁰ / γ⁰)**
Two ways to normalise backscatter so pixels can be compared. Sigma-0 divides by the area on
the ground. Gamma-0 divides by the area facing the radar, which removes much of the
brightness change caused by slopes and viewing angle. NISAR GCOV is gamma-0 with terrain
correction; Sentinel-1 GRD in Earth Engine is sigma-0.

**dB (decibel)**
A log scale for backscatter: `dB = 10 * log10(linear)`. GCOV stores linear power, so convert
to dB for plots, but average in linear units first and convert after.

**Polarization (HH, HV, VV, VH)**
The orientation of the radar wave, horizontal (H) or vertical (V), when sent and when
received. HH means sent H and received H. NISAR here records HH and HV; Sentinel-1 records
VV and VH.

**Cross-pol**
Sent and received polarizations differ (HV or VH). Volume scattering inside vegetation
produces most of the cross-pol signal, so it tracks biomass well. When cane is cut, cross-pol
usually drops sharply.

**Speckle**
The grainy "salt-and-pepper" noise in every SAR image. It comes from many small echoes adding
and cancelling inside one pixel. Averaging pixels or dates reduces it.

**Looks**
The number of independent samples averaged to make a pixel value. More looks means less
speckle but coarser effective resolution. Averaging all pixels inside a field adds many looks.

**Incidence angle**
The angle between the radar beam and the vertical at the ground. Steeper (smaller) angles
usually give brighter returns. Compare a field only with itself from the same track, so its
angle stays the same.

**Ascending / descending**
The direction of the orbit: ascending passes go south-to-north, descending north-to-south.
They view the ground from opposite sides at different local times, so keep them as separate
time series.

**Track / frame**
A track (relative orbit) is a repeating ground path; a frame is a fixed slice along that path.
Track 19 / frame 11 always covers the same area, which makes a consistent time series.

**GCOV**
NISAR Level-2 Geocoded Polarimetric Covariance. A map-gridded, terrain-corrected product that
stores backscatter power (and polarization cross-terms) for each pixel, in HDF5. It is the
easiest NISAR product to use for land-cover change.

**Revisit**
How often a satellite images the same spot from the same track. NISAR repeats every 12 days.
With the ascending and descending passes interleaved, looks are 2-12 days apart, and a
harvest is dated to within half the gap between the last standing and the first harvested
look.

**FIRMS**
NASA's Fire Information for Resource Management System. It serves near real-time and archived
active-fire detections from MODIS and VIIRS as points with a date, time and confidence.

**FRP (Fire Radiative Power)**
The heat output of a detected fire, in megawatts. Larger or hotter fires have higher FRP;
cane burns are short and usually low to moderate FRP.

**Zafra**
The sugarcane harvest season in Mexico and other Spanish-speaking countries, roughly
November to May/June. It is also the work of cutting and milling the cane in that season.

**KARBE**
*Kilogramos de Azúcar Recuperable Base Estándar*: kilograms of standard-basis recoverable
sugar per tonne of cane. It measures cane quality and drives the price paid to Mexican cane
growers, and CONADESUCA reports it per mill.

## Machine learning terms

**Feature (ML)**
A number computed from the raw data and fed to a model, e.g. "HV drop in dB since the
previous date" or "rain in the last 48 h". Choosing good features usually matters more than
choosing the model.

**Classifier**
A model that assigns each input to a category (here: harvested / not harvested), usually
with a probability. Random Forest and XGBoost are classifiers built from many decision trees.

**Cross-validation**
Splitting the labelled data into several parts and, in turn, training on all but one and
testing on the one left out. Here, the split is by field, so a field's own dates never help
score that same field.

**DTW (Dynamic Time Warping)**
A way to measure how alike two time series are, even when one is shifted or stretched in
time. It can match a field's backscatter curve to a template "harvest" curve when harvests
happen on different dates.

**IoU (Intersection over Union)**
A score for how well a predicted shape matches a reference shape: overlapping area divided
by combined area. 1 is a perfect match, 0 means no overlap.

**Segmentation**
Splitting an image into regions, labelling every pixel with the object it belongs to. Here:
outlining individual fields in a satellite image.

**SAM (Segment Anything Model)**
A general-purpose pretrained image segmentation model from Meta. It outlines objects in any
image, automatically or from prompt points or boxes, but it was not trained specifically on
satellite imagery.

## Project and pipeline terms

One line each, alphabetical.

**Ablation**: re-running a model with some inputs removed or added back, to see which ones matter.

**AOI (area of interest)**: the rectangle processed for a site (`aoi.bbox` in `configs/<site>.yaml`).

**ASF DAAC**: the Alaska Satellite Facility archive, which distributes NISAR data.

**AUC**: area under the ROC curve; 1 means the model ranks every positive above every negative.

**Baseline (radar)**: a polygon's typical HV before harvest, the median of the first 3 looks of an orbit in the baseline months.

**Baseline rule (3-look)**: a harvest date is kept only after ≥ 3 standing looks in the candidate's own orbit.

**C-band / L-band**: radar wavelengths of ~5.6 cm (Sentinel-1) and ~24 cm (NISAR); L-band reaches deeper into the cane canopy.

**Cane gate**: the model that decides whether a polygon is sugarcane before the harvest detector runs.

**Coherence**: how similar the radar phase of a pixel is between two dates (0-1); a cut, burn or tillage makes it drop.

**Crop year**: the 12 months ending at the end of the season, used for NDVI phenology.

**CRID**: the NISAR processing version stamped on each product (here P05023).

**`dudoso`**: "doubtful"; a reference polygon that the NDVI labelling rule did not call cane or not cane.

**Earth Engine**: Google's platform used to read Sentinel-1/2, GPM, WorldCover and MapBiomas.

**ENL (equivalent number of looks)**: mean² / variance of a polygon's pixel intensities; how many independent looks the data behave like.

**EPSG / UTM**: codes for map projections; each site uses its UTM zone (Morelos EPSG:32614, Brazil EPSG:32723) so distances and areas are in metres.

**F test / ratio test**: a test of whether a polygon's mean backscatter changed between two dates beyond what speckle explains.

**Felzenszwalb segmentation**: a classical graph-based algorithm that groups similar neighbouring pixels into regions.

**Gate A / gate B**: the radar-only and optical-only variants of the cane gate.

**GroupKFold**: cross-validation where all rows of one field stay in the same fold.

**GUNW**: NISAR Level-2 Geocoded Unwrapped Interferogram; here only its coherence layer is used.

**Green harvest**: mechanised cutting without burning, leaving a straw "trash blanket" on the field (Brazil site).

**HDF5**: the file format of NISAR products.

**Ingenio**: a sugar mill; the Morelos AOI is the supply area of the Ingenio Emiliano Zapata (Zacatepec).

**Leave-one-field-out**: cross-validation that holds out one field at a time.

**LLM (large language model)**: a pretrained text model; here Claude Haiku 5.5 writes the field summaries.

**Logistic regression**: a linear model that outputs a probability.

**MapBiomas**: annual 30 m land-cover maps of Brazil with a sugar-cane class.

**NBR (normalised burn ratio)**: (B8 − B12) / (B8 + B12) from Sentinel-2; drops after burning.

**NDVI**: (NIR − red) / (NIR + red) from Sentinel-2; high for green canopy, low for bare soil.

**NISAR**: the NASA-ISRO L-band and S-band SAR satellite; this project uses its L-band products.

**`no_baseline`**: status of a cane polygon whose candidate harvest has fewer than 3 standing looks before it in its orbit.

**Out-of-fold**: a prediction made by a model that did not see that field in training.

**`p_cane`, `p_cane_radar`, `p_cane_optical`**: the delivered gate's P(cane), and the radar-only and optical-only probabilities.

**PostGIS**: the spatial extension of PostgreSQL used by the product database.

**Precision / recall**: share of predicted cane that is cane / share of cane that is predicted.

**Provisional (data maturity)**: NISAR products released before final calibration; values may change.

**Ratoon**: cane regrowing from the stubble after a cut.

**Safra**: the Brazilian name for the cane harvest season (Centre-South: April to November).

**SCL (scene classification layer)**: the Sentinel-2 L2A per-pixel class map used to mask clouds and shadows.

**Segment**: one polygon from the automatic segmentation; often a fragment of a field.

**Sentinel-1 / Sentinel-2**: ESA's C-band radar and optical satellites.

**SNIC**: a superpixel segmentation in Earth Engine, used only for p36's neighbours.

**Supplemental frames**: NISAR acquisitions before the standard observation series (2026-06-17), with no guaranteed cadence.

**Vector tiles (MVT)**: map data cut into tiles on the server (`ST_AsMVT` in PostGIS), so the browser loads only what is in view.

**Verifier**: deterministic code that checks every summary against its facts before it is shown.

**VIIRS**: the instrument on Suomi-NPP, NOAA-20 and NOAA-21 whose 375 m fire detections FIRMS serves.

**WorldCover**: ESA's 10 m global land-cover map for 2021.

**xgboost**: a gradient-boosted decision-tree library; the delivered harvest detector.
