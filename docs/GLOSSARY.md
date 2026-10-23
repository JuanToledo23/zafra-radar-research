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
A harvest can only be dated to within one revisit interval.

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
