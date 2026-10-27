"""VIIRS active-fire detections (NASA FIRMS) and how to match them to cane fields.

What a FIRMS row is
-------------------
* **One row = one hot pixel, not one fire.** VIIRS (on the SNPP, NOAA-20 and NOAA-21
  satellites) has a 375 m "I-band" pixel at nadir. A row says "something inside this
  ~375 x 375 m pixel was hot enough to stand out from its neighbours at that instant".
  latitude/longitude is the pixel *centre*, not the location of the flames. Off nadir the
  pixel grows (up to roughly 0.8 km along scan at the swath edge), so the real footprint
  is often bigger than 375 m. A cane field here is ~100-300 m across: one pixel can cover
  a whole field plus its neighbours, and a detection 300 m from a field may be a burn in
  the field next door.
* **Snapshots, not a continuous record.** Each satellite passes over Morelos about twice a
  day: ~13:30 and ~01:30 local time. With three satellites that is ~6 looks per day,
  bunched around those two times. A cane burn that is lit and finished between overpasses
  (common: they last 30-60 min) is simply not in the data. "No detection" never means
  "no burn".
* **acq_time is HHMM in UTC, stored as an integer.** Leading zeros are lost, so 846 means
  08:46 UTC and 5 would mean 00:05 UTC. acq_date is the UTC date as well.
* **UTC vs local.** Morelos uses America/Mexico_City time, which is UTC-6 all year (Mexico
  dropped daylight saving time in October 2022). A 19:40 UTC detection is 13:40 local; an
  07:50 UTC detection is 01:50 local *on the same date*, but a 03:00 UTC detection is
  21:00 local on the *previous* date. Rules like "burn only between 09:00 and 17:00" are
  about local time, so always compare against `acq_local`.
* **frp = Fire Radiative Power, in megawatts.** The radiant energy the fire emits per
  second, estimated from the pixel's middle-infrared brightness. Bigger and hotter fires
  give higher FRP; it is a rough proxy for how much biomass is burning *at that moment*.
  Cane burns here typically give single-digit MW.
* **confidence for VIIRS 375 m is a letter, not a percentage** (MODIS uses 0-100):
    l = low:     weak thermal anomaly, or a daytime pixel affected by sun glint.
                 Could be a real small fire or a false alarm (hot bare soil, roofs).
    n = nominal: clear thermal anomaly, no glint. Most real fires.
    h = high:    the pixel saturated the sensor; almost certainly a fire.
* **daynight** is D or N (whether the pixel was imaged in daylight). **bright_ti4** is the
  brightness temperature (kelvin) in the I4 channel (3.7 um, middle infrared): the channel
  that "sees" fire; an ordinary surface is ~290-310 K, a fire pixel reads higher.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

LOCAL_TZ = "America/Mexico_City"
# Metric CRS for distances and buffers. UTM zone 14N covers Morelos (-102 to -96 deg lon).
# Lon/lat degrees are not metres (and a degree of longitude is shorter than a degree of
# latitude), so we never buffer or measure in EPSG:4326.
METRIC_CRS = "EPSG:32614"
VIIRS_PIXEL_M = 375

# The CSV's own `satellite` column uses short codes; map them to readable names.
SATELLITE_NAMES = {"N": "SNPP", "N20": "NOAA-20", "N21": "NOAA-21"}
KEEP = ["latitude", "longitude", "frp", "confidence", "daynight", "bright_ti4", "satellite"]


def load_firms(firms_dir) -> gpd.GeoDataFrame:
    """Merge all FIRMS VIIRS CSVs in `firms_dir` into one GeoDataFrame of points (EPSG:4326).

    Adds `acq_utc` (tz-aware UTC) and `acq_local` (tz-aware America/Mexico_City), parsed
    from acq_date + acq_time. Sorted by time.
    """
    paths = sorted(Path(firms_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no FIRMS CSVs in {firms_dir}")
    df = pd.concat([pd.read_csv(p, dtype={"acq_time": str}) for p in paths], ignore_index=True)

    unknown = set(df["satellite"]) - set(SATELLITE_NAMES)
    if unknown:
        raise ValueError(f"unexpected satellite codes in FIRMS CSVs: {unknown}")
    df["satellite"] = df["satellite"].map(SATELLITE_NAMES)

    # "846" -> "0846" -> 08:46. Both date and time are UTC, so we localise to UTC first and
    # only then convert to local time (converting is a shift; localising just labels).
    hhmm = df["acq_time"].str.zfill(4)
    df["acq_utc"] = pd.to_datetime(
        df["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M"
    ).dt.tz_localize("UTC")
    df["acq_local"] = df["acq_utc"].dt.tz_convert(LOCAL_TZ)

    # SP and NRT downloads can overlap: the same pixel at the same time is one detection.
    df = df.drop_duplicates(subset=["satellite", "acq_utc", "latitude", "longitude"])

    df = df[["acq_utc", "acq_local", *KEEP]].sort_values("acq_utc").reset_index(drop=True)
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )


def fires_near_field(
    fires: gpd.GeoDataFrame, field_geom, buffer_m: float = 500, field_crs="EPSG:4326"
) -> gpd.GeoDataFrame:
    """Fires whose pixel centre lies within `buffer_m` metres of the field polygon.

    `field_geom` is a shapely geometry in `field_crs` (fields.geojson is lon/lat, EPSG:4326).
    Adds:
      distance_m   distance from the fire pixel centre to the field centroid
      dist_edge_m  distance to the field boundary (0 = pixel centre inside the field)
    Remember the pixel itself is >= 375 m wide: even dist_edge_m = 0 does not prove the
    fire was in this field rather than in a neighbour sharing the same pixel.
    """
    field = gpd.GeoSeries([field_geom], crs=field_crs).to_crs(METRIC_CRS).iloc[0]
    pts = fires.to_crs(METRIC_CRS)
    near = pts[pts.within(field.buffer(buffer_m))].copy()
    near["distance_m"] = near.distance(field.centroid).round(0)
    near["dist_edge_m"] = near.distance(field).round(0)
    return near.to_crs(fires.crs)


FIELD_TABLE_COLUMNS = [
    "field_id",
    "acq_utc",
    "acq_local",
    "satellite",
    "daynight",
    "confidence",
    "frp",
    "latitude",
    "longitude",
    "distance_m",
    "dist_edge_m",
]


def fires_per_field(fires: gpd.GeoDataFrame, fields: gpd.GeoDataFrame, buffer_m: float = 500):
    """Every detection within `buffer_m` of each field: one row per (field, detection).

    A detection near two fields appears once for each. Filter on `dist_edge_m` for a tighter
    buffer (e.g. <= 200 m) without recomputing. Times are tz-aware (UTC and local).
    """
    parts = []
    for _, f in fields.iterrows():
        near = fires_near_field(fires, f.geometry, buffer_m=buffer_m, field_crs=fields.crs)
        parts.append(pd.DataFrame(near.drop(columns="geometry")).assign(field_id=f["id"]))
    out = pd.concat(parts, ignore_index=True).reindex(columns=FIELD_TABLE_COLUMNS)
    return out.sort_values(["field_id", "acq_utc"], ignore_index=True)


def load_fire_table(path) -> pd.DataFrame:
    """Read a table written by fires_per_field (CSV), restoring the tz-aware times."""
    df = pd.read_csv(path)
    df["acq_utc"] = pd.to_datetime(df["acq_utc"], utc=True)
    df["acq_local"] = df["acq_utc"].dt.tz_convert(LOCAL_TZ)
    return df
