-- Zafra Radar product schema (exercise 7). Empty tables only; columns follow
-- docs/METHODS.md ("Frontend export") and results/4e/weekly.csv. Idempotent.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS sites (
    site_id     text PRIMARY KEY,              -- 'morelos', 'brazil' (configs/<site>.yaml)
    name        text NOT NULL,
    season      text,                          -- e.g. '2025-26'
    bbox        geometry(Polygon, 4326)
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id            text PRIMARY KEY,
    site_id               text NOT NULL REFERENCES sites (site_id),
    geom                  geometry(MultiPolygon, 4326) NOT NULL,
    area_ha               real,
    p_cane                real,
    p_cane_radar          real,
    harvest_date          date,                -- midpoint of the standing/harvested gap
    date_uncertainty_days real,                -- ± days (half the gap)
    status                text,                -- harvested | not_detected | no_baseline
    confidence            text,                -- high | medium | low | uncertain | none
    hv_drop_db            real,
    coh_min               real,
    wc_disagrees          boolean
);
CREATE INDEX IF NOT EXISTS segments_geom_gist ON segments USING gist (geom);
CREATE INDEX IF NOT EXISTS segments_site_date ON segments (site_id, harvest_date);

CREATE TABLE IF NOT EXISTS segment_series (
    segment_id       text NOT NULL REFERENCES segments (segment_id) ON DELETE CASCADE,
    acquisition_date date NOT NULL,
    orbit            text NOT NULL,            -- 'asc' | 'desc', never averaged
    hv_db            real,
    hh_db            real,
    coherence        real,
    PRIMARY KEY (segment_id, acquisition_date, orbit)
);

CREATE TABLE IF NOT EXISTS fires (
    fire_id        bigserial PRIMARY KEY,
    site_id        text REFERENCES sites (site_id),
    geom           geometry(Point, 4326) NOT NULL,
    datetime_local timestamp NOT NULL,
    sensor         text,                       -- VIIRS satellite (snpp, noaa20, noaa21)
    frp            real,
    segment_id     text REFERENCES segments (segment_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS fires_geom_gist ON fires USING gist (geom);
CREATE INDEX IF NOT EXISTS fires_segment ON fires (segment_id);

CREATE TABLE IF NOT EXISTS weekly (
    site_id     text NOT NULL REFERENCES sites (site_id),
    week_start  date NOT NULL,
    harvests    integer NOT NULL DEFAULT 0,
    fires_all   integer NOT NULL DEFAULT 0,
    fires_cane  integer NOT NULL DEFAULT 0,
    PRIMARY KEY (site_id, week_start)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id    bigserial PRIMARY KEY,
    segment_id  text NOT NULL REFERENCES segments (segment_id) ON DELETE CASCADE,
    code        text NOT NULL,                 -- e.g. FIRE_ACTIVE_OUTSIDE_HOURS
    severity    text NOT NULL,
    evidence    jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS alerts_segment ON alerts (segment_id);
