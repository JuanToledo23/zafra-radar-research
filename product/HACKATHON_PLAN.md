# Hackathon build order (Saturday / Sunday)

> Plan for the separate hackathon application; nothing here is implemented in this repo.

Starting point: the skeleton in this directory (empty schema, `/health`, `/version`, map of a
bundled sample). Estimates in minutes. Inputs are the delivered outputs, nothing is
recomputed: `data/results/harvest_map_2025_26_slim.geojson` (and the Brazil one),
`results/4e/weekly.csv`, `data/features/segments_timeseries.parquet`,
`data/features/segments_coherence.parquet`, `results/6/summaries.jsonl`.

## Before Nov 13

Checked on the dev Mac on 2026-11-11.

- [x] `docker pull postgis/postgis:16-3.4 python:3.12-slim node:24-alpine` (all present)
- [x] `docker compose build --no-cache` from clean (api and web rebuilt 2026-11-11; pip and
      npm need the network only here). The `pgdata` volume was kept; for a fully clean
      start add `docker compose down -v --rmi local` first
- [x] Start offline once: `docker compose -f docker-compose.yml -f docker-compose.offline.yml
      up -d --wait` (healthy in 12 s), then `cd scripts && node screenshot.mjs --offline
      --out ../docs/skeleton_offline.png`: map ready in 1.5 s, `api: ok`, plain basemap, no
      page errors; the only blocked host is the online basemap
- [x] Host ports free: API **8010** (moved from 8000, taken by another local API
      container), web **5174** (moved from 5173, taken by another project's frontend), db
      5434: `lsof -iTCP:8010,5174,5434 -sTCP:LISTEN` prints nothing. Override with
      `API_PORT` / `WEB_PORT` / `DB_PORT` in `product/.env`
- [x] Tokens in the repo-root `.env` present (values not checked against the services):
  - [x] Earthdata (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, ASF login)
  - [x] Earth Engine (`EE_PROJECT`; `earthengine authenticate` credentials present, last
        used successfully for the Sentinel-1 pull on 2026-10-29, not re-tested today)
  - [x] FIRMS (`FIRMS_MAP_KEY`)
  - [x] Anthropic (`ANTHROPIC_API_KEY`, optional: the template is the fallback)
- [x] Inputs on disk for both sites:
  - [x] Morelos: `data/results/harvest_map_2025_26_slim.geojson` (8.5 MB),
        `results/4e/weekly.csv`
  - [x] Brazil (gate B, 07-18 excluded, 2026-11-11):
        `data/results/brazil/harvest_map_slim.geojson` (25.6 MB, 12 210 polygons),
        `results/5_brazil/weekly.csv`
- [x] NISAR raster tiles present for both sites (exercise 9, `scripts/18_web_rasters.py
      --site <site>`, ~70 s each): `data/tiles/<site>/tiles_index.json` and its 24 layer
      folders under `data/tiles/morelos/` and `data/tiles/brazil/`. The Brazil tiles have
      no 2026-07-18 layer (dates 06-24, 07-06, 08-11, 09-04, 09-28, 10-22)

## Saturday: data in, endpoints out

- [ ] **Loader** `api/app/load.py --site <site>` (90)
  - [ ] `sites` row from `configs/<site>.yaml` (name, season, bbox) (10)
  - [ ] slim GeoJSON → `segments` (MultiPolygon 4326, `harvest_date` as date) with
        `COPY` or `executemany`; idempotent per site (delete + insert in one transaction) (30)
  - [ ] `weekly.csv` → `weekly` (10)
  - [ ] series parquet → `segment_series` (HV/HH per orbit; coherence joined on
        `sec_date`), only for segments present in `segments` (30)
  - [ ] `summaries.jsonl` alerts → `alerts` (`evidence` jsonb) (10)
- [ ] Mount `../data` and `../results` read-only into the api container for the loader (10)
- [ ] `GET /sites` (15)
- [ ] `GET /tiles/{z}/{x}/{y}.pbf` via `ST_AsMVT` / `ST_AsMVTGeom` over `segments`
      (`ST_TileEnvelope`, properties harvest_date, confidence, status, segment_id;
      `Cache-Control`) (60)
- [ ] `GET /segments/{id}`: facts + summary text + series (45)
- [ ] `GET /weekly?site=` (15)
- [ ] `GET /alerts?site=&severity=` (20)
- [ ] API tests against a loaded test database (integration marker) (40)

Saturday total: ~5 h

## Sunday: product

- [ ] Map wired to `/api/tiles` (vector source replaces the sample GeoJSON; keep the sample
      as offline demo fallback) (45)
- [ ] Time slider filters by `harvest_date` (`filter` expression on the tile layer; play
      button) (45)
- [ ] Detail panel: facts table + HV / coherence chart per orbit (orbits never averaged)
      with the harvest date ± uncertainty marked (75)
- [ ] Weekly harvests vs fires chart from `/weekly` (30)
- [ ] Alerts list (click → fly to segment) (40)
- [ ] Summary text in the panel (es / en) (20)
- [ ] **Radar swipe** (tiles already on disk, nothing to compute): serve
      `data/tiles/<site>/` as static files, one MapLibre `raster` source per layer from
      `tiles_index.json` (`url_template`, `minzoom`/`maxzoom`, `bounds`), a before / after
      date pair with a swipe control (two maps synced, or a clip on the top layer), the
      change layer `hv_change_<d1>_<d2>` / `hv_change_season` as an overlay with its
      legend from the index `ramp` and `stretch` (60)
- [ ] **Polarization panel**: HH, HV and cross-ratio (HV - HH) curves per field from
      `segment_series` (orbits never averaged), harvest date marked (30)
- [ ] **L-band vs C-band panel** for the 38 labelled fields: NISAR HV / HH per orbit next
      to Sentinel-1 VH / VV per track (asc t78, desc t143, desc t41; never averaged) from
      `data/results/morelos/s1_series_2025_26.parquet` (scripts/19_l_vs_c.py), harvest
      window marked; drops at harvest in `results/9/l_vs_c.md` (45)
- [ ] Site switcher (Morelos / Brazil): fit to the site bbox, reload tiles and weekly (30)
- [ ] Offline re-check with `docker-compose.offline.yml` on the loaded database (15)
- [ ] Screenshots / screen recording for the video (`scripts/screenshot.mjs` per view) (40)

Sunday total: ~7.9 h (5.7 h without the three radar panels)
