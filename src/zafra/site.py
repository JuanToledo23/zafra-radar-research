"""Site configuration (exercise 5): everything that used to be hard-coded for Morelos.

    from zafra.site import load_site
    site = load_site("morelos")      # configs/morelos.yaml
    site.aoi, site.metric_crs, site.season, site.path("segments")

A site is one area of interest and one harvest season. The YAML holds:

* aoi: lon0, lat0, lon1, lat1 (also the FIRMS bbox unless `firms.bbox` is given) and the
  metric CRS (`epsg`, the UTM zone the S2 grid and all buffers / areas use);
* nisar: the GCOV ascending, GCOV descending and GUNW ascending track/frame, the folder
  each is read from, and the search window used by scripts/00_download_nisar.py;
  optional `exclude_dates`: acquisitions left out of every radar table (zonal.drop_excluded),
  each with its date, the sources it applies to and the reason;
* zafra: the season's name, its window (start, end exclusive), the crop year the NDVI
  phenology is read over, the Dec-Jan-like full-canopy scene window, the months in which
  the radar baseline is taken (cane still standing everywhere), and optional diagnostic
  windows for script 14's summary (check_windows);
* firms: bbox and date range for scripts/00_download_firms.py, and the folder;
* gate_variant: the cane-gate variant delivered for this site (cane_gate.VARIANT_ALIASES or
  the letter: optical / B, radar / A, ...); the other gate is kept alongside as a check;
* reference_fields: hand-traced polygons to score the segmentation against (null: none);
* paths: every input and output file of scripts 12-15, relative to the repo root;
* alerts, summary (optional, exercise 6): local time zone, burn window and alert-rule
  parameters (zafra.alerts); model and limits of the field summaries (zafra.summary);
* web_rasters (optional, exercise 9): GCOV source, dates, boxcar size and tile zooms of
  the web map layers (zafra.rasters, scripts/18_web_rasters.py).

The trained models are not part of a site: the cane gate and the harvest detector are
always the exercise-4b/4c models fitted on the Morelos fields_v2 polygons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from zafra.paths import CONFIGS_DIR, ROOT

DEFAULT_SITE = "morelos"
NISAR_ROLES = ("gcov_asc", "gcov_desc", "gunw_asc")
PATH_KEYS = (
    "s2_dir",
    "ndvi_stack",
    "temporal",
    "worldcover",
    "segments",
    "segmentation_results_dir",
    "ts_cache",
    "timeseries",
    "coherence",
    "cane_features",
    "harvest_geojson",
    "merged_geojson",
    "slim_geojson",
    "results_dir",
)


@dataclass(frozen=True)
class NisarSource:
    product: str  # GCOV | GUNW
    direction: str  # ASCENDING | DESCENDING
    track: int
    frame: int
    dir: Path


@dataclass(frozen=True)
class ExcludedScene:
    date: str  # YYYY-MM-DD (UTC acquisition day, as in the file names)
    sources: tuple[str, ...]  # NISAR roles: gcov_asc, gcov_desc, gunw_asc
    reason: str


@dataclass(frozen=True)
class Site:
    name: str
    description: str
    aoi: tuple[float, float, float, float]
    epsg: int
    nisar: dict[str, NisarSource]
    nisar_root: Path
    nisar_search: tuple[str, str]
    zafra: str
    season: tuple[pd.Timestamp, pd.Timestamp]
    crop_year: tuple[pd.Timestamp, pd.Timestamp]
    scene_window: tuple[str, str]
    baseline_months: tuple[int, ...]
    firms_bbox: tuple[float, float, float, float]
    firms_dates: tuple[str, str]
    firms_dir: Path
    gate_variant: str
    reference_fields: Path | None
    paths: dict[str, Path] = field(default_factory=dict)
    check_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=dict)
    alerts: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    web_rasters: dict = field(default_factory=dict)
    nisar_exclude: tuple[ExcludedScene, ...] = ()

    @property
    def metric_crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def center(self) -> tuple[float, float]:
        lon0, lat0, lon1, lat1 = self.aoi
        return (lon0 + lon1) / 2, (lat0 + lat1) / 2

    @property
    def season_months(self) -> tuple[int, ...]:
        """Calendar months of the season window (e.g. 11, 12, 1..5 for Nov 1 .. Jun 1)."""
        a, b = self.season
        return tuple(pd.date_range(a, b - pd.Timedelta(days=1), freq="MS").month)

    def path(self, key: str) -> Path:
        return self.paths[key]

    def excluded_dates(self, role: str) -> set[str]:
        """Acquisition dates excluded for one NISAR role (gcov_asc, gcov_desc, gunw_asc)."""
        return {e.date for e in self.nisar_exclude if role in e.sources}

    def register(self) -> None:
        """Make the zafra known to cane_features (window, crop year, baseline months)."""
        from zafra import cane_features as CF

        CF.register_zafra(
            self.zafra,
            *self.season,
            crop_year_start=self.crop_year[0],
            pre_months=self.baseline_months,
        )


def _p(v) -> Path:
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def _bbox(v) -> tuple[float, float, float, float]:
    t = tuple(float(x) for x in v)
    if len(t) != 4 or not (t[0] < t[2] and t[1] < t[3]):
        raise ValueError(f"bbox must be lon0, lat0, lon1, lat1 with lon0 < lon1, lat0 < lat1: {v}")
    return t


def config_path(name: str) -> Path:
    return CONFIGS_DIR / f"{name}.yaml"


def available_sites() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))


def load_site(name: str = DEFAULT_SITE, register: bool = True) -> Site:
    """Read configs/<name>.yaml. With `register`, the zafra is registered in cane_features."""
    path = config_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no site config {path} (available: {', '.join(available_sites())})"
        )
    c = yaml.safe_load(path.read_text())
    if c.get("name") != name:
        raise ValueError(f"{path}: name {c.get('name')!r} != file name {name!r}")
    aoi = _bbox(c["aoi"]["bbox"])
    n = c["nisar"]
    nisar = {
        role: NisarSource(
            product=s["product"],
            direction=s["direction"],
            track=int(s["track"]),
            frame=int(s["frame"]),
            dir=_p(s["dir"]),
        )
        for role, s in n["sources"].items()
    }
    missing = set(NISAR_ROLES) - set(nisar)
    if missing:
        raise ValueError(f"{path}: nisar.sources lacks {sorted(missing)}")
    exclude = []
    for e in n.get("exclude_dates") or []:
        sources = tuple(e["sources"])
        unknown = set(sources) - set(nisar)
        if unknown or not e.get("reason"):
            raise ValueError(f"{path}: exclude_dates {e}: unknown sources or no reason")
        exclude.append(
            ExcludedScene(pd.Timestamp(e["date"]).date().isoformat(), sources, str(e["reason"]))
        )
    z = c["zafra"]
    season = (pd.Timestamp(z["start"]), pd.Timestamp(z["end"]))
    crop_year = (pd.Timestamp(z["crop_year_start"]), season[1])
    if not crop_year[0] <= season[0] < season[1]:
        raise ValueError(f"{path}: need crop_year_start <= zafra start < zafra end")
    f = c["firms"]
    paths = {k: _p(v) for k, v in c["paths"].items()}
    missing = set(PATH_KEYS) - set(paths)
    if missing:
        raise ValueError(f"{path}: paths lacks {sorted(missing)}")
    ref = c.get("reference_fields")
    from zafra.detect import cane_gate as G

    try:
        gate_variant = G.resolve_variant(str(c["gate_variant"]))
    except KeyError:
        raise ValueError(f"{path}: gate_variant missing") from None
    site = Site(
        name=name,
        description=c.get("description", ""),
        aoi=aoi,
        epsg=int(c["aoi"]["epsg"]),
        nisar=nisar,
        nisar_root=_p(n["root"]),
        nisar_search=(str(n["search"]["start"]), str(n["search"]["end"])),
        zafra=str(z["name"]),
        season=season,
        crop_year=crop_year,
        scene_window=(str(z["scene_window"][0]), str(z["scene_window"][1])),
        baseline_months=tuple(int(m) for m in z["baseline_months"]),
        firms_bbox=_bbox(f.get("bbox") or c["aoi"]["bbox"]),
        firms_dates=(str(f["start"]), str(f["end"])),
        firms_dir=_p(f["dir"]),
        gate_variant=gate_variant,
        reference_fields=_p(ref) if ref else None,
        paths=paths,
        check_windows={
            k: (pd.Timestamp(v[0]), pd.Timestamp(v[1]))
            for k, v in (z.get("check_windows") or {}).items()
        },
        alerts=c.get("alerts") or {},
        summary=c.get("summary") or {},
        web_rasters=c.get("web_rasters") or {},
        nisar_exclude=tuple(exclude),
    )
    if register:
        site.register()
    return site


def add_site_argument(parser) -> None:
    parser.add_argument(
        "--site",
        default=DEFAULT_SITE,
        help=f"site config in configs/<site>.yaml (default {DEFAULT_SITE}; "
        f"available: {', '.join(available_sites())})",
    )
