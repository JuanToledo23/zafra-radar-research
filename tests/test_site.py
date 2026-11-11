"""Tests for exercise 5: site configuration (configs/*.yaml, zafra.site) and the Morelos
outputs being unchanged by the refactor (golden test on results/4e/weekly.csv)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from zafra import cane_features as CF
from zafra import features as FT
from zafra import fires as F
from zafra import plots
from zafra import segments as S
from zafra.detect import cane_gate as G
from zafra.paths import RESULTS_DIR, ROOT
from zafra.site import NISAR_ROLES, PATH_KEYS, available_sites, load_site

# sha256 of results/4e/weekly.csv as committed with exercise 4e (c58a18e) and reproduced by
# scripts/14_area_harvest_map.py --site morelos after the refactor.
WEEKLY_SHA256 = "b5ff3b159abd28c602734bda8078f9cfde7957070498a342606f4112c1941186"
WEEKLY = RESULTS_DIR / "4e" / "weekly.csv"


# --- config loading ------------------------------------------------------------------------


def test_both_sites_are_available():
    assert {"morelos", "brazil"} <= set(available_sites())


@pytest.mark.parametrize("name", ["morelos", "brazil"])
def test_site_config_is_complete(name):
    s = load_site(name)
    assert s.name == name
    lon0, lat0, lon1, lat1 = s.aoi
    assert lon0 < lon1 and lat0 < lat1
    assert set(NISAR_ROLES) <= set(s.nisar) and set(PATH_KEYS) <= set(s.paths)
    assert s.nisar["gcov_asc"].direction == "ASCENDING"
    assert s.nisar["gcov_desc"].direction == "DESCENDING"
    assert s.nisar["gunw_asc"].product == "GUNW"
    assert s.crop_year[0] <= s.season[0] < s.season[1] == s.crop_year[1]
    assert all(p.is_relative_to(ROOT) for p in s.paths.values())
    # the zafra is registered for cane_features with the config's window and crop year
    assert CF.zafra_bounds(s.zafra) == s.season
    assert CF.crop_year_bounds(s.zafra) == s.crop_year
    assert CF.pre_months(s.zafra) == s.baseline_months


def test_morelos_config_holds_the_old_constants():
    """The values that were hard-coded in scripts 00, 12-15 before exercise 5."""
    s = load_site("morelos")
    assert s.aoi == s.firms_bbox == (-99.35, 18.50, -99.05, 18.75)
    assert s.metric_crs == S.METRIC_CRS == F.METRIC_CRS == "EPSG:32614"
    assert s.zafra == G.TRAIN_ZAFRA == "2025-26"
    assert s.season == (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-06-01"))
    assert s.crop_year == (pd.Timestamp("2025-06-01"), pd.Timestamp("2026-06-01"))
    assert s.scene_window == ("2025-12-01", "2026-02-01")
    assert s.baseline_months == FT.BASELINE_MONTHS == CF.PRE_MONTHS == (11, 12)
    assert s.season_months == tuple(G.MORELOS_ZAFRA_MONTHS)
    asc, desc, gunw = (s.nisar[r] for r in NISAR_ROLES)
    assert (asc.track, asc.frame, asc.dir) == (19, 11, ROOT / "data" / "nisar")
    assert (desc.track, desc.frame) == (113, 79)
    assert desc.dir.name == "GCOV_DESCENDING_t113f79"
    assert (gunw.track, gunw.frame) == (19, 11) and gunw.dir.name == "GUNW_ASCENDING_t19f11"
    assert s.path("results_dir") == RESULTS_DIR / "4e"
    assert s.path("segments") == ROOT / "data" / "fields" / "segments.geojson"
    assert s.reference_fields == ROOT / "data" / "fields" / "fields_v2.geojson"


def test_brazil_config_is_a_separate_site():
    m, b = load_site("morelos"), load_site("brazil")
    assert b.epsg == 32723 and b.season_months == (4, 5, 6, 7, 8, 9, 10, 11)
    # the season is shifted, not copied: green plateau (Dec-Mar) first, harvest Apr-Nov
    assert b.crop_year == (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-12-01"))
    assert b.season == (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-12-01"))
    assert b.reference_fields is None and not b.check_windows
    # nothing of the new site may overwrite a Morelos file
    assert not set(b.paths.values()) & set(m.paths.values())
    assert b.firms_dir != m.firms_dir and b.nisar_root != m.nisar_root


def test_gate_variant_is_per_site():
    # both sites deliver the optical gate (B); Brazil delivered the radar gate (A) until the
    # corrupt 07-18 scene was excluded, and A is still reported there as radar evidence
    assert load_site("morelos").gate_variant == "B"
    assert load_site("brazil").gate_variant == "B"


def test_unknown_site_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="available"):
        load_site("atlantis")


def test_registering_a_known_zafra_with_another_window_fails():
    load_site("morelos")
    with pytest.raises(ValueError, match="already registered"):
        CF.register_zafra("2025-26", "2025-10-01", "2026-06-01")


def test_collapse_in_zafra_matches_the_morelos_month_rule():
    """Window rule (new) == Nov-May month rule (old) for every day of the Morelos crop year."""
    load_site("morelos")
    days = pd.date_range("2025-06-01", "2026-05-31", freq="D")
    df = pd.DataFrame(
        {
            "zafra": "2025-26",
            "collapse_date": days,
            "ndvi_collapse_month": days.month,
        }
    )
    new = G.design_matrix(df, ["collapse_in_zafra"])["collapse_in_zafra"]
    old = G.design_matrix(df.drop(columns=["collapse_date"]), ["collapse_in_zafra"])
    assert new.equals(old["collapse_in_zafra"])
    assert np.isnan(CF.collapse_in_zafra(pd.NaT, "2025-26"))


def test_collapse_in_zafra_follows_the_brazil_season():
    load_site("brazil")
    assert CF.collapse_in_zafra("2026-05-15", "2026-27") == 1.0  # Apr-Nov harvest
    assert CF.collapse_in_zafra("2026-11-30", "2026-27") == 1.0
    assert CF.collapse_in_zafra("2026-01-15", "2026-27") == 0.0  # rainy season


def test_season_baseline_uses_the_given_months():
    d = pd.Series(pd.to_datetime(["2026-06-22", "2026-07-04", "2026-07-16", "2026-08-21"]))
    v = pd.Series([-10.0, -11.0, -12.0, -20.0])
    assert FT.season_baseline(d, v, months=(6, 7)) == -11.0
    # Morelos months: no Nov-Dec scene, so the fallback (first 3 overall) gives the same
    assert FT.season_baseline(d, v) == -11.0


# --- golden test: Morelos weekly series unchanged -------------------------------------------


def test_morelos_weekly_csv_is_unchanged():
    assert hashlib.sha256(WEEKLY.read_bytes()).hexdigest() == WEEKLY_SHA256


def test_morelos_weekly_series_rebuilt_from_config():
    """The weekly harvest and fire counts, rebuilt from the delivered per-segment GeoJSON and
    the FIRMS CSVs with only the morelos config's season, bbox and CRS, equal weekly.csv."""
    import geopandas as gpd

    s = load_site("morelos")
    path = s.path("harvest_geojson")
    if not path.exists() or not any(s.firms_dir.glob("*.csv")):
        pytest.skip("Morelos results / FIRMS not on disk (git-ignored)")
    weekly = pd.read_csv(WEEKLY, index_col=0, parse_dates=True)
    out = gpd.read_file(path, columns=["harvest_date", "harvest_date_no_baseline_rule", "area_ha"])
    end = s.season[1] - pd.Timedelta(days=1)
    for col, src in (
        ("harvests", "harvest_date"),
        ("harvests_no_baseline_rule", "harvest_date_no_baseline_rule"),
    ):
        got = plots.weekly_counts(out[src], s.season[0], end)
        assert got.to_numpy().tolist() == weekly[col].to_numpy().tolist(), col
    d = out.dropna(subset=["harvest_date"])
    wk = pd.to_datetime(d["harvest_date"]).dt.to_period("W-SUN").dt.start_time
    ha = d.groupby(wk)["area_ha"].sum().reindex(weekly.index).fillna(0).round(2)
    np.testing.assert_allclose(ha.to_numpy(), weekly["harvest_ha"].to_numpy(), atol=0.005)
    fires = F.load_firms(s.firms_dir)
    fires = fires.cx[s.aoi[0] : s.aoi[2], s.aoi[1] : s.aoi[3]]
    t = fires["acq_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    t = t[(t >= s.season[0]) & (t < s.season[1])]
    got = plots.weekly_counts(t, s.season[0], end)
    assert got.to_numpy().tolist() == weekly["fires_all"].to_numpy().tolist()


def test_polygons_follow_a_product_grid_in_another_utm_zone():
    """zonal: polygons on the site's grid are used as they are; on a frame gridded in the
    neighbouring UTM zone they are reprojected and the AOI bounds recomputed."""
    import geopandas as gpd
    from shapely.geometry import box

    from zafra import zonal

    g = gpd.GeoDataFrame(geometry=[box(190_000, 7_670_000, 191_000, 7_671_000)], crs=32723)
    bounds = tuple(g.total_bounds)
    same, b = zonal._on_grid(g, bounds, 32723)
    assert same is g and b == bounds
    other, b = zonal._on_grid(g, bounds, 32722)
    assert other.crs.to_epsg() == 32722
    assert other.geometry.iloc[0].area == pytest.approx(1e6, rel=0.01)
    assert b[0] < other.total_bounds[0] and b[2] > other.total_bounds[2]
