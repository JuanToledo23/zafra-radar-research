"""Tests for field segmentation cleaning, IoU matching and the vectorized zonal helpers."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import Polygon, box

from zafra import plots, zonal
from zafra import segments as S

CRS = S.METRIC_CRS
X0, Y0 = 480_000.0, 2_060_000.0  # somewhere in the AOI, UTM 14N


def sq(x, y, side):
    return box(X0 + x, Y0 + y, X0 + x + side, Y0 + y + side)


def gdf(geoms, **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs=CRS)


# --- cleaning -----------------------------------------------------------------------------


def test_area_filter_keeps_only_0_5_to_60_ha():
    # 50 m side = 0.25 ha, 100 m = 1 ha, 700 m = 49 ha, 800 m = 64 ha
    g = gdf([sq(0, 0, 50), sq(1000, 0, 100), sq(2000, 0, 700), sq(4000, 0, 800)], k=[0, 1, 2, 3])
    out, counts = S.clean_segments(g)
    assert sorted(out["k"]) == [1, 2]
    assert counts["too_small"] == 1 and counts["too_large"] == 1 and counts["kept"] == 2
    assert out["area_ha"].round(2).tolist() == [1.0, 49.0]


def test_invalid_bowtie_is_fixed_not_dropped():
    # Self-intersecting "bowtie": two 200 x 100 m triangles, 1 ha each.
    bow = Polygon([(X0, Y0), (X0 + 200, Y0 + 200), (X0 + 200, Y0), (X0, Y0 + 200), (X0, Y0)])
    assert not bow.is_valid
    out, _ = S.clean_segments(gdf([bow]))
    assert len(out) == 1
    assert out.geometry.iloc[0].is_valid
    assert out["area_ha"].iloc[0] == pytest.approx(2.0, rel=0.01)


def test_simplify_removes_tiny_wiggles():
    pts = [(X0 + x, Y0 + (1.0 if x % 20 == 10 else 0.0)) for x in range(0, 201, 10)]
    poly = Polygon([*pts, (X0 + 200, Y0 + 200), (X0, Y0 + 200)])
    out, _ = S.clean_segments(gdf([poly]), min_ha=0)
    assert len(out.geometry.iloc[0].exterior.coords) < len(poly.exterior.coords)


def test_polygonal_part_of_collection():
    from shapely.geometry import GeometryCollection, LineString

    gc = GeometryCollection([sq(0, 0, 100), LineString([(0, 0), (1, 1)])])
    assert S._polygonal(gc).area == pytest.approx(10_000)
    assert S._polygonal(LineString([(0, 0), (1, 1)])) is None


def test_labels_to_polygons_and_mask_to_labels():
    mask = np.zeros((10, 10), np.uint8)
    mask[1:4, 1:4] = 255  # one block
    mask[6:9, 6:9] = 255  # another, not touching
    lab = S.mask_to_labels(mask, grow_px=0)
    assert lab.max() == 2
    tr = Affine(10, 0, X0, 0, -10, Y0)
    polys = S.labels_to_polygons(lab, tr)
    assert len(polys) == 2
    assert sorted(polys.area.round()) == [900.0, 900.0]
    # growing gives the eroded border back
    assert (S.mask_to_labels(mask, grow_px=1) > 0).sum() > (lab > 0).sum()


# --- IoU matching -------------------------------------------------------------------------


def test_iou_exact_and_half_overlap():
    ref = gdf([sq(0, 0, 100), sq(1000, 0, 100)], id=["a", "b"])
    seg = gdf([sq(0, 0, 100), sq(1050, 0, 100)], seg_id=["s1", "s2"])
    m = S.match_fields(ref, seg).set_index("field_id")
    assert m.loc["a", "iou"] == pytest.approx(1.0)
    assert m.loc["a", "best_segment"] == "s1"
    # half overlap: 5000 / (10000 + 10000 - 5000) = 1/3
    assert m.loc["b", "iou"] == pytest.approx(1 / 3)
    assert m.loc["b", "cover_share"] == pytest.approx(0.5)


def test_missed_field():
    m = S.match_fields(gdf([sq(0, 0, 100)], id=["a"]), gdf([sq(500, 500, 100)], seg_id=["s"]))
    assert m["iou"].iloc[0] == 0 and m["best_segment"].iloc[0] is None
    assert S.summarize_matches(m)["missed"] == 1


def test_over_segmented_when_two_pieces_above_20pct():
    ref = gdf([sq(0, 0, 100)], id=["a"])
    halves = gdf(
        [
            sq(0, 0, 100).intersection(box(X0, Y0, X0 + 50, Y0 + 100)),
            sq(0, 0, 100).intersection(box(X0 + 50, Y0, X0 + 100, Y0 + 100)),
        ],
        seg_id=["l", "r"],
    )
    m = S.match_fields(ref, halves).iloc[0]
    assert m["over_segmented"] and m["n_pieces_gt20"] == 2
    assert not m["under_segmented"]
    assert m["iou"] == pytest.approx(0.5)


def test_sliver_does_not_count_as_a_piece():
    ref = gdf([sq(0, 0, 100)], id=["a"])
    seg = gdf(
        [box(X0, Y0, X0 + 90, Y0 + 100), box(X0 + 90, Y0, X0 + 100, Y0 + 100)],
        seg_id=["big", "sliver"],
    )
    m = S.match_fields(ref, seg).iloc[0]
    assert m["n_pieces_gt20"] == 1 and not m["over_segmented"]


def test_under_segmented_when_segment_above_1_5x():
    ref = gdf([sq(0, 0, 100)], id=["a"])
    m = S.match_fields(ref, gdf([box(X0, Y0, X0 + 200, Y0 + 100)], seg_id=["s"])).iloc[0]
    assert m["under_segmented"] and not m["over_segmented"]
    assert m["iou"] == pytest.approx(0.5)
    m = S.match_fields(ref, gdf([box(X0, Y0, X0 + 140, Y0 + 100)], seg_id=["s"])).iloc[0]
    assert not m["under_segmented"]  # 1.4x


def test_summary_counts():
    m = pd.DataFrame(
        {
            "iou": [0.9, 0.5, 0.2, 0.0],
            "over_segmented": [False, True, False, False],
            "under_segmented": [False, False, True, False],
            "best_segment": ["s1", "s2", "s3", None],
        }
    )
    s = S.summarize_matches(m)
    assert s["median_iou"] == pytest.approx(0.35)
    assert s["n_iou_ge_0_5"] == 2 and s["share_iou_ge_0_5"] == 0.5
    assert (s["over_segmented"], s["under_segmented"], s["missed"]) == (1, 1, 1)


# --- zonal helpers ------------------------------------------------------------------------


def test_rasterize_pixel_centre_rule_and_bincount_stats():
    tr = Affine(10, 0, X0, 0, -10, Y0 + 40)  # 4 x 4 grid of 10 m pixels
    # covers the centres of the left 2 columns only (x < 24 m, centres at 5 and 15)
    lab = zonal.rasterize_ids([box(X0, Y0, X0 + 24, Y0 + 40)], tr, (4, 4))
    assert (lab[:, :2] == 1).all() and (lab[:, 2:] == 0).all()
    vals = np.arange(16, dtype=float).reshape(4, 4)
    valid = np.ones_like(lab, bool)
    valid[0, 0] = False
    assert zonal.zonal_count(lab, valid, 1)[0] == 7
    assert zonal.zonal_sum(lab, vals, valid, 1)[0] == pytest.approx(vals[:, :2].sum())
    assert zonal.zonal_min(lab, vals, valid, 1)[0] == 1.0


def test_zonal_min_empty_label_is_nan():
    lab = np.array([[1, 1], [0, 0]])
    out = zonal.zonal_min(lab, np.ones((2, 2)), np.ones((2, 2), bool), 2)
    assert out[0] == 1 and np.isnan(out[1])


def test_weekly_counts_zero_fill():
    w = plots.weekly_counts(["2026-01-05", "2026-01-07", "2026-01-20"], "2026-01-01", "2026-01-31")
    assert w.loc[pd.Timestamp("2026-01-05")] == 2
    assert w.loc[pd.Timestamp("2026-01-12")] == 0
    assert w.loc[pd.Timestamp("2026-01-19")] == 1
    assert w.sum() == 3
