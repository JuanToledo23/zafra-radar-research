"""Area-wide harvest figures (exercise 4d), reusable by the frontend.

Two views of the same per-segment table (data/results/harvest_dates_<zafra>.geojson):

* harvest_map: every segment, cane coloured by harvest date on one sequential hue (light =
  early, dark = late), non-cane in a light neutral, cane without a detection hatched.
* harvest_vs_fire_weeks: harvested segments per week and VIIRS detections per week in the
  AOI, as two stacked panels on a shared week axis (two measures, two y-scales -> two
  panels, never a dual axis).

The data-preparation helpers (weekly_counts) are separate from the drawing, so a web page
can use the same numbers.

Colours: the dataviz skill's reference palette (sequential blue 250..700, ink / muted /
gridline tokens, light surface).
"""

from __future__ import annotations

import pandas as pd

SURFACE = "#fcfcfb"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e1e0d9"
NON_CANE = "#dcdbd5"  # recessive neutral: context, not data
NO_DETECTION = "#a8a7a0"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # 250..700
BAR = "#2a78d6"  # blue 450: harvests
FIRE_BAR = "#eb6834"  # categorical slot 2 (orange): fires, a different measure
FIRE_ALL = "#f6c2a8"  # light orange: all fires, behind the cane subset (same measure)
WC_EDGE = INK  # outline of cane segments that WorldCover maps as trees / grassland


def weekly_counts(dates, start, end) -> pd.Series:
    """Counts per ISO week (Monday start) from `start` to `end`, zero-filled."""
    d = pd.to_datetime(pd.Series(dates)).dropna()
    d = d.dt.tz_localize(None) if d.dt.tz is not None else d
    weeks = pd.date_range(pd.Timestamp(start).to_period("W-SUN").start_time, end, freq="W-MON")
    c = d.dt.to_period("W-SUN").dt.start_time.value_counts()
    return c.reindex(weeks, fill_value=0).rename_axis("week")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def harvest_map(
    gdf,
    path=None,
    date_col="harvest_date",
    cane_col="cane_pred",
    title=None,
    ax=None,
    flag_col="wc_disagrees",
    crs="EPSG:32614",
):
    """Segments coloured by harvest date; non-cane grey; cane without a date hatched.
    Drawn in the metric CRS `crs` (default Morelos UTM 14N; a site's metric_crs).

    If `flag_col` is present, cane segments where it is true get a dark outline (e.g. the
    gate says cane but WorldCover says trees or grassland); segments with confidence
    "uncertain" are cross-hatched on top. With a `status` column, cane
    without a date is split into "harvested before coverage / no baseline" and "no harvest
    detected"."""
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    g = gdf.to_crs(crs).copy()
    g[date_col] = pd.to_datetime(g[date_col])
    cane = g[cane_col].fillna(False).astype(bool)
    dated = cane & g[date_col].notna()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 10), facecolor=SURFACE)
    else:
        fig = ax.figure
    g[~cane].plot(ax=ax, color=NON_CANE, linewidth=0)
    early = cane & ~dated
    if "status" in g:
        early = early & g["status"].isin(["harvested_before_coverage", "no_baseline"])
    g[cane & ~dated & ~early].plot(
        ax=ax, facecolor=SURFACE, edgecolor=NO_DETECTION, hatch="////", linewidth=0.2
    )
    g[early].plot(ax=ax, facecolor=NO_DETECTION, edgecolor=SURFACE, linewidth=0.2)
    cmap = mcolors.LinearSegmentedColormap.from_list("harvest", BLUE_RAMP)
    if dated.any():
        t = mdates.date2num(g.loc[dated, date_col])
        norm = mcolors.Normalize(t.min(), t.max())
        g[dated].plot(ax=ax, color=cmap(norm(t)), linewidth=0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
        cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        cb.outline.set_visible(False)
        cb.set_label("estimated harvest date", color=MUTED)
    flag = cane & g[flag_col].fillna(False).astype(bool) if flag_col in g else cane & False
    if flag.any():
        g[flag].plot(ax=ax, facecolor="none", edgecolor=WC_EDGE, linewidth=0.5)
    unsure = cane & (g["confidence"] == "uncertain") if "confidence" in g else cane & False
    if unsure.any():
        g[unsure].plot(ax=ax, facecolor="none", edgecolor=WC_EDGE, hatch="xxxx", linewidth=0.3)
    extra = []
    if "status" in g:
        extra.append(
            Patch(
                facecolor=NO_DETECTION,
                label=f"cane, harvested before coverage / no baseline ({int(early.sum())})",
            )
        )
    if unsure.any():
        extra.append(
            Patch(
                facecolor="none",
                edgecolor=WC_EDGE,
                hatch="xxxx",
                label=f"uncertain: cane by the gate, trees/grassland by WorldCover, "
                f"no date ({int(unsure.sum())})",
            )
        )
    if flag.any():
        extra.append(
            Patch(
                facecolor="none",
                edgecolor=WC_EDGE,
                label=f"cane by the gate, trees/grassland by WorldCover ({int(flag.sum())})",
            )
        )
    ax.legend(
        handles=[
            Patch(facecolor=NON_CANE, label=f"not cane ({int((~cane).sum())})"),
            Patch(
                facecolor=SURFACE,
                edgecolor=NO_DETECTION,
                hatch="////",
                label=f"cane, no harvest detected ({int((cane & ~dated & ~early).sum())})",
            ),
            Patch(facecolor=BLUE_RAMP[2], label=f"cane, harvest dated ({int(dated.sum())})"),
            *extra,
        ],
        loc="upper left",
        bbox_to_anchor=(0, 0),  # below the map, so it never covers segments
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    ax.set_axis_off()
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
    return ax


def harvest_vs_fire_weeks(
    harvest_dates, fire_times, start, end, path=None, title=None, fires_cane=None
):
    """Two panels on one week axis: harvested segments per week, VIIRS detections per week.

    With `fires_cane` (a subset of fire_times, e.g. detections near cane), the fire panel
    shows all detections as a light bar and the cane subset in front of it, same scale.
    Returns the weekly table (week, harvests, fires[, fires_cane]) so callers can report it.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    h = weekly_counts(harvest_dates, start, end)
    f = weekly_counts(fire_times, start, end)
    tab = pd.DataFrame({"harvests": h, "fires": f})
    if fires_cane is not None:
        tab["fires_cane"] = weekly_counts(fires_cane, start, end)
    tab = tab.fillna(0).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, facecolor=SURFACE)
    fire_label = "VIIRS fire detections in the AOI"
    if fires_cane is not None:
        fire_label = "VIIRS fire detections: all in the AOI (light) and within 200 m of cane"
    for ax, col, color, label in (
        (axes[0], "harvests", BAR, "cane segments harvested (detector date)"),
        (axes[1], "fires", FIRE_BAR, fire_label),
    ):
        _style(ax)
        x = tab.index + pd.Timedelta(days=3.5)
        if col == "fires" and fires_cane is not None:
            ax.bar(x, tab["fires"], width=5.5, color=FIRE_ALL, zorder=2, label="all AOI")
            ax.bar(x, tab["fires_cane"], width=5.5, color=color, zorder=3, label="near cane")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        else:
            ax.bar(x, tab[col], width=5.5, color=color, zorder=2)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_ylabel("per week", color=MUTED, fontsize=9)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        peak = tab[col].idxmax()
        if tab[col].max() > 0:
            ax.annotate(
                f"peak {tab[col].max()} (wk of {peak:%b %d})",
                (peak + pd.Timedelta(days=3.5), tab[col].max()),
                xytext=(6, -2),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
                va="top",
            )
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    if title:
        fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return tab


def week_correlation(harvests: pd.Series, fires: pd.Series, max_lag: int = 4) -> pd.DataFrame:
    """Pearson r between weekly harvests in week w and fires in week w + k, for
    k = -max_lag..max_lag (`fires_weeks_after`; k = 1: fires one week after the harvests)."""
    rows = [
        {"fires_weeks_after": k, "r": harvests.corr(fires.shift(-k))}
        for k in range(-max_lag, max_lag + 1)
    ]
    return pd.DataFrame(rows)


__all__ = ["harvest_map", "harvest_vs_fire_weeks", "weekly_counts", "week_correlation"]
