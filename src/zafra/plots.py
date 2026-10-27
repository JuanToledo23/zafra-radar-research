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


def harvest_map(gdf, path=None, date_col="harvest_date", cane_col="cane_pred", title=None, ax=None):
    """Segments coloured by harvest date; non-cane grey; cane without detection hatched."""
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    g = gdf.to_crs("EPSG:32614").copy()
    g[date_col] = pd.to_datetime(g[date_col])
    cane = g[cane_col].fillna(False).astype(bool)
    dated = cane & g[date_col].notna()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 10), facecolor=SURFACE)
    else:
        fig = ax.figure
    g[~cane].plot(ax=ax, color=NON_CANE, linewidth=0)
    g[cane & ~dated].plot(
        ax=ax, facecolor=SURFACE, edgecolor=NO_DETECTION, hatch="////", linewidth=0.2
    )
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
    ax.legend(
        handles=[
            Patch(facecolor=NON_CANE, label=f"not cane ({int((~cane).sum())})"),
            Patch(
                facecolor=SURFACE,
                edgecolor=NO_DETECTION,
                hatch="////",
                label=f"cane, no harvest detected ({int((cane & ~dated).sum())})",
            ),
            Patch(facecolor=BLUE_RAMP[2], label=f"cane, harvest dated ({int(dated.sum())})"),
        ],
        loc="lower left",
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


def harvest_vs_fire_weeks(harvest_dates, fire_times, start, end, path=None, title=None):
    """Two panels on one week axis: harvested segments per week, VIIRS detections per week.

    Returns the weekly table (week, harvests, fires) so callers can report it.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    h = weekly_counts(harvest_dates, start, end)
    f = weekly_counts(fire_times, start, end)
    tab = pd.DataFrame({"harvests": h, "fires": f}).fillna(0).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, facecolor=SURFACE)
    for ax, col, color, label in (
        (axes[0], "harvests", BAR, "cane segments harvested (detector date)"),
        (axes[1], "fires", FIRE_BAR, "VIIRS fire detections in the AOI"),
    ):
        _style(ax)
        ax.bar(tab.index + pd.Timedelta(days=3.5), tab[col], width=5.5, color=color, zorder=2)
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


def week_correlation(tab: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Pearson r between weekly harvests and fires shifted by -max_lag..max_lag weeks
    (lag > 0: fires lead harvests)."""
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        r = tab["harvests"].corr(tab["fires"].shift(lag))
        rows.append({"lag_weeks": lag, "r": r})
    return pd.DataFrame(rows)


__all__ = ["harvest_map", "harvest_vs_fire_weeks", "weekly_counts", "week_correlation"]
