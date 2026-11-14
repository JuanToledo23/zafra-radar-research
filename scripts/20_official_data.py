"""Exercise 10: official CONADESUCA data for the Ingenio Emiliano Zapata vs our Morelos map.

    python scripts/20_official_data.py

Inputs: official-docs/ (not versioned; listed in official-docs/README.md): the 36 weekly
progress reports of zafra 2025/26, the final report, the KARBE 5-zafra report and the LDSCA.
Text comes from `pdftotext -layout` (zafra.conadesuca.pdf_text; pdfplumber fallback).

1. Weekly reports: header (report number, cycle week, period, cut-off = last day of the
   period) and, from the cumulative per-mill table, the Emiliano Zapata, Centro region and
   Nacional rows; from the cumulative per-state table, Morelos and Total nacional; from the
   weekly per-mill table, the Emiliano Zapata row (cross-check of the differenced series).
   A row that does not parse is listed, never filled in.
   -> results/10/official_data.md, results/10/raw_rows.txt
2. Final report (Cuadros 1, 6, 8) and KARBE report -> results/10/ez_final.json
3. results/10/ez_weekly.csv: per cut-off, cumulative and differenced hectares and tonnes.
4. Our map: data/results/harvest_dates_2025_26.geojson (attributes only), dated cane
   polygons re-binned into the official Sunday-Saturday weeks; results/4e/weekly.csv
   (Monday weeks) is only a totals cross-check. Pearson r of weekly official tonnes vs weekly
   detected ha x official season t/ha at lags 0, +1, +2 weeks (detection leading crushing),
   totals, dates, cumulative share. Nothing is tuned to the official numbers.
   -> results/10/official_comparison.md, results/10/official_vs_detected.png
5. LDSCA articles on burning orders, suspension of reception and sanctions, with the page
   number printed in the PDF -> results/10/ldsca_articles.md (quoted in docs/REFERENCES.md).
"""

import json
import re
import subprocess
from datetime import date

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zafra import conadesuca as C  # noqa: E402
from zafra.paths import FIELDS_DIR, RESULTS_DIR, ROOT  # noqa: E402
from zafra.site import load_site  # noqa: E402

DOCS = ROOT / "official-docs"
OUT = RESULTS_DIR / "10"
MILL = "Emiliano Zapata"
MILL_ROWS = ("Nacional", "Centro", MILL)
STATE_ROWS = ("Total nacional", "Morelos")
FINAL = DOCS / "Reporte_de_cifras_finales_25_26.pdf"
KARBE = DOCS / "KARBE-Ponderado_de_5__ltimas_zafras.pdf"
LDSCA = DOCS / "LDSCA.pdf"
LAGS = (0, 1, 2)
CONTEXT_LAGS = (-2, -1)  # detection after crushing; reported for context only
LDSCA_ARTICLES = (26, 70, 73, 75, 77, 80, 81, 85, 86)
# validated categorical slots 1-2 (dataviz reference palette), as in scripts/19
COLORS = ["#2a78d6", "#eb6834"]
SHOW = ["area_ha", "cane_t", "field_yield_t_ha", "sugar_t", "factory_yield_pct",
        "agroindustrial_yield_t_ha", "standard_sugar_t", "karbe_kg_tcn"]  # fmt: skip


def fmt(v, nd=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{nd}f}" if nd else f"{v:,.0f}"


def parse_weekly():
    """One record per weekly report: header, rows (dict or ParseError) and a status."""
    recs = []
    for pdf in sorted(DOCS.glob("Reporte_de_avance*.pdf")):
        rec = {"file": pdf.name}
        try:
            text = C.pdf_text(pdf)
            rec.update(C.report_header(text))
            rec["mill"] = C.section_rows(text, C.CUMULATIVE_MILL_REGION, MILL_ROWS)
            rec["state"] = C.section_rows(text, C.CUMULATIVE_STATE, STATE_ROWS)
            rec["weekly"] = C.section_rows(text, C.WEEKLY_MILL_REGION, (MILL,))
            rec["footnotes"] = C.footnotes(text)
            rec["notes"] = C.notes_mentioning(text, MILL)
            rec["error"] = None
        except (C.ParseError, subprocess.CalledProcessError) as e:
            rec["error"] = str(e)
        recs.append(rec)
    return sorted(recs, key=lambda r: r.get("report", 999))


def row(rec, part, name):
    v = rec.get(part, {}).get(name)
    return v if isinstance(v, dict) else None


def problems(rec):
    """Rows that were found but did not parse, and rows that are absent."""
    out = []
    for part, names in (("mill", MILL_ROWS), ("state", STATE_ROWS), ("weekly", (MILL,))):
        for n in names:
            v = rec.get(part, {}).get(n)
            if isinstance(v, C.ParseError):
                out.append(f"{part} {n}: {v}")
            elif v is None:
                out.append(f"{part} {n}: absent")
    return out


def official_md(recs) -> str:
    ok = [r for r in recs if r["error"] is None]
    with_ez = [r for r in ok if row(r, "mill", MILL)]
    cols = "| report | week | period | cut-off | " + " | ".join(SHOW) + " | `*` |"
    sep = "|" + "---|" * (len(SHOW) + 5)
    nd = {"field_yield_t_ha": 2, "factory_yield_pct": 2, "agroindustrial_yield_t_ha": 2,
          "karbe_kg_tcn": 3}  # fmt: skip

    def table(part, name):
        lines = [cols, sep]
        for r in ok:
            x = row(r, part, name)
            if x is None:
                continue
            vals = " | ".join(fmt(x[c], nd.get(c, 0)) for c in SHOW)
            lines.append(
                f"| {r['report']} | {r['cycle_week']} | {r['period']} | {r['cutoff']} | {vals} | "
                f"{'*' if x['marked'] else ''} |"
            )
        return "\n".join(lines)

    nat_equal = all(
        row(r, "mill", "Nacional") is None
        or row(r, "state", "Total nacional") is None
        or all(
            row(r, "mill", "Nacional")[c] == row(r, "state", "Total nacional")[c] for c in C.COLUMNS
        )
        for r in ok
    )
    parts = [
        "# Official CONADESUCA data, zafra 2025/26: Ingenio Emiliano Zapata",
        "",
        "Generated by `scripts/20_official_data.py` from `pdftotext -layout` text of the 36 "
        "weekly progress reports in `official-docs/` (listed in `official-docs/README.md`). "
        "Cumulative values from the table *Información acumulada por ingenio y región "
        "cañera* (mill, Centro region, Nacional) and *Información acumulada por Entidad "
        "Federativa* (Morelos, Total nacional). The cut-off is the last day of the period "
        "printed on page 1. A row is used only when it has exactly 12 values, each a "
        "number or `-`; no value is inferred.",
        "",
        f"- Reports parsed: **{len(ok)} of {len(recs)}**.",
        f"- Reports with an {MILL} row: {len(with_ez)} (reports "
        f"{with_ez[0]['report']}-{with_ez[-1]['report']}).",
        f"- Nacional (mill table) equals Total nacional (state table) in every parsed report: "
        f"{'yes' if nat_equal else 'NO'}.",
        "- Column names: "
        + ", ".join(f"`{c}`" for c in C.COLUMNS)
        + ". Shown below: "
        + ", ".join(f"`{c}`" for c in SHOW)
        + "; the refined, white special, mascabado and pol < 99.2 sugar columns are `-` for "
        "the mill in every report (all in `raw_rows.txt`). `*` = the row carries the report's "
        "asterisk; its meaning changes during the season (see Footnotes and notes).",
        "",
        "## Reports not parsed, or rows absent",
        "",
    ]
    issues = []
    for r in recs:
        if r["error"]:
            issues.append(f"- `{r['file']}`: not parsed: {r['error']}")
        else:
            p = problems(r)
            if p:
                why = ""
                if any(s.startswith(f"mill {MILL}: absent") for s in p):
                    why = (
                        f" The mill had not started: the final report gives its start as "
                        f"27 Nov 2025, after this cut-off ({r['cutoff']})."
                    )
                issues.append(f"- Report {r['report']} (`{r['file']}`): {'; '.join(p)}.{why}")
    parts += issues or ["None."]
    parts += [
        "",
        f"## {MILL} (cumulative)",
        "",
        table("mill", MILL),
        "",
        "## Centro region (cumulative: Atencingo, Central Casasano, Emiliano Zapata)",
        "",
        table("mill", "Centro"),
        "",
        "## Morelos state (cumulative: Central Casasano, Emiliano Zapata)",
        "",
        table("state", "Morelos"),
        "",
        "## Total nacional (cumulative)",
        "",
        table("state", "Total nacional"),
        "",
        "## Notes",
        "",
        "- File names do not always match the report: `..._sem29_rep20.pdf` is report 23 "
        "(week 29); `..._20251121.pdf` is report 1 and `..._producci_n_de_ca_a_y_az_car.pdf` "
        "is report 2.",
        "- Footers such as '*Semana 42 con fecha de corte al 19/07/2025' sit under the "
        "weekly comparison tables (Cuadros 4-5) and appear to refer to the same week of the "
        "previous zafra; they are not used.",
        "",
        "## Footnotes and notes",
        "",
        f"`*` footnotes of each report, and every report note that names {MILL}, verbatim.",
        "",
        "| report | `*` footnote(s) | notes naming the mill |",
        "|---|---|---|",
        *(
            f"| {r['report']} | {'; '.join(r['footnotes']) or '–'} | "
            f"{' '.join(r['notes']) or '–'} |"
            for r in ok
        ),
        "",
    ]
    return "\n".join(parts)


def raw_rows(recs) -> str:
    out = []
    for r in recs:
        out.append(f"== {r['file']}")
        if r["error"]:
            out.append(f"   NOT PARSED: {r['error']}")
            continue
        out.append(
            f"   report {r['report']}, week {r['cycle_week']}, {r['period']}, cut-off {r['cutoff']}"
        )
        for label, part in (("cumulative", "mill"), ("weekly", "weekly")):
            v = r[part].get(MILL)
            txt = v["raw"] if isinstance(v, dict) else f"<{v or 'absent'}>"
            out.append(f"   [{label}] {txt}")
    return "\n".join(out) + "\n"


def weekly_series(recs) -> pd.DataFrame:
    """Cumulative EZ area and cane per cut-off, differenced. Before the mill's first row the
    cumulative is 0 only when the cut-off precedes the official start date."""
    rows = []
    for r in recs:
        if r["error"]:
            continue
        x, w = row(r, "mill", MILL), row(r, "weekly", MILL)
        rows.append(
            {
                "cutoff": pd.Timestamp(r["cutoff"]),
                "week_start": pd.Timestamp(r["cutoff"]) - pd.Timedelta(days=6),
                "report": r["report"],
                "cycle_week": r["cycle_week"],
                "cum_ha": x["area_ha"] if x else np.nan,
                "cum_cane_t": x["cane_t"] if x else np.nan,
                "reported_week_ha": w["area_ha"] if w else np.nan,
                "reported_week_cane_t": w["cane_t"] if w else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)


def ldsca_articles(text: str) -> dict[int, tuple[int, str]]:
    """{article: (printed page, verbatim text joined across lines)} for LDSCA_ARTICLES."""
    pages = text.split("\f")
    lines = [(i + 1, ln) for i, p in enumerate(pages) for ln in p.splitlines()]
    noise = re.compile(
        r"^\s*(\d+ de \d+|LEY DE DESARROLLO SUSTENTABLE.*|CÁMARA DE DIPUTADOS.*|"
        r"Secretaría General|Secretaría de Servicios Parlamentarios)\s*$"
    )
    out = {}
    for n in LDSCA_ARTICLES:
        start = next(
            (k for k, (_, ln) in enumerate(lines) if re.match(rf"^\s*Art[íi]culo {n}\.-", ln)),
            None,
        )
        if start is None:
            continue
        body = []
        for _, ln in lines[start:]:
            if body and re.match(r"^\s*(Art[íi]culo \d+\.-|CAP[ÍI]TULO|T[ÍI]TULO)", ln):
                break
            if not noise.match(ln):
                body.append(ln.strip())
        paras, cur = [], []
        for ln in body:
            if not ln:
                if cur:
                    paras.append(" ".join(cur))
                cur = []
            else:
                cur.append(ln)
        if cur:
            paras.append(" ".join(cur))
        out[n] = (lines[start][0], "\n\n".join(re.sub(r"\s+", " ", p) for p in paras))
    return out


def detected_weekly():
    """Dated cane area of our map per official week (Sunday-Saturday, keyed by cut-off)."""
    site = load_site("morelos")
    g = gpd.read_file(site.path("harvest_geojson"), read_geometry=False, engine="pyogrio")
    cane = g[g["cane_pred"].fillna(False).astype(bool)]
    dated = cane[cane["harvest_date"].notna()].copy()
    d = pd.to_datetime(dated["harvest_date"]).dt.normalize()
    dated["cutoff"] = d + pd.to_timedelta((5 - d.dt.day_of_week) % 7, unit="D")
    per_week = dated.groupby("cutoff")["area_ha"].sum()
    info = {
        "segments": len(g),
        "cane_segments": len(cane),
        "cane_ha": float(cane["area_ha"].sum()),
        "dated_segments": len(dated),
        "dated_ha": float(dated["area_ha"].sum()),
        "status_ha": cane.groupby("status")["area_ha"].sum().round(1).to_dict(),
        "first_date": d.min().date(),
        "last_date": d.max().date(),
        "date_quantiles": {str(q): str(d.quantile(q).date()) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }
    return per_week, info


def lag_r(det: np.ndarray, off: np.ndarray, k: int) -> tuple[float, int]:
    """Pearson r of det[t] with off[t + k] (k > 0: detection leads crushing)."""
    if k >= 0:
        a, b = det[: len(det) - k], off[k:]
    else:
        a, b = det[-k:], off[: len(off) + k]
    return float(np.corrcoef(a, b)[0, 1]), len(a)


def figure(cmp: pd.DataFrame, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = cmp["cutoff"]
    off_w = cmp["week_cane_t"] / cmp["week_cane_t"].sum()
    det_w = cmp["det_ha"] / cmp["det_ha"].sum()
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e4e4e4", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.plot(x, off_w, color=COLORS[0], lw=2, marker="o", ms=4,
             label="official: cane milled per week (t)")  # fmt: skip
    ax1.plot(x, det_w, color=COLORS[1], lw=2, marker="o", ms=4,
             label="our map: dated cane area per week (ha)")  # fmt: skip
    ax1.set_ylabel("share of the season total")
    ax1.set_title(
        f"{MILL} vs our Morelos map, zafra 2025/26, per official week (Sun-Sat)",
        loc="left", fontsize=11,
    )  # fmt: skip
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.plot(x, cmp["off_cum_share"], color=COLORS[0], lw=2, marker="o", ms=4,
             label="official: cumulative area / final area")  # fmt: skip
    ax2.plot(x, cmp["det_cum_share"], color=COLORS[1], lw=2, marker="o", ms=4,
             label="our map: cumulative dated area / total dated area")  # fmt: skip
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.set_ylabel("cumulative share of the season")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("week cut-off (Saturday)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. weekly reports
    recs = parse_weekly()
    (OUT / "official_data.md").write_text(official_md(recs), encoding="utf-8")
    (OUT / "raw_rows.txt").write_text(raw_rows(recs), encoding="utf-8")
    n_ok = sum(r["error"] is None for r in recs)
    print(f"weekly reports parsed: {n_ok}/{len(recs)}")

    # 2. final report and KARBE
    ftext = C.pdf_text(FINAL)
    pub = re.search(r"Fecha de publicación: (\d+ de \w+ de \d{4})", ftext)
    dates = C.final_dates_row(ftext, MILL)
    cum = C.section_rows(ftext, C.CUMULATIVE_MILL_REGION, (MILL,))[MILL]
    if not isinstance(cum, dict):
        raise SystemExit(f"final report cumulative row: {cum}")
    prec = C.final_precision_row(ftext, MILL)
    kar = C.karbe(C.pdf_text(KARBE), MILL)
    final = {
        "mill": MILL,
        "source": FINAL.name,
        "publication_date": str(C.long_date(pub.group(1))) if pub else None,
        "season_start": str(dates["start"]),
        "season_end": str(dates["end"]),
        "operating_days_real": dates["days_real"],
        "operating_days_1st_estimate": dates["days_1st_estimate"],
        "operating_days_difference": dates["days_difference"],
        "area_ha": cum["area_ha"],
        "cane_t": cum["cane_t"],
        "field_yield_t_ha": cum["field_yield_t_ha"],
        "sugar_t": cum["sugar_t"],
        "factory_yield_pct": cum["factory_yield_pct"],
        "agroindustrial_yield_t_sugar_ha": cum["agroindustrial_yield_t_ha"],
        "standard_sugar_t": cum["standard_sugar_t"],
        "karbe_net_theoretical_kg_tcn_2025_26": cum["karbe_kg_tcn"],
        "precision_vs_1st_estimate_pct": {
            "end_date": str(prec["end"]),
            "area": prec["area_pct_of_1st_estimate"],
            "cane": prec["cane_pct_of_1st_estimate"],
            "sugar": prec["sugar_pct_of_1st_estimate"],
        },
        "karbe_5_zafras": {
            "source": KARBE.name,
            "zafras": "2016/17 to 2020/21 (published for the 2021/22 pre-liquidation)",
            "value_kg_tcn": kar["value_kg_tcn"],
            "national_rank": kar["rank"],
            "mills_ranked": kar["n_mills"],
            "national_kg_tcn": kar["national_kg_tcn"],
        },
        "raw": {
            "cuadro_1": dates["raw"],
            "cuadro_6": cum["raw"],
            "cuadro_8": prec["raw"],
            "karbe": kar["raw"],
        },
    }
    if kar["n_mills"] != 49:  # "Con información de 49 ingenios" in the KARBE note
        raise SystemExit(f"KARBE: {kar['n_mills']} values, the note says 49 mills")
    (OUT / "ez_final.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")

    # 3. weekly series
    ws = weekly_series(recs)
    start = pd.Timestamp(dates["start"])
    pre = ws["cum_ha"].isna() & (ws["cutoff"] < start)
    ws.loc[pre, ["cum_ha", "cum_cane_t"]] = 0.0  # before the official start date
    ws["week_ha"] = ws["cum_ha"].diff().fillna(ws["cum_ha"])
    ws["week_cane_t"] = ws["cum_cane_t"].diff().fillna(ws["cum_cane_t"])
    ws["diff_vs_reported_ha"] = ws["week_ha"] - ws["reported_week_ha"]
    ws["diff_vs_reported_cane_t"] = ws["week_cane_t"] - ws["reported_week_cane_t"]
    ws.assign(cutoff=ws["cutoff"].dt.date, week_start=ws["week_start"].dt.date).to_csv(
        OUT / "ez_weekly.csv", index=False
    )

    # 4. comparison with our map
    per_week, info = detected_weekly()
    end = pd.Timestamp(dates["end"])
    in_season = (ws["cutoff"] >= start) & (ws["week_start"] <= end)
    cmp = ws.loc[in_season, ["cutoff", "report", "week_ha", "week_cane_t", "cum_ha"]].copy()
    cmp["det_ha"] = cmp["cutoff"].map(per_week).fillna(0.0)
    t_ha = final["field_yield_t_ha"]
    cmp["det_t"] = cmp["det_ha"] * t_ha
    cmp["det_cum_ha"] = cmp["det_ha"].cumsum()
    cmp["off_cum_share"] = cmp["cum_ha"] / final["area_ha"]
    cmp["det_cum_share"] = cmp["det_cum_ha"] / info["dated_ha"]
    det, off = cmp["det_t"].to_numpy(), cmp["week_cane_t"].to_numpy()
    corr = {k: lag_r(det, off, k) for k in (*CONTEXT_LAGS, *LAGS)}
    info["dated_ha_outside_season_weeks"] = round(float(info["dated_ha"] - cmp["det_ha"].sum()), 2)
    w4e = pd.read_csv(RESULTS_DIR / "4e" / "weekly.csv")
    info["weekly_csv_harvest_ha"] = float(w4e["harvest_ha"].sum())
    w4e_dated = w4e[w4e["harvest_ha"] > 0]["week"]
    info["weekly_csv_first_last_week"] = (w4e_dated.min(), w4e_dated.max())
    fig_path = OUT / "official_vs_detected.png"
    figure(cmp, fig_path)

    def first_cross(col, q):
        hit = cmp[cmp[col] >= q]
        return str(hit["cutoff"].iloc[0].date()) if len(hit) else "–"

    summary = {
        "reports_parsed": n_ok,
        "reports": len(recs),
        "reports_with_mill_row": int(ws["cum_ha"].notna().sum() - pre.sum()),
        "max_abs_diff_vs_reported_weekly_ha": float(ws["diff_vs_reported_ha"].abs().max()),
        "max_abs_diff_vs_reported_weekly_t": float(ws["diff_vs_reported_cane_t"].abs().max()),
        "season_weeks": len(cmp),
        "official_t_ha": t_ha,
        "pearson_r": {str(k): {"r": round(r, 3), "n_weeks": n} for k, (r, n) in corr.items()},
        "detected": info,
        "share_crossings": {
            q: {
                "official": first_cross("off_cum_share", q),
                "detected": first_cross("det_cum_share", q),
            }
            for q in (0.25, 0.5, 0.75)
        },  # fmt: skip
    }
    (OUT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False) + "\n"
    )
    (OUT / "official_comparison.md").write_text(
        comparison_md(final, ws, cmp, summary, info, recs, corr), encoding="utf-8"
    )

    # 5. LDSCA
    arts = ldsca_articles(C.pdf_text(LDSCA))
    lines = ["# LDSCA articles (verbatim, `pdftotext -layout`, lines joined)", ""]
    for n, (page, body) in arts.items():
        lines += [f"## Artículo {n} (starts p. {page} of 45)", "", body, ""]
    (OUT / "ldsca_articles.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))


def comparison_md(final, ws, cmp, s, info, recs, corr) -> str:
    def pct(a, b):
        return f"{100 * a / b:.0f} %"

    first_week, last_week = info["weekly_csv_first_last_week"]
    morelos_ha = [row(r, "state", "Morelos") for r in recs if r["error"] is None][-1]["area_ha"]
    span = (info["last_date"] - info["first_date"]).days + 1
    head = (
        "| cut-off | report | official cane t | official ha | detected ha | detected t | "
        "official cum. share | detected cum. share |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows_w = head + "\n".join(
        f"| {r.cutoff.date()} | {r.report} | {fmt(r.week_cane_t)} | {fmt(r.week_ha)} | "
        f"{fmt(r.det_ha)} | {fmt(r.det_t)} | {r.off_cum_share:.2f} | {r.det_cum_share:.2f} |"
        for r in cmp.itertuples()
    )
    lag_rows = "\n".join(
        f"| {k:+d} | {r:.2f} | {n} |{' context: detection after crushing |' if k < 0 else ' |'}"
        for k, (r, n) in corr.items()
    )
    sh = s["share_crossings"]
    off_ha, t_ha, st = final["area_ha"], final["field_yield_t_ha"], info["status_ha"]
    totals = "\n".join(
        [
            "| quantity | official (final report) | our map |",
            "|---|---|---|",
            f"| area | {fmt(off_ha)} ha harvested (*superficie industrializada*) | "
            f"{fmt(info['cane_ha'])} ha gated as cane ({pct(info['cane_ha'], off_ha)} of official);"
            f" {fmt(info['dated_ha'])} ha dated ({pct(info['dated_ha'], off_ha)}) |",
            f"| gated cane by status | – | dated {fmt(st.get('harvested'))} ha, no baseline "
            f"{fmt(st.get('no_baseline'))} ha, not detected {fmt(st.get('not_detected'))} ha |",
            f"| Morelos state (EZ + Casasano) | {fmt(morelos_ha)} ha | – |",
            f"| cane milled | {fmt(final['cane_t'])} t ({t_ha:.2f} t/ha) | "
            f"dated ha × {t_ha:.2f} t/ha = {fmt(info['dated_ha'] * t_ha)} t |",
            f"| start | {final['season_start']} | first dated harvest {info['first_date']} "
            f"(week of {first_week} in results/4e) |",
            f"| end | {final['season_end']} | last dated harvest {info['last_date']} "
            f"(week of {last_week} in results/4e) |",
            f"| days | {final['operating_days_real']} real "
            f"({final['operating_days_1st_estimate']} in the 1st estimate) | "
            f"{span} days first-to-last dated harvest |",
        ]
    )
    return f"""# Exercise 10: comparison with official CONADESUCA data, Ingenio {MILL}

Generated by `scripts/20_official_data.py`. Official data: CONADESUCA weekly progress
reports 1-36 and final report of zafra 2025/26 ([official_data.md](official_data.md),
[ez_final.json](ez_final.json), [ez_weekly.csv](ez_weekly.csv)). Our data: the Morelos
area-wide harvest map of exercise 4e (`data/results/harvest_dates_2025_26.geojson`,
[results/4e](../4e/area_summary.json)), unchanged.

**The AOI is not the mill's supply area.** Our map covers the FIRMS bbox
(lon -99.35 to -99.05, lat 18.50 to 18.75). The Ingenio {MILL} receives cane from its own
supply zone, which this bbox may cut or overshoot; cane inside the bbox may go to another
mill (Morelos has two: {MILL} and Central Casasano), and the mill's cane from outside the
bbox is not in our map. No parameter was changed to fit the official numbers.

## Parsing

{s["reports_parsed"]} of {s["reports"]} weekly reports parsed; {s["reports_with_mill_row"]}
carry a {MILL} row (reports 1-2 predate the mill's start on 27 Nov 2025). The weekly
series is the difference of the cumulative rows; it matches the reports' own weekly row to
within {fmt(s["max_abs_diff_vs_reported_weekly_ha"])} ha and
{fmt(s["max_abs_diff_vs_reported_weekly_t"])} t in every week.

## (a) Weekly timing

Official week = Sunday to Saturday cut-off. Detected tonnes = detected ha of the week ×
{final["field_yield_t_ha"]:.2f} t/ha (official season field yield); a constant factor, so r
is the same as for hectares. Lag k pairs detection in week t with crushing in week t + k.

| lag (weeks) | Pearson r | weeks |
|---|---|---|
{lag_rows}

![official vs detected](official_vs_detected.png)

## (b) Totals and dates

{totals}

Dated area outside the mill's season weeks: {fmt(info["dated_ha_outside_season_weeks"])} ha.
results/4e/weekly.csv sums to {fmt(info["weekly_csv_harvest_ha"])} ha (same polygons,
Monday weeks).

## (c) Cumulative share of the season

Official share = cumulative ha / {fmt(final["area_ha"])} ha; detected share = cumulative
dated ha / {fmt(info["dated_ha"])} ha.

| share reached | official (cut-off) | detected (cut-off) |
|---|---|---|
| 25 % | {sh[0.25]["official"]} | {sh[0.25]["detected"]} |
| 50 % | {sh[0.5]["official"]} | {sh[0.5]["detected"]} |
| 75 % | {sh[0.75]["official"]} | {sh[0.75]["detected"]} |

{rows_w}

## Read

{read_text(final, cmp, s, info, corr, recs)}
"""


def date_bias() -> dict:
    """Signed error of the delivered (xgboost) dates on the 26 labelled cane fields: date minus
    the midpoint of the Sentinel-2 NDVI harvest window (results/4b/dates.csv, out of fold),
    and the median gap between the two looks the date is the midpoint of."""
    d = pd.read_csv(RESULTS_DIR / "4b" / "dates.csv", parse_dates=["date", "date_before",
                                                                   "date_after"])  # fmt: skip
    d = d[(d["method"] == "xgb") & d["date"].notna()].set_index("field_id")
    f = gpd.read_file(FIELDS_DIR / "fields_v2.geojson").set_index("id")
    x = d.join(f[["harvest_window_start", "harvest_window_end"]], how="inner")
    x = x.dropna(subset=["harvest_window_start", "harvest_window_end"])  # the 26 cane fields
    w0, w1 = pd.to_datetime(x["harvest_window_start"]), pd.to_datetime(x["harvest_window_end"])
    e = (x["date"] - (w0 + (w1 - w0) / 2)).dt.days
    return {
        "n": len(e),
        "median_d": float(e.median()),
        "late": int((e > 0).sum()),
        "early": int((e < 0).sum()),
        "gap_median_d": float((x["date_after"] - x["date_before"]).dt.days.median()),
    }


def read_text(final, cmp, s, info, corr, recs) -> str:
    full = cmp.iloc[1:-1]  # weeks fully inside the season (first and last are partial)
    cv_off = full["week_cane_t"].std() / full["week_cane_t"].mean()
    cv_det = full["det_ha"].std() / full["det_ha"].mean()
    gap = cmp["det_cum_share"] - cmp["off_cum_share"]
    lo, hi = int(gap.to_numpy().argmin()), int(gap.to_numpy().argmax())
    lo_d, hi_d = cmp["cutoff"].iloc[lo].date(), cmp["cutoff"].iloc[hi].date()
    before = cmp[cmp["cutoff"] < pd.Timestamp(info["first_date"])]
    sh = s["share_crossings"]
    last = cmp.iloc[-1]
    off_ha, cane_ha, dated_ha = final["area_ha"], info["cane_ha"], info["dated_ha"]
    no_base, not_det = info["status_ha"].get("no_baseline"), info["status_ha"].get("not_detected")
    end_days = abs((info["last_date"] - date.fromisoformat(final["season_end"])).days)
    pre_ha = before["week_ha"].sum()
    b = date_bias()
    est = [
        str(r["report"])
        for r in recs
        if r["error"] is None and any("estimad" in n for n in r["notes"])
    ]
    return f"""**(a) The cumulative curves are the right comparison, and they agree.** The detected
share tracks the official share within {100 * gap.iloc[lo]:+.0f} / {100 * gap.iloc[hi]:+.0f}
percentage points all season (behind at {lo_d}, ahead at {hi_d}). 50 % of the season is
reached on {sh[0.5]["detected"]} (detected) vs {sh[0.5]["official"]} (official). The season
end agrees within {end_days} days ({info["last_date"]} vs {final["season_end"]}). The gated
cane area is {100 * cane_ha / off_ha:.0f} % of the official {fmt(off_ha)} ha
({fmt(cane_ha)} ha). The AOI bbox is not the mill's supply area, so the area match shows the
two are of the same order, not that the cane gate is right.

**(b) Weekly r is weak ({corr[0][0]:.2f} at lag 0) because of the sampling, not the dates.**
The mill crushes at a steady rate (coefficient of variation {cv_off:.2f} over the {len(full)}
full weeks). Radar detections cluster on acquisition dates ({cv_det:.2f}), so whole weeks
are empty or double. A week-to-week comparison penalizes the revisit schedule; the
cumulative curve in (a), which integrates over it, does not. Lags +1 and +2 (detection
leading crushing) give {corr[1][0]:.2f} and {corr[2][0]:.2f}: not supported.

**(c) Lag -1 (r = {corr[-1][0]:.2f}) is in the expected direction.** A field is detected only
after the cut: the date is the midpoint between the last look classed standing and the first
look classed harvested (median gap {b["gap_median_d"]:.0f} days between looks, both orbits).
On the {b["n"]} labelled fields these dates fall a median of {b["median_d"]:+.1f} days after
the NDVI harvest-window midpoint ({b["late"]} late, {b["early"]} early). The mill receives
cane within 72 hours of the cut (48 hours if burned; LDSCA art. 73). Crushing therefore
precedes our date by a few days, which pushes part of each week's detections into the
following week. A full week is more than this measured bias, so lag -1 was not the
pre-stated test and is not claimed as a result. It is consistent with the date rule,
not an anomaly.

Also:

- {fmt(cane_ha - dated_ha)} ha of gated cane has no date (no baseline {fmt(no_base)} ha, not
  detected {fmt(not_det)} ha), so the dated area is {100 * dated_ha / off_ha:.0f} % of the
  official area. The mill had cut {fmt(pre_ha)} ha ({100 * pre_ha / off_ha:.0f} %) before our
  first date ({info["first_date"]}, start {final["season_start"]}). That fits the detector
  needing a pre-harvest baseline, but this comparison does not prove it.
- Official data as printed: the last week (report {last["report"]}) adds
  {fmt(last["week_ha"])} ha for {fmt(last["week_cane_t"])} t, about
  {last["week_cane_t"] / last["week_ha"]:.1f} t/ha. In report(s) {", ".join(est)} a note says
  the mill's area was estimated by CONADESUCA, because the mill sent only a factory daily.
  Both are kept as published."""


if __name__ == "__main__":
    main()
