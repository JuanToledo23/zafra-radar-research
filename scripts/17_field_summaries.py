"""Exercise 6: per-field facts, alert rules and plain-language summaries.

    python scripts/17_field_summaries.py [--site morelos] [--n-sample 20] [--seed 0]

Inputs (scripts 14-15 run with the same --site): the delivered harvest layer
(paths.harvest_geojson, or paths.merged_geojson if results merge_summary.json says the
merged layer is delivered), the radar series (paths.timeseries), the cane features
(paths.cane_features, for coh_min_date) and the FIRMS CSVs. For Morelos, the 38
reference_fields and their field-level outputs (below).

1. Facts (zafra.summary.field_facts) and alerts (zafra.alerts.evaluate) for every gated
   cane segment of the delivered layer: counts per alert code, share of cane segments with
   at least one alert and with at least one warning.
2. Summaries for:
   * p36 and the 38 reference fields (fields_v2), with facts from the field-level outputs,
     not from a segment:
     - date: the 4b xgboost date (results/4b/dates.csv). The status comes from the
       delivered 4e baseline rule (model.dates_with_baseline) applied to the 4b
       out-of-fold (cane) and full-fit (non-cane) predictions;
     - gate: 4c leave-one-field-out P (results/4c/loo_predictions.csv); the site's
       gate_variant column is p_cane, radar (A) is p_cane_radar, optical (B) p_cane_optical;
     - fires data/fires/fields_v2_fires.csv, radar series fields_v2_features.parquet,
       coh_min / coh_min_date / hv_drop from fields_v2_cane_features.parquet, GPM rain
       data/features/gpm_rain_3d.csv;
     - segments_covering: the delivered segments overlapping the field by >= one S2 pixel
       (0.01 ha), their count and the share of the field's area they cover;
   * a seeded random sample of gated cane segments;
   * if none of the above has one, a segment with FIRE_ACTIVE_OUTSIDE_HOURS, and one with
     HIGH_DATE_UNCERTAINTY (or, as none exceeds the limit, the largest uncertainty).
   Each gets the template summary and the LLM summary (template fallback without
   ANTHROPIC_API_KEY), both in Spanish and English, with the verifier result.
3. results/6/ (other sites: results/6_<site>/): summaries.jsonl (one record per polygon),
   summaries.md (readable), alert_counts.json, with the API cost per 1,000 summaries from
   the token counts (response.usage when the API ran, otherwise estimated from the prompt
   length).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from zafra import alerts as A
from zafra import fires as F
from zafra import summary as SM
from zafra.detect import cane_gate as G
from zafra.detect import model as M
from zafra.paths import FEATURES_DIR, FIRES_DIR, RESULTS_DIR
from zafra.site import add_site_argument, load_site

RAIN_CACHE = FEATURES_DIR / "gpm_rain_3d.csv"
P36 = "p36"
# field-level outputs of exercises 4b / 4c (Morelos reference fields)
DATES_4B = RESULTS_DIR / "4b" / "dates.csv"
OOF_4B = RESULTS_DIR / "4b" / "oof_predictions.csv"
NONCANE_4B = RESULTS_DIR / "4b" / "noncane_predictions.csv"
GATE_4C = RESULTS_DIR / "4c" / "loo_predictions.csv"
FIELD_FEATURES = FEATURES_DIR / "fields_v2_features.parquet"
FIELD_CANE_FEATURES = FEATURES_DIR / "fields_v2_cane_features.parquet"
FIELD_FIRES = FIRES_DIR / "fields_v2_fires.csv"
DETECTOR = "xgb"  # script 14's default detector
MIN_PIXEL_HA = 0.01  # one 10 m Sentinel-2 pixel: smaller overlaps are tracing slivers


def log(msg):
    print(msg, flush=True)


def delivered_layer(site) -> Path:
    ms = site.path("results_dir") / "merge_summary.json"
    merged = ms.exists() and json.loads(ms.read_text()).get("delivered") == "merged"
    return site.path("merged_geojson" if merged else "harvest_geojson")


def segments_covering(fields: gpd.GeoDataFrame, seg: gpd.GeoDataFrame, crs) -> dict:
    """Per reference field: delivered segments overlapping it by >= MIN_PIXEL_HA, their
    count and the share of the field's area they cover."""
    f = fields[["id", "geometry"]].to_crs(crs)
    s = seg[["seg_id", "geometry"]].to_crs(crs)
    j = gpd.overlay(f, s, how="intersection", keep_geom_type=True)
    j["ha"] = j.area / 1e4
    j = j[j["ha"] >= MIN_PIXEL_HA]
    fa = (f.set_index("id").area / 1e4).to_dict()
    g = j.groupby("id")["ha"]
    return {
        fid: {
            "count": int(g.size().get(fid, 0)),
            "share_of_area": round(float(g.sum().get(fid, 0.0)) / fa[fid], 3),
        }
        for fid in f["id"]
    }


def reference_field_facts(fields, seg, site, rain) -> dict:
    """facts + alerts per reference field, from the field-level outputs (see docstring)."""
    dates = pd.read_csv(DATES_4B)
    dates = dates[dates["method"] == DETECTOR].set_index("field_id")
    rows = pd.concat([pd.read_csv(OOF_4B), pd.read_csv(NONCANE_4B)], ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["acq_utc"] = pd.to_datetime(rows["acq_utc"], utc=True)
    status = M.dates_with_baseline(rows, f"p_{DETECTOR}").set_index("field_id")
    gate = pd.read_csv(GATE_4C).set_index("field_id")
    cf = pd.read_parquet(FIELD_CANE_FEATURES)
    cf = cf[cf["zafra"] == site.zafra].set_index("field_id")
    acq = pd.read_parquet(FIELD_FEATURES, columns=["field_id", "date", "orbit", "hv_db"])
    acq_by = dict(tuple(acq.groupby("field_id")))
    fires = pd.read_csv(FIELD_FIRES)
    fires_by = dict(tuple(fires.groupby("field_id")))
    cover = segments_covering(fields, seg, site.metric_crs)
    v = site.gate_variant
    out, mismatched = {}, []
    for _, fl in fields.iterrows():
        fid = fl["id"]
        g, d, st = gate.loc[fid], dates.loc[fid], status.loc[fid]
        p_cane = float(g[f"p_{v}"])
        cane = bool(g[f"pred_{v}"])
        raw = st["status"] if cane else "not_cane"
        dated = raw == M.STATUS_HARVESTED
        if dated and str(pd.Timestamp(st["date"]).date()) != str(pd.Timestamp(d["date"]).date()):
            mismatched.append(fid)
        before, after = pd.Timestamp(d["date_before"]), pd.Timestamp(d["date_after"])
        mu = fl.get("municipio")
        row = {
            "seg_id": None,
            "field_id": fid,
            "municipality": None if pd.isna(mu) else mu,
            "area_ha": fl["area_ha"],
            "cane_pred": cane,
            "p_cane": p_cane,
            "p_cane_radar": g["p_A"],
            "p_cane_optical": g["p_B"],
            "status": raw,
            "harvest_date": pd.Timestamp(d["date"]).strftime("%Y-%m-%d") if dated else None,
            "harvest_after": after if dated else None,
            "date_uncertainty_days": (after.normalize() - before.normalize()).days / 2
            if dated
            else None,
            "hv_drop_db": cf.loc[fid, "hv_max_single_drop_db"],
            "coh_min": cf.loc[fid, "coh_min"],
            "wc_class": None,
            "wc_disagrees": None,
            "gate_confidence": G.confidence([p_cane])[0],
            "segments_covering": cover[fid],
        }
        series = {
            "acq": acq_by.get(fid),
            "fires": fires_by.get(fid),
            "coh_min_date": cf.loc[fid, "coh_min_date"],
            "rain": rain.get(fid) if rain is not None else None,
        }
        facts = SM.field_facts(row, series, site)
        out[fid] = (facts, A.evaluate(facts, site.alerts))
    if mismatched:
        raise RuntimeError(f"4e baseline date differs from the 4b date for {mismatched}")
    return out


def facts_table(seg, ids, site, extra=None, rain=None):
    """facts + alerts for the segments `ids` (extra: per-id dict of added row keys)."""
    ids = list(dict.fromkeys(ids))
    sub = seg[seg["seg_id"].isin(ids)]
    ts = pd.read_parquet(
        site.path("timeseries"),
        columns=["field_id", "date", "orbit", "hv_db"],
        filters=[("field_id", "in", ids)],
    )
    ts_by = dict(tuple(ts.groupby("field_id")))
    cf = pd.read_parquet(site.path("cane_features"), columns=["field_id", "coh_min_date"])
    coh_date = cf.set_index("field_id")["coh_min_date"]
    buf = float(site.alerts.get("fire_buffer_m", 200))
    ft = F.fires_per_field_fast(
        F.load_firms(site.firms_dir), sub.rename(columns={"seg_id": "id"}), buf, crs=site.metric_crs
    )
    ft_by = dict(tuple(ft.groupby("field_id")))
    rows = sub.drop(columns="geometry").set_index("seg_id")
    out = {}
    for key, sid, add in extra or [(sid, sid, {}) for sid in ids]:
        row = rows.loc[sid].to_dict() | {"seg_id": sid} | add
        series = {
            "acq": ts_by.get(sid),
            "fires": ft_by.get(sid),
            "coh_min_date": coh_date.get(sid),
            "rain": rain.get(add.get("field_id")) if rain is not None else None,
        }
        facts = SM.field_facts(row, series, site)
        out[key] = (facts, A.evaluate(facts, site.alerts))
    return out


def alert_counts(table, seg) -> dict:
    n = len(table)
    conf = seg.set_index("seg_id")["confidence"]
    dated = [k for k, (f, _) in table.items() if f["status"] == "dated"]
    codes = {c: 0 for c in A.CODES}
    any_alert = any_warning = 0
    for _, al in table.values():
        any_alert += bool(al)
        any_warning += any(a["severity"] == "warning" for a in al)
        for a in al:
            codes[a["code"]] += 1
    return {
        "cane_segments": n,
        "per_code": codes,
        "per_code_share": {c: round(v / n, 4) for c, v in codes.items()},
        "segments_with_alert": any_alert,
        "share_with_alert": round(any_alert / n, 4),
        "segments_with_warning": any_warning,
        "share_with_warning": round(any_warning / n, 4),
        "status": pd.Series([f["status"] for f, _ in table.values()]).value_counts().to_dict(),
        # cane_confidence must reproduce the GeoJSON `confidence` column on dated segments
        "check_cane_confidence_vs_geojson": {
            "dated": len(dated),
            "agree": sum(table[k][0]["cane_confidence"] == conf.get(k) for k in dated),
        },
    }


def render(facts, alerts, site, client):
    mw = SM._settings(site)["max_words"]
    tpl = {lang: SM.template_summary(facts, alerts, lang, mw) for lang in SM.LANGS}
    tpl["verifier"] = {lang: SM.verify(tpl[lang], facts, alerts, lang, mw) for lang in SM.LANGS}
    llm = SM.llm_summaries(facts, alerts, site=site, client=client)
    prompt_in = SM.load_prompt(word_budget=SM._settings(site)["word_budget"]) + SM.llm_input(
        facts, alerts
    )
    est = {
        "input_tokens": SM.estimate_tokens(prompt_in),
        "output_tokens": SM.estimate_tokens(json.dumps({k: tpl[k] for k in SM.LANGS})),
    }
    return {"template": tpl, "llm": llm, "tokens_estimated": est}


def cost(records, site) -> dict:
    price = (site.summary or {}).get("price_usd_per_mtok", {"input": 0.10, "output": 0.50})
    real = [r["llm"]["usage"] for r in records if r["llm"]["usage"]]
    src = "response.usage" if real else "estimated (no API call made; ~3.5 chars per token)"
    use = real or [r["tokens_estimated"] for r in records]
    tin = float(np.mean([u["input_tokens"] for u in use]))
    tout = float(np.mean([u["output_tokens"] for u in use]))
    per_k = 1000 * (tin * price["input"] + tout * price["output"]) / 1e6
    return {
        "model": SM._settings(site)["model"],
        "token_source": src,
        "calls_with_usage": len(real),
        "mean_input_tokens": round(tin),
        "mean_output_tokens": round(tout),
        "price_usd_per_mtok": price,
        "usd_per_1000_summaries": round(per_k, 4),
        "note": "one call = both languages; adaptive thinking tokens, if any, bill as output",
    }


# --- report -------------------------------------------------------------------------------

FACT_ROWS = [
    "segment_id",
    "field_id",
    "municipality",
    "area_ha",
    "p_cane",
    "p_cane_radar",
    "gate_variant",
    "status",
    "harvest_date",
    "date_uncertainty_days",
    "last_standing_date",
    "first_harvested_date",
    "hv_drop_db",
    "coh_min",
    "coh_min_date",
    "rain_3d_mm_at_harvest",
    "wc_class",
    "wc_disagrees",
    "cane_confidence",
    "date_precision_days",
    "acquisitions_used",
    "segments_covering",
]


def _cell(v):
    return "–" if v is None else str(v).replace("|", "/")


def record_md(r) -> list[str]:
    f = r["facts"]
    head = f"### {r['label']}"
    lines = [head, "", f"Tags: {', '.join(r['tags'])}", ""]
    lines += ["| fact | value |", "|---|---|"]
    lines += [f"| {k} | {_cell(f.get(k))} |" for k in FACT_ROWS]
    fires = f["fires"]
    fire_txt = "; ".join(
        f"{x['datetime_local']} {x['sensor']} FRP {x['frp']} {x['location']} {x['overpass']}"
        for x in fires[:5]
    )
    lines.append(f"| fires ({len(fires)}) | {_cell(fire_txt or None)} |")
    codes = ", ".join(f"{a['code']} ({a['severity']})" for a in r["alerts"]) or "none"
    lines += ["", f"Alerts: {codes}", ""]
    t, llm = r["template"], r["llm"]
    for name, rec in (("template", t), ("LLM path", llm)):
        if name == "LLM path":
            used = rec["renderer"]
            why = f", fallback_reason `{rec['fallback_reason']}`" if rec["fallback_reason"] else ""
            name = f"LLM path (renderer used: {used}{why})"
        for lang in SM.LANGS:
            v = rec["verifier"][lang]
            ok = "pass" if v["ok"] else "FAIL: " + "; ".join(v["problems"])
            lines.append(f"- **{name}, {lang}** (verifier {ok}): {rec[lang]}")
    return [*lines, ""]


def write_md(path, records, counts, cst, site, layer):
    ver = [r["template"]["verifier"][lang]["ok"] for r in records for lang in SM.LANGS]
    llm_used = sum(r["llm"]["renderer"] == "llm" for r in records)
    reasons = pd.Series([r["llm"]["fallback_reason"] for r in records]).value_counts().to_dict()
    pc = counts["per_code"]
    lines = [
        "# Exercise 6: field summaries and alerts",
        "",
        f"Generated by `scripts/17_field_summaries.py --site {site.name}` from `{layer}`. "
        "The summaries are generated text built only from the facts table above each one; "
        "they are not a separate analysis.",
        "",
        "## Alerts over the gated cane segments",
        "",
        f"{counts['cane_segments']} gated cane segments; {counts['segments_with_alert']} "
        f"({counts['share_with_alert']:.1%}) have at least one alert, "
        f"{counts['segments_with_warning']} ({counts['share_with_warning']:.1%}) at least one "
        "warning (info-only alerts excluded).",
        "",
        "| code | severity | segments | share |",
        "|---|---|---|---|",
    ]
    rules = site.alerts.get("rules", {})
    for c in A.CODES:
        p = rules.get(c, {})
        sev = p.get("severity", "–") if p.get("enabled") else "disabled"
        lines.append(f"| {c} | {sev} | {pc[c]} | {counts['per_code_share'][c]:.1%} |")
    chk = counts["check_cane_confidence_vs_geojson"]
    lines += [
        "",
        f"Check: `cane_confidence` equals the GeoJSON `confidence` column on {chk['agree']} of "
        f"{chk['dated']} dated cane segments.",
        "",
        "## Renderers",
        "",
        f"- Template: verifier pass {sum(ver)}/{len(ver)} texts.",
        f"- LLM ({cst['model']}): used for {llm_used}/{len(records)} polygons; fallback "
        f"reasons: {reasons}.",
        f"- Cost per 1,000 summaries: ${cst['usd_per_1000_summaries']} "
        f"({cst['mean_input_tokens']} input + {cst['mean_output_tokens']} output tokens per "
        f"call, {cst['token_source']}).",
        "",
        "## Polygons",
        "",
    ]
    for r in records:
        lines += record_md(r)
    path.write_text("\n".join(lines))


# --- main ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_site_argument(ap)
    ap.add_argument("--n-sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    site = load_site(args.site)
    global OUT_DIR
    OUT_DIR = RESULTS_DIR / ("6" if site.name == "morelos" else f"6_{site.name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    layer = delivered_layer(site)
    seg = gpd.read_file(layer)
    is_cane = seg["cane_pred"].fillna(False).astype(bool)
    cane_ids = sorted(seg.loc[is_cane, "seg_id"])
    log(f"{len(seg)} segments, {len(cane_ids)} gated cane, from {layer}")

    # 1. alerts over all cane segments
    table = facts_table(seg, cane_ids, site)
    counts = alert_counts(table, seg)
    log(json.dumps(counts, indent=1))

    # 2. polygons to summarise: (key, facts, alerts, tags)
    picks = []
    if site.reference_fields is not None and DATES_4B.exists() and GATE_4C.exists():
        fields = gpd.read_file(site.reference_fields)
        rain = (
            dict(tuple(pd.read_csv(RAIN_CACHE).groupby("field_id")))
            if RAIN_CACHE.exists()
            else None
        )
        ref = reference_field_facts(fields, seg, site, rain)
        lab = fields.set_index("id")["label"]
        for fid in [P36] + [f for f in fields["id"] if f != P36]:
            tags = (["p36"] if fid == P36 else []) + [f"reference field ({lab[fid]})"]
            picks.append((fid, *ref[fid], tags))
    rng = np.random.default_rng(args.seed)
    for sid in sorted(rng.choice(cane_ids, size=args.n_sample, replace=False)):
        picks.append((sid, *table[sid], [f"random cane sample (seed {args.seed})"]))
    have = {a["code"] for p in picks for a in p[2]}
    shown = {p[0] for p in picks}
    if "FIRE_ACTIVE_OUTSIDE_HOURS" not in have:
        code = "FIRE_ACTIVE_OUTSIDE_HOURS"
        ex = next((k for k in cane_ids if any(a["code"] == code for a in table[k][1])), None)
        if ex is not None:
            picks.append((ex, *table[ex], [f"example: {code}"]))
    if "HIGH_DATE_UNCERTAINTY" not in have:
        u = {k: f["date_uncertainty_days"] for k, (f, _) in table.items() if f["status"] == "dated"}
        hit = [
            k for k in cane_ids if any(a["code"] == "HIGH_DATE_UNCERTAINTY" for a in table[k][1])
        ]
        top = max([p[1]["date_uncertainty_days"] or 0 for p in picks], default=0)
        if hit:
            picks.append((hit[0], *table[hit[0]], ["example: HIGH_DATE_UNCERTAINTY"]))
        elif u and max(u.values()) > top:
            ex = max(sorted(u), key=u.get)
            tag = "example: largest date_uncertainty_days (HIGH_DATE_UNCERTAINTY fires on none)"
            if ex not in shown:
                picks.append((ex, *table[ex], [tag]))

    client, reason = SM.make_client()
    log(f"LLM client: {'ready' if client else reason}")
    records = []
    for key, facts, al, tags in picks:
        label = (
            f"{key} (reference field, field-level outputs)"
            if facts["field_id"]
            else f"segment {key}"
        )
        rec = {"label": label, "key": key, "tags": tags, "facts": facts, "alerts": al}
        records.append(rec | render(facts, al, site, client))

    cst = cost(records, site)
    with (OUT_DIR / "summaries.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    ver_t = [r["template"]["verifier"][lang]["ok"] for r in records for lang in SM.LANGS]
    llm_ok = [r["llm"]["renderer"] == "llm" for r in records]
    out = {
        "site": site.name,
        "layer": str(layer.relative_to(layer.parents[2])),
        "alerts": counts,
        "summaries": len(records),
        "template_verifier_pass": f"{sum(ver_t)}/{len(ver_t)}",
        "llm_used": f"{sum(llm_ok)}/{len(llm_ok)}",
        "llm_fallback_reasons": pd.Series([r["llm"]["fallback_reason"] for r in records])
        .value_counts(dropna=False)
        .to_dict(),
        "cost": cst,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (OUT_DIR / "alert_counts.json").write_text(json.dumps(out, indent=2, default=str))
    write_md(OUT_DIR / "summaries.md", records, counts, cst, site, out["layer"])
    log(json.dumps({k: v for k, v in out.items() if k != "alerts"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
