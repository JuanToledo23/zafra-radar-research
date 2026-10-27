"""Per-field facts and plain-language summaries (exercise 6, in-product AI component 3).

    from zafra import summary as SM
    facts = SM.field_facts(row, series, site)       # row of the harvest GeoJSON, or a
                                                    # reference-field row (script 17)
    alerts = zafra.alerts.evaluate(facts, site.alerts)
    SM.template_summary(facts, alerts, "es")        # deterministic, always available
    SM.llm_summary(facts, alerts, "en", site=site)  # LLM + verifier, template fallback

1. Facts. `field_facts` copies one polygon's results out of the existing outputs. For a
   segment: the delivered harvest GeoJSON row (script 14), its radar series, its VIIRS
   detections and its cane features (coh_min_date). For a reference field (fields_v2),
   scripts/17 builds the same row from the field-level outputs instead (4b dates, 4c gate,
   field fires and coherence). Nothing is re-estimated. Two values are read off the
   date rule rather than stored: the last standing look is first_harvested_date minus
   twice date_uncertainty_days (the date is the midpoint of that gap, script 14), and
   the status is the GeoJSON status in three classes (STATUS).
   Confidence is two separate facts: cane_confidence (gate certainty x agreement of the
   other gate, script 14's rule; the GeoJSON keeps its combined `confidence` column) and
   date_precision_days (= date_uncertainty_days, the ± of the date).
2. Display values. Both renderers see the same facts formatted once per language
   (`display`): "Caña: confianza baja" / "Cane: low confidence", "Fecha: 12 feb 2026
   ±2.5 días" / "Date: Feb 12, 2026 ±2.5 days", the first fire and the fire count, the
   alert codes with their short messages. The LLM is given these strings, not the raw
   facts, so every number it may write exists verbatim in its input.
3. Renderers, render(facts, alerts, lang) -> str, <= max_words (70) words:
   * template_summary: fixed sentences;
   * llm_summary: one Messages API call returning {"es", "en"}, prompt in
     prompts/field_summary.md. The output must pass `verify` in both languages; if the
     check fails, the key is missing or the call fails, the template text is returned and
     the reason recorded (`llm_summaries` returns the full record).
4. Verifier (`verify`): required = the area, the cane line and the date line (if any) and
   every alert code; forbidden = any date, time, number or alert-like code that is not in
   that language's display values, a date in another format (ISO, the other language),
   confirmation / interpretation phrases (FORBIDDEN_PHRASES), more than max_words (70)
   words. The prompt asks for at most word_budget (55) words, so the limit has a margin.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

PROMPT_PATH = Path(__file__).parent / "prompts" / "field_summary.md"
LANGS = ("es", "en")
MAX_WORDS = 70  # verifier hard limit
WORD_BUDGET = 55  # what the prompt asks for
DEFAULT_MODEL = "claude-haiku-5-5"

# GeoJSON status -> summary status. harvested_before_coverage (series starts harvested)
# has no standing baseline either; the original value stays in status_detail.
STATUS = {
    "harvested": "dated",
    "no_baseline": "no_baseline",
    "harvested_before_coverage": "no_baseline",
    "not_detected": "no_harvest",
    "not_cane": "no_harvest",
}
MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CONFIDENCE = {
    "es": {"high": "alta", "medium": "media", "low": "baja", "uncertain": "incierta"},
    "en": {"high": "high", "medium": "medium", "low": "low", "uncertain": "uncertain"},
}
NO_DATE = {
    "es": {
        "no_baseline": "Fecha: sin fecha; hay señal de cosecha, pero no suficientes "
        "observaciones previas de la caña en pie",
        "no_harvest": "Fecha: sin fecha; no se detectó cosecha en las observaciones de la zafra",
    },
    "en": {
        "no_baseline": "Date: not dated; there is a harvest signal, but not enough earlier "
        "looks at standing cane",
        "no_harvest": "Date: not dated; no harvest was detected in this season's looks",
    },
}
WC_NAME = {"es": {10: "arbolado", 30: "pastizal"}, "en": {10: "tree cover", 30: "grassland"}}
OVERPASS = {"N": "night", "D": "day"}  # FIRMS daynight
ALERT_LABEL = {
    "es": {
        "FIRE_ACTIVE_OUTSIDE_HOURS": "fuego activo fuera del horario permitido",
        "BURN_WITHOUT_HARVEST": "fuego sin cosecha posterior",
        "HARVEST_WITHOUT_FIRE": "cosecha sin fuego previo, informativo",
        "HIGH_DATE_UNCERTAINTY": "fecha imprecisa",
        "RADAR_OPTICAL_DISAGREE": "el otro filtro no confirma caña",
        "NO_BASELINE": "sin observaciones previas suficientes",
    },
    "en": {
        "FIRE_ACTIVE_OUTSIDE_HOURS": "fire active outside permitted hours",
        "BURN_WITHOUT_HARVEST": "fire with no harvest after it",
        "HARVEST_WITHOUT_FIRE": "harvest with no fire before it, informational",
        "HIGH_DATE_UNCERTAINTY": "imprecise date",
        "RADAR_OPTICAL_DISAGREE": "the other gate does not confirm cane",
        "NO_BASELINE": "not enough earlier looks",
    },
}


# --- formatting ---------------------------------------------------------------------------


def fmt_date(d, lang: str) -> str:
    """12 feb 2026 (es) / Feb 12, 2026 (en)."""
    t = pd.Timestamp(d)
    if lang == "es":
        return f"{t.day} {MONTHS_ES[t.month - 1]} {t.year}"
    return f"{MONTHS_EN[t.month - 1]} {t.day}, {t.year}"


def fmt_num(x, nd: int = 1) -> str:
    """Rounded to `nd` decimals, trailing zeros dropped: 6.0 -> 6, 2.50 -> 2.5."""
    s = f"{float(x):.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _none(x):
    """NaN / NaT / pd.NA -> None, numpy scalars -> Python."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x.item() if hasattr(x, "item") else x


def _day(x) -> str | None:
    x = _none(x)
    return None if x is None else pd.Timestamp(x).strftime("%Y-%m-%d")


def _round(x, nd):
    x = _none(x)
    return None if x is None else round(float(x), nd)


# --- 1. facts -----------------------------------------------------------------------------


def field_facts(row, series: dict, site) -> dict:
    """The facts of one polygon, all read from existing outputs.

    row: one row of the site's harvest GeoJSON (dict or Series; scripts 14/15 columns), or
      a reference-field row with the same columns, seg_id None and the extra keys field_id,
      municipality, segments_covering ({count, share_of_area} of delivered segments).
    series: that polygon's inputs:
      acq   radar rows (date, orbit, hv_db) of segments_timeseries.parquet (fields:
            fields_v2_features.parquet);
      fires VIIRS rows of fires.fires_per_field(_fast) (acq_utc, satellite, daynight, frp,
            dist_edge_m);
      coh_min_date  from the cane-feature table (the pair with the lowest mean coherence);
      rain  optional (date, rain_3d_mm) GPM rows, only cached for the labelled fields.
    site: zafra.site.Site (season, gate variant, alerts.timezone / fire_buffer_m).
    """
    get = row.get
    tz = site.alerts.get("timezone", "UTC")
    buf = float(site.alerts.get("fire_buffer_m", 200))
    raw_status = str(get("status"))
    status = STATUS[raw_status]
    dated = status == "dated"
    unc = _round(get("date_uncertainty_days"), 1) if dated else None
    first_h = _day(get("harvest_after")) if dated else None
    last_s = (
        (pd.Timestamp(first_h) - pd.Timedelta(days=2 * unc)).strftime("%Y-%m-%d")
        if dated and first_h and unc is not None
        else None
    )

    fires = []
    f = series.get("fires")
    if f is not None and len(f):
        t = pd.to_datetime(f["acq_utc"], utc=True)
        lo, hi = (s.tz_localize("UTC") for s in site.season)
        f = f.assign(_t=t)[(t >= lo) & (t < hi) & (f["dist_edge_m"] <= buf)]
        for _, r in f.sort_values("_t").iterrows():
            fires.append(
                {
                    "datetime_local": r["_t"].tz_convert(tz).strftime("%Y-%m-%d %H:%M"),
                    "sensor": f"VIIRS {r['satellite']}",
                    "frp": _round(r["frp"], 1),
                    "location": "inside_polygon" if r["dist_edge_m"] <= 0 else "within_buffer",
                    "overpass": OVERPASS.get(r.get("daynight")),
                }
            )

    rain = None
    rn = series.get("rain")
    if dated and rn is not None and len(rn):
        hit = rn[pd.to_datetime(rn["date"]).dt.strftime("%Y-%m-%d") == first_h]
        rain = _round(hit["rain_3d_mm"].mean(), 1) if len(hit) else None

    acq = series.get("acq")
    counts = {"asc": 0, "desc": 0}
    if acq is not None and len(acq):
        d = pd.to_datetime(acq["date"])
        a = acq[(d >= site.season[0]) & (d < site.season[1]) & acq["hv_db"].notna()]
        counts |= {k: int(v) for k, v in a["orbit"].value_counts().items()}

    wc = _none(get("wc_class"))
    wcd = _none(get("wc_disagrees"))
    sid = _none(get("seg_id"))
    facts = {
        "segment_id": None if sid is None else str(sid),
        "field_id": _none(get("field_id")),
        "municipality": _none(get("municipality")),
        "area_ha": _round(get("area_ha"), 2),
        "site": site.name,
        "cane_pred": bool(_none(get("cane_pred")) or False),
        "p_cane": _round(get("p_cane"), 3),
        "p_cane_radar": _round(get("p_cane_radar"), 3),
        "p_cane_optical": _round(get("p_cane_optical"), 3),
        "gate_variant": site.gate_variant,
        "status": status,
        "status_detail": raw_status,
        "harvest_date": _day(get("harvest_date")) if dated else None,
        "date_uncertainty_days": unc,
        "last_standing_date": last_s,
        "first_harvested_date": first_h,
        "hv_drop_db": _round(get("hv_drop_db"), 2),
        "coh_min": _round(get("coh_min"), 3),
        "coh_min_date": _day(series.get("coh_min_date")),
        "fires": fires,
        "rain_3d_mm_at_harvest": rain,
        "wc_class": None if wc is None else int(wc),
        "wc_disagrees": None if wcd is None else bool(wcd),
        "cane_confidence": None,
        "date_precision_days": unc,
        "acquisitions_used": counts,
        "segments_covering": _none(get("segments_covering")),
    }
    if facts["cane_pred"]:
        from zafra.alerts import other_gate

        # unrounded P: rounding to 3 decimals would move 0.4996 onto the 0.5 threshold
        name = other_gate(facts)[0]
        conf = cane_confidence(get("gate_confidence"), _none(get(f"p_cane_{name}")))
        # script 14: undated cane that WorldCover maps as trees / grassland is "uncertain"
        facts["cane_confidence"] = "uncertain" if facts["wc_disagrees"] and not dated else conf
    return facts


def cane_confidence(gate_conf, p_check) -> str:
    """Confidence in the cane call, script 14's harvest_confidence rule without the date
    condition: high = the gate is sure (tier "high") and the other gate agrees (P >= 0.5);
    low = the gate is unsure and the other gate disagrees; medium otherwise."""
    sure = gate_conf == "high"
    other = p_check is not None and not pd.isna(p_check) and float(p_check) >= 0.5
    return "high" if sure and other else "low" if not sure and not other else "medium"


# --- 2. display values --------------------------------------------------------------------


def display(facts: dict, alerts: list[dict], lang: str) -> dict:
    """The facts formatted for one language: the only strings a summary may draw from."""
    es = lang == "es"
    d = {
        "id": facts["field_id"] or facts["segment_id"],
        "area": f"{fmt_num(facts['area_ha'])} ha",
        "is_cane": facts["cane_pred"],
        "status": facts["status"],
    }
    conf = CONFIDENCE[lang].get(facts["cane_confidence"])
    if facts["cane_pred"] and conf:
        d["cane_line"] = f"Caña: confianza {conf}" if es else f"Cane: {conf} confidence"
    if facts["status"] == "dated":
        u = fmt_num(facts["date_precision_days"])
        date = fmt_date(facts["harvest_date"], lang)
        d["date_line"] = f"Fecha: {date} ±{u} días" if es else f"Date: {date} ±{u} days"
    elif facts["cane_pred"]:
        d["no_date"] = NO_DATE[lang][facts["status"]]
    if facts["fires"]:
        n = len(facts["fires"])
        d["fires_count"] = str(n)
        d["fires"] = (
            f"{_n_detections(n, lang)} {'cerca del polígono' if es else 'near the polygon'}"
        )
        d["first_fire"] = _first_fire(facts, lang)
    elif facts["cane_pred"]:
        d["fires"] = "Sin fuego VIIRS detectado cerca" if es else "No VIIRS fire detected nearby"
    if facts["wc_disagrees"] and facts["wc_class"] in WC_NAME[lang]:
        d["worldcover"] = WC_NAME[lang][facts["wc_class"]]
    d["alerts"] = [{"code": a["code"], "message": a[f"message_{lang}"]} for a in alerts]
    return d


# --- 3a. template -------------------------------------------------------------------------


def _n_detections(n: int, lang: str) -> str:
    if lang == "es":
        return f"{n} detección VIIRS" if n == 1 else f"{n} detecciones VIIRS"
    return f"{n} VIIRS detection" if n == 1 else f"{n} VIIRS detections"


def _first_fire(facts, lang) -> str:
    t = pd.Timestamp(facts["fires"][0]["datetime_local"])
    return f"{fmt_date(t, lang)} {t:%H:%M}"


def _template(facts, alerts, lang, labels: bool) -> str:
    d = display(facts, alerts, lang)
    es = lang == "es"
    parts = [f"{d['id']}, {d['area']}."]
    if not facts["cane_pred"]:
        parts.append(
            "El filtro no lo clasifica como caña; no se estima la cosecha."
            if es
            else "The gate does not classify it as cane; no harvest is estimated."
        )
    else:
        if "cane_line" in d:
            parts.append(d["cane_line"] + ".")
        parts.append((d["date_line"] if "date_line" in d else d["no_date"]) + ".")
    if "worldcover" in d:
        parts.append(
            f"WorldCover lo mapea como {d['worldcover']}."
            if es
            else f"WorldCover maps it as {d['worldcover']}."
        )
    if "first_fire" in d:
        first = d["first_fire"]
        parts.append(
            f"{d['fires']}; la primera el {first}."
            if es
            else f"{d['fires']}; the first on {first}."
        )
    elif "fires" in d:
        parts.append(d["fires"] + ".")
    if alerts:
        items = [
            f"{a['code']} ({ALERT_LABEL[lang][a['code']]})" if labels else a["code"] for a in alerts
        ]
        parts.append(("Alertas: " if es else "Alerts: ") + "; ".join(items) + ".")
    elif facts["cane_pred"]:
        parts.append("Sin alertas." if es else "No alerts.")
    return " ".join(parts)


def template_summary(facts: dict, alerts: list[dict], lang: str, max_words=MAX_WORDS) -> str:
    """Deterministic summary; alert labels are dropped if they push it past max_words."""
    text = _template(facts, alerts, lang, labels=True)
    if word_count(text) > max_words:
        text = _template(facts, alerts, lang, labels=False)
    return text


# --- 4. verifier --------------------------------------------------------------------------

_RE_DATE = {
    "es": re.compile(r"\b\d{1,2} (?:" + "|".join(MONTHS_ES) + r") \d{4}\b"),
    "en": re.compile(r"\b(?:" + "|".join(MONTHS_EN) + r") \d{1,2}, \d{4}\b"),
}
_RE_ISO = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_RE_TIME = re.compile(r"(?<![\w.:])\d{1,2}:\d{2}(?![\w:])")
_RE_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_RE_CODE = re.compile(r"\b[A-Z]{2,}(?:_[A-Z]+)+\b")
# Month words that would mean a date was written in a form the regexes above do not take
# (e.g. "12 de febrero", "February 12"): any of them in the text is rejected.
_RE_MONTH_WORD = re.compile(
    r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|"
    r"noviembre|diciembre|January|February|March|April|June|July|August|September|October|"
    r"November|December)\b",
    re.IGNORECASE,
)


# Confirmation, interpretation and causal language: the summary reports, it does not
# conclude. Matched case-insensitively on word boundaries, in both languages ("(0)": no
# restating of zero counts).
FORBIDDEN_PHRASES = {
    "es": [
        "se confirma", "se confirmó", "confirmado", "confirmada", "confirmamos", "por lo que",
        "indica que", "lo que indica", "lo que significa", "significa que", "debido a",
        "porque", "ya que", "por tanto", "por lo tanto", "por ello", "por eso", "así que",
        "es decir", "probablemente", "posiblemente", "seguramente", "parece", "sugiere",
        "se debe a", "lo cual", "en consecuencia", "(0)",
    ],
    "en": [
        "confirmed", "confirms that", "which means", "this means", "means that", "indicates",
        "indicating", "therefore", "because", "due to", "as a result", "thus", "hence",
        "so the", "probably", "likely", "suggests", "appears to", "seems", "which is why",
        "(0)",
    ],
}  # fmt: skip
_RE_FORBIDDEN = {
    lang: re.compile(
        "|".join(
            re.escape(ph) if ph.startswith("(") else rf"\b{re.escape(ph)}\b" for ph in phrases
        ),
        re.IGNORECASE,
    )
    for lang, phrases in FORBIDDEN_PHRASES.items()
}


def word_count(text: str) -> int:
    return len(text.split())


def tokens(text: str, lang: str) -> dict[str, list[str]]:
    """Dates (agreed format of `lang`), other-format dates, times, numbers, codes."""
    out = {"dates": _RE_DATE[lang].findall(text)}
    rest = _RE_DATE[lang].sub(" ", text)
    other = "en" if lang == "es" else "es"
    out["foreign_dates"] = _RE_DATE[other].findall(rest) + _RE_ISO.findall(rest)
    rest = _RE_ISO.sub(" ", _RE_DATE[other].sub(" ", rest))
    out["times"] = _RE_TIME.findall(rest)
    rest = _RE_TIME.sub(" ", rest)
    out["codes"] = _RE_CODE.findall(rest)
    out["numbers"] = [n.replace(",", ".") for n in _RE_NUM.findall(_RE_CODE.sub(" ", rest))]
    return out


def _strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def allowed_tokens(facts, alerts, lang) -> dict[str, set[str]]:
    al = {"dates": set(), "times": set(), "numbers": set(), "codes": set()}
    for s in _strings(display(facts, alerts, lang)):
        for k, v in tokens(s, lang).items():
            if k in al:
                al[k] |= set(v)
    al["codes"] = {a["code"] for a in alerts}
    return al


def verify(text: str, facts: dict, alerts: list[dict], lang: str, max_words=MAX_WORDS) -> dict:
    """{ok, problems}: required facts present verbatim, nothing outside the display values."""
    d = display(facts, alerts, lang)
    found = tokens(text, lang)
    allowed = allowed_tokens(facts, alerts, lang)
    problems = []
    if not text.strip():
        problems.append("empty")
    if word_count(text) > max_words:
        problems.append(f"too_long: {word_count(text)} words > {max_words}")
    low = text.lower()
    lines = [d[k] for k in ("cane_line", "date_line") if k in d]
    problems += [f"missing: {r}" for r in [d["area"], *lines] if r.lower() not in low]
    problems += [f"missing: {a['code']}" for a in alerts if a["code"] not in text]
    problems += [f"forbidden_phrase: {m}" for m in _RE_FORBIDDEN[lang].findall(text)]
    for kind in ("dates", "times", "numbers", "codes"):
        problems += [f"extra_{kind[:-1]}: {t}" for t in found[kind] if t not in allowed[kind]]
    problems += [f"date_format: {t}" for t in found["foreign_dates"]]
    problems += [f"date_format: {m}" for m in _RE_MONTH_WORD.findall(text)]
    return {"ok": not problems, "problems": problems}


# --- 3b. LLM ------------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"es": {"type": "string"}, "en": {"type": "string"}},
    "required": ["es", "en"],
    "additionalProperties": False,
}


def load_prompt(path=PROMPT_PATH, word_budget: int = WORD_BUDGET) -> str:
    return Path(path).read_text().replace("{word_budget}", str(word_budget))


def llm_input(facts: dict, alerts: list[dict]) -> str:
    """The user message: the display values of both languages, as JSON (stable order)."""
    payload = {lang: display(facts, alerts, lang) for lang in LANGS}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1)


def estimate_tokens(text: str) -> int:
    """Rough count for the cost estimate when no API response is available (~3.5 chars per
    token for this mixed Spanish / English JSON). Real counts come from response.usage."""
    return math.ceil(len(text) / 3.5)


def _settings(site) -> dict:
    s = dict(getattr(site, "summary", None) or {})
    return {
        "model": s.get("model", DEFAULT_MODEL),
        "effort": s.get("effort", "low"),
        "max_tokens": int(s.get("max_tokens", 1024)),
        "max_words": int(s.get("max_words", MAX_WORDS)),
        "word_budget": int(s.get("word_budget", WORD_BUDGET)),
    }


def _api_key() -> str | None:
    from zafra import config

    return config.get("ANTHROPIC_API_KEY")


def make_client():
    """(client, None) from ANTHROPIC_API_KEY in .env, or (None, reason). The key is never
    printed or stored."""
    key = _api_key()
    if not key:
        return None, "no_api_key"
    try:
        import anthropic
    except ImportError:
        return None, "sdk_not_installed"
    return anthropic.Anthropic(api_key=key), None


def llm_summaries(facts: dict, alerts: list[dict], site=None, client=None) -> dict:
    """Both languages from one LLM call, verified; template text on any failure.

    Returns {es, en, renderer ("llm" | "template"), fallback_reason (None or why),
    verifier {es, en}, model, usage {input_tokens, output_tokens} or None, rejected (the
    LLM text that failed the check, if any)}. `client` is injected in tests; by default it
    is built from ANTHROPIC_API_KEY (.env)."""
    cfg = _settings(site)
    mw = cfg["max_words"]
    rec = {"model": cfg["model"], "usage": None, "rejected": None}

    def fallback(reason):
        out = {lang: template_summary(facts, alerts, lang, mw) for lang in LANGS}
        ver = {lang: verify(out[lang], facts, alerts, lang, mw) for lang in LANGS}
        return rec | out | {"renderer": "template", "fallback_reason": reason, "verifier": ver}

    if client is None:
        client, reason = make_client()
        if client is None:
            return fallback(reason)
    try:
        resp = client.messages.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            system=load_prompt(word_budget=cfg["word_budget"]),
            messages=[{"role": "user", "content": llm_input(facts, alerts)}],
            output_config={
                "effort": cfg["effort"],
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        )
    except Exception as e:  # any API, network or client error ends in the template
        return fallback(f"api_error: {type(e).__name__}")
    u = getattr(resp, "usage", None)
    if u is not None:
        rec["usage"] = {
            "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
        }
    if resp.stop_reason in ("refusal", "max_tokens"):
        return fallback(f"stop_reason: {resp.stop_reason}")
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        out = json.loads(text)
        out = {lang: str(out[lang]).strip() for lang in LANGS}
    except (ValueError, KeyError, TypeError):
        rec["rejected"] = text
        return fallback("bad_json")
    ver = {lang: verify(out[lang], facts, alerts, lang, mw) for lang in LANGS}
    if not all(v["ok"] for v in ver.values()):
        rec["rejected"] = out | {"verifier": ver}
        return fallback("verifier_failed")
    return rec | out | {"renderer": "llm", "fallback_reason": None, "verifier": ver}


_CACHE: dict[str, dict] = {}


def llm_summary(facts: dict, alerts: list[dict], lang: str, site=None, client=None) -> str:
    """render(facts, alerts, lang) form of llm_summaries (one API call serves both
    languages; cached on the input)."""
    key = llm_input(facts, alerts)
    if key not in _CACHE:
        _CACHE[key] = llm_summaries(facts, alerts, site=site, client=client)
    return _CACHE[key][lang]
