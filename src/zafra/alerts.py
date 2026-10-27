"""Alert rules on the per-field facts (exercise 6).

    from zafra.alerts import evaluate
    alerts = evaluate(facts, site.alerts)   # facts from zafra.summary.field_facts

Each rule reads only the facts dict and its own parameters from the site config
(`alerts.rules.<CODE>` in configs/<site>.yaml) and returns at most one alert:
{code, severity, message_es, message_en, evidence}. A rule with `enabled: false` (or no
entry in the config) never fires. Rules run only on segments the gate calls cane; a
non-cane polygon gets no alerts.

Rules (parameters in brackets):
  FIRE_ACTIVE_OUTSIDE_HOURS
                          a VIIRS detection inside / within the fire buffer of the polygon
                          at a local time outside the burn window [burn_window, per
                          municipality if configured, else the default]. VIIRS sees a fire
                          only at its overpasses (about 01:30 and 13:30 local in Morelos):
                          a night detection shows a fire still burning, not when it was
                          lit. The evidence says which overpass (night / day).
  BURN_WITHOUT_HARVEST    a detection with no harvested state by [days_after] days after it
                          (first harvested acquisition later than that, or no date at all)
  HARVEST_WITHOUT_FIRE    a dated harvest with no detection in the [days_before] days before
                          the first harvested acquisition (disabled on both sites since
                          exercise 6 review: VIIRS misses most cane burns; the summary still
                          says when no fire was detected)
  HIGH_DATE_UNCERTAINTY   date_uncertainty_days > [max_days]
  RADAR_OPTICAL_DISAGREE  the delivered gate says cane, the other gate's P < [max_other_p]
  NO_BASELINE             status no_baseline

Dates in messages are the agreed formats of zafra.summary (12 feb 2026 / Feb 12, 2026), so
the summary verifier can check them verbatim.
"""

from __future__ import annotations

import pandas as pd

from zafra.detect import cane_gate as G
from zafra.summary import fmt_date, fmt_num

CODES = (
    "FIRE_ACTIVE_OUTSIDE_HOURS",
    "BURN_WITHOUT_HARVEST",
    "HARVEST_WITHOUT_FIRE",
    "HIGH_DATE_UNCERTAINTY",
    "RADAR_OPTICAL_DISAGREE",
    "NO_BASELINE",
)
SEVERITIES = ("info", "warning")


# --- helpers ------------------------------------------------------------------------------


def _params(cfg: dict, code: str) -> dict | None:
    p = (cfg.get("rules") or {}).get(code)
    return p if p and p.get("enabled", False) else None


def _alert(code, params, message_es, message_en, evidence) -> dict:
    sev = params.get("severity", "info")
    if sev not in SEVERITIES:
        raise ValueError(f"{code}: severity {sev!r} not in {SEVERITIES}")
    return {
        "code": code,
        "severity": sev,
        "message_es": message_es,
        "message_en": message_en,
        "evidence": evidence,
    }


def _fire_time(f) -> pd.Timestamp:
    return pd.Timestamp(f["datetime_local"])


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def burn_window(cfg: dict, municipality: str | None) -> tuple[str, str, bool] | None:
    """(start, end, confirmed) of the local burn window, or None if none is configured.

    A municipality listed in `burn_window.municipalities` uses its own window (confirmed);
    any other polygon uses `default`, confirmed only if `default_confirmed` is true."""
    bw = cfg.get("burn_window")
    if not bw:
        return None
    own = (bw.get("municipalities") or {}).get(municipality) if municipality else None
    if own:
        return str(own[0]), str(own[1]), True
    d = bw.get("default")
    if not d:
        return None
    return str(d[0]), str(d[1]), bool(bw.get("default_confirmed", False))


def other_gate(facts: dict) -> tuple[str, float | None]:
    """Name and P of the gate the delivered one is checked against (as script 14's
    check_column): radar-only for an optical / combined gate, optical for a radar gate."""
    v = G.resolve_variant(facts["gate_variant"])
    if G.USES_RADAR[v] and not G.USES_OPTICAL[v]:
        return "optical", facts.get("p_cane_optical")
    return "radar", facts.get("p_cane_radar")


def _date_es(d) -> str:
    return fmt_date(d, "es")


def _date_en(d) -> str:
    return fmt_date(d, "en")


# --- rules --------------------------------------------------------------------------------


def fire_active_outside_hours(facts: dict, cfg: dict) -> dict | None:
    p = _params(cfg, "FIRE_ACTIVE_OUTSIDE_HOURS")
    win = burn_window(cfg, facts.get("municipality"))
    if p is None or win is None:
        return None
    a, b, confirmed = win
    lo, hi = _hhmm(a), _hhmm(b)
    out = []
    for f in facts["fires"]:
        t = _fire_time(f)
        m = t.hour * 60 + t.minute
        if not lo <= m <= hi:
            out.append(f)
    if not out:
        return None
    first = out[0]
    hhmm = f"{_fire_time(first):%H:%M}"
    return _alert(
        "FIRE_ACTIVE_OUTSIDE_HOURS",
        p,
        f"fuego activo a las {hhmm}, fuera del horario permitido ({a}–{b})",
        f"fire active at {hhmm}, outside permitted hours ({a}–{b})",
        {
            "burn_window": [a, b],
            "window_confirmed": confirmed,
            "municipality": facts.get("municipality"),
            "overpass": first.get("overpass"),
            "datetime_local": first["datetime_local"],
            "fires_outside": out,
        },
    )


def burn_without_harvest(facts: dict, cfg: dict) -> dict | None:
    p = _params(cfg, "BURN_WITHOUT_HARVEST")
    if p is None or not facts["fires"]:
        return None
    days = int(p["days_after"])
    first_h = facts.get("first_harvested_date")
    fh = pd.Timestamp(first_h) if first_h and facts["status"] == "dated" else None
    out = [
        f
        for f in facts["fires"]
        if fh is None or fh > _fire_time(f).normalize() + pd.Timedelta(days=days)
    ]
    if not out:
        return None
    first = _fire_time(out[0])
    n = len(out)
    return _alert(
        "BURN_WITHOUT_HARVEST",
        p,
        f"{n} detección(es) VIIRS sin cosecha observada en los {days} días siguientes; "
        f"la primera el {_date_es(first)}.",
        f"{n} VIIRS detection(s) with no harvest seen in the following {days} days; "
        f"the first on {_date_en(first)}.",
        {"days_after": days, "first_harvested_date": first_h, "fires": out},
    )


def harvest_without_fire(facts: dict, cfg: dict) -> dict | None:
    p = _params(cfg, "HARVEST_WITHOUT_FIRE")
    if p is None or facts["status"] != "dated" or not facts.get("first_harvested_date"):
        return None
    days = int(p["days_before"])
    fh = pd.Timestamp(facts["first_harvested_date"])
    near = [
        f for f in facts["fires"] if fh - pd.Timedelta(days=days) <= _fire_time(f).normalize() <= fh
    ]
    if near:
        return None
    return _alert(
        "HARVEST_WITHOUT_FIRE",
        p,
        f"Cosecha fechada sin detección VIIRS en los {days} días previos.",
        f"Dated harvest with no VIIRS detection in the {days} days before.",
        {"days_before": days, "first_harvested_date": facts["first_harvested_date"]},
    )


def high_date_uncertainty(facts: dict, cfg: dict) -> dict | None:
    p = _params(cfg, "HIGH_DATE_UNCERTAINTY")
    u = facts.get("date_uncertainty_days")
    if p is None or u is None or not u > float(p["max_days"]):
        return None
    lim = fmt_num(p["max_days"])
    return _alert(
        "HIGH_DATE_UNCERTAINTY",
        p,
        f"Fecha imprecisa: ±{fmt_num(u)} días (límite {lim}).",
        f"Imprecise date: ±{fmt_num(u)} days (limit {lim}).",
        {"date_uncertainty_days": u, "max_days": p["max_days"]},
    )


def radar_optical_disagree(facts: dict, cfg: dict) -> dict | None:
    p = _params(cfg, "RADAR_OPTICAL_DISAGREE")
    if p is None:
        return None
    name, pr = other_gate(facts)
    lim = float(p["max_other_p"])
    if pr is None or not pr < lim:
        return None
    es = {"radar": "radar", "optical": "óptico"}[name]
    return _alert(
        "RADAR_OPTICAL_DISAGREE",
        p,
        f"El filtro {es} no confirma caña (P = {fmt_num(pr, 2)}).",
        f"The {name} gate does not confirm cane (P = {fmt_num(pr, 2)}).",
        {"other_gate": name, "p_other": pr, "max_other_p": lim, "p_cane": facts["p_cane"]},
    )


def no_baseline(facts: dict, cfg: dict) -> dict | None:
    p = _params(cfg, "NO_BASELINE")
    if p is None or facts["status"] != "no_baseline":
        return None
    return _alert(
        "NO_BASELINE",
        p,
        "Señal de cosecha sin suficientes observaciones previas en pie: sin fecha.",
        "Harvest signal without enough earlier standing looks: not dated.",
        {"status_detail": facts.get("status_detail")},
    )


RULES = {
    "FIRE_ACTIVE_OUTSIDE_HOURS": fire_active_outside_hours,
    "BURN_WITHOUT_HARVEST": burn_without_harvest,
    "HARVEST_WITHOUT_FIRE": harvest_without_fire,
    "HIGH_DATE_UNCERTAINTY": high_date_uncertainty,
    "RADAR_OPTICAL_DISAGREE": radar_optical_disagree,
    "NO_BASELINE": no_baseline,
}


def evaluate(facts: dict, cfg: dict) -> list[dict]:
    """Every rule that fires on `facts`, in CODES order. Non-cane polygons: none."""
    if not facts.get("cane_pred", True):
        return []
    return [a for code in CODES if (a := RULES[code](facts, cfg)) is not None]
