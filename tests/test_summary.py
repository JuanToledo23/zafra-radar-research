"""Tests for exercise 6: alert rules (zafra.alerts), the template summary, the verifier and
the LLM fallback (zafra.summary). Synthetic facts only; the API client is always mocked,
so no test touches the network."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from zafra import alerts as A
from zafra import summary as SM
from zafra.site import load_site


@pytest.fixture(scope="module")
def site():
    return load_site("morelos")


@pytest.fixture(scope="module")
def cfg(site):
    return site.alerts


def make_facts(**kw) -> dict:
    """A dated, gated cane segment with no fire and both gates agreeing."""
    f = {
        "segment_id": "cc00001",
        "field_id": None,
        "municipality": None,
        "area_ha": 2.34,
        "site": "morelos",
        "cane_pred": True,
        "p_cane": 0.9,
        "p_cane_radar": 0.8,
        "p_cane_optical": 0.9,
        "gate_variant": "B",
        "status": "dated",
        "status_detail": "harvested",
        "harvest_date": "2026-02-12",
        "date_uncertainty_days": 6.0,
        "last_standing_date": "2026-02-06",
        "first_harvested_date": "2026-02-18",
        "hv_drop_db": -3.1,
        "coh_min": 0.21,
        "coh_min_date": "2026-02-18",
        "fires": [],
        "rain_3d_mm_at_harvest": None,
        "wc_class": 40,
        "wc_disagrees": False,
        "cane_confidence": "high",
        "acquisitions_used": {"asc": 15, "desc": 15},
    }
    f |= kw
    f.setdefault("date_precision_days", None)
    if "date_precision_days" not in kw:
        f["date_precision_days"] = f["date_uncertainty_days"]
    return f


def fire(dt, location="within_buffer"):
    overpass = "night" if int(dt[11:13]) < 6 else "day"
    return {
        "datetime_local": dt,
        "sensor": "VIIRS SNPP",
        "frp": 3.2,
        "location": location,
        "overpass": overpass,
    }


def codes(facts, cfg):
    return [a["code"] for a in A.evaluate(facts, cfg)]


# --- alert rules, one per rule ------------------------------------------------------------


def test_fire_active_outside_hours(cfg):
    night = make_facts(fires=[fire("2026-02-15 01:37")])
    day = make_facts(fires=[fire("2026-02-15 13:30")])
    assert "FIRE_ACTIVE_OUTSIDE_HOURS" in codes(night, cfg)
    assert "FIRE_ACTIVE_OUTSIDE_HOURS" not in codes(day, cfg)
    a = next(a for a in A.evaluate(night, cfg) if a["code"] == "FIRE_ACTIVE_OUTSIDE_HOURS")
    assert a["evidence"]["burn_window"] == ["09:00", "17:00"]
    assert a["evidence"]["window_confirmed"] is False  # no municipality: assumed window
    assert a["evidence"]["overpass"] == "night"
    assert a["message_es"] == "fuego activo a las 01:37, fuera del horario permitido (09:00–17:00)"
    assert a["message_en"] == "fire active at 01:37, outside permitted hours (09:00–17:00)"


def test_burn_window_per_municipality(cfg):
    assert A.burn_window(cfg, "Tlaquiltenango") == ("09:00", "17:00", True)
    assert A.burn_window(cfg, None) == ("09:00", "17:00", False)
    bw = cfg["burn_window"] | {"municipalities": {"X": ["10:00", "12:00"]}}
    narrow = cfg | {"burn_window": bw}
    f = make_facts(municipality="X", fires=[fire("2026-02-15 13:30")])
    assert "FIRE_ACTIVE_OUTSIDE_HOURS" in codes(f, narrow)


def test_burn_without_harvest(cfg):
    # fire 2026-02-01, first harvested look 2026-02-18: 17 days later > 14
    late = make_facts(fires=[fire("2026-02-01 13:30")])
    soon = make_facts(fires=[fire("2026-02-10 13:30")])
    undated = make_facts(
        status="no_harvest", harvest_date=None, first_harvested_date=None,
        date_uncertainty_days=None, last_standing_date=None, fires=[fire("2026-02-10 13:30")],
    )  # fmt: skip
    assert "BURN_WITHOUT_HARVEST" in codes(late, cfg)
    assert "BURN_WITHOUT_HARVEST" not in codes(soon, cfg)
    assert "BURN_WITHOUT_HARVEST" in codes(undated, cfg)


def test_harvest_without_fire(cfg, site):
    # disabled on both sites (exercise-6 review); the rule itself still works when enabled
    assert "HARVEST_WITHOUT_FIRE" not in codes(make_facts(), cfg)
    assert "HARVEST_WITHOUT_FIRE" not in codes(make_facts(), load_site("brazil").alerts)
    rules = cfg["rules"] | {
        "HARVEST_WITHOUT_FIRE": cfg["rules"]["HARVEST_WITHOUT_FIRE"] | {"enabled": True}
    }
    on = cfg | {"rules": rules}
    assert "HARVEST_WITHOUT_FIRE" in codes(make_facts(), on)
    with_fire = make_facts(fires=[fire("2026-02-14 13:30")])
    assert "HARVEST_WITHOUT_FIRE" not in codes(with_fire, on)
    a = next(a for a in A.evaluate(make_facts(), on) if a["code"] == "HARVEST_WITHOUT_FIRE")
    assert a["severity"] == "info"
    # the summary still states the absence of fire
    assert "Sin fuego VIIRS detectado cerca." in SM.template_summary(make_facts(), [], "es")


def test_high_date_uncertainty(cfg):
    assert "HIGH_DATE_UNCERTAINTY" in codes(make_facts(date_uncertainty_days=10.5), cfg)
    assert "HIGH_DATE_UNCERTAINTY" not in codes(make_facts(date_uncertainty_days=9.0), cfg)


def test_radar_optical_disagree(cfg):
    assert "RADAR_OPTICAL_DISAGREE" in codes(make_facts(p_cane_radar=0.12), cfg)
    assert "RADAR_OPTICAL_DISAGREE" not in codes(make_facts(p_cane_radar=0.31), cfg)
    # radar gate delivered: the optical P is the check
    radar = make_facts(gate_variant="A", p_cane_radar=0.9, p_cane_optical=0.2)
    assert "RADAR_OPTICAL_DISAGREE" in codes(radar, cfg)


def test_no_baseline(cfg):
    f = make_facts(
        status="no_baseline", status_detail="no_baseline", harvest_date=None,
        date_uncertainty_days=None, last_standing_date=None, first_harvested_date=None,
    )  # fmt: skip
    assert codes(f, cfg) == ["NO_BASELINE"]


def test_non_cane_gets_no_alerts(cfg):
    f = make_facts(cane_pred=False, status="no_harvest", fires=[fire("2026-02-15 01:37")])
    assert A.evaluate(f, cfg) == []


# --- template -----------------------------------------------------------------------------


def test_date_formats():
    assert SM.fmt_date("2026-02-12", "es") == "12 feb 2026"
    assert SM.fmt_date("2026-02-12", "en") == "Feb 12, 2026"
    assert SM.fmt_num(6.0) == "6" and SM.fmt_num(2.5) == "2.5"


@pytest.mark.parametrize(
    "lang, date, unc", [("es", "12 feb 2026", "±6 días"), ("en", "Feb 12, 2026", "±6 days")]
)
def test_template_has_date_and_uncertainty(cfg, lang, date, unc):
    f = make_facts(fires=[fire("2026-02-15 01:37")])
    al = A.evaluate(f, cfg)
    text = SM.template_summary(f, al, lang)
    assert date in text and unc in text
    # cane confidence and date precision are stated separately
    cane = "Caña: confianza alta." if lang == "es" else "Cane: high confidence."
    line = f"Fecha: {date} {unc}." if lang == "es" else f"Date: {date} {unc}."
    assert cane in text and line in text
    assert all(a["code"] in text for a in al)
    assert SM.word_count(text) <= SM.MAX_WORDS
    assert SM.verify(text, f, al, lang)["ok"]


def test_template_undated_and_non_cane(cfg):
    for f in (
        make_facts(status="no_baseline", harvest_date=None, date_uncertainty_days=None,
                   last_standing_date=None, first_harvested_date=None),
        make_facts(cane_pred=False, status="no_harvest", harvest_date=None,
                   date_uncertainty_days=None, first_harvested_date=None, cane_confidence=None),
    ):  # fmt: skip
        al = A.evaluate(f, cfg)
        for lang in SM.LANGS:
            assert SM.verify(SM.template_summary(f, al, lang), f, al, lang)["ok"]


# --- verifier -----------------------------------------------------------------------------


def test_verifier_rejects_extra_date(cfg):
    f = make_facts()
    al = A.evaluate(f, cfg)
    good = SM.template_summary(f, al, "es")
    bad = good + " La zafra terminó el 30 may 2026."
    v = SM.verify(bad, f, al, "es")
    assert not v["ok"] and any("30 may 2026" in p for p in v["problems"])


@pytest.mark.parametrize(
    "lang, extra",
    [
        ("es", "Se confirma que es caña."),
        ("es", "El radar no confirma caña, por lo que la clasificación es de advertencia."),
        ("es", "Esto indica que hubo quema."),
        ("es", "Sin alertas (0)."),
        ("en", "It is confirmed as cane."),
        ("en", "The radar disagrees, which means the call is weak."),
        ("en", "No fires detected (0)."),
    ],
)
def test_verifier_rejects_forbidden_phrases(cfg, lang, extra):
    f = make_facts()
    al = A.evaluate(f, cfg)
    v = SM.verify(SM.template_summary(f, al, lang) + " " + extra, f, al, lang)
    assert not v["ok"] and any(p.startswith("forbidden_phrase") for p in v["problems"])


def test_alert_message_with_confirma_is_not_forbidden(cfg):
    # "no confirma" (an alert's own message) is a report, not a confirmation
    f = make_facts(p_cane_radar=0.1)
    al = A.evaluate(f, cfg)
    assert "RADAR_OPTICAL_DISAGREE" in [a["code"] for a in al]
    assert SM.verify(SM.template_summary(f, al, "es"), f, al, "es")["ok"]


@pytest.mark.parametrize(
    "extra",
    ["El 2026-02-12 se cortó.", "Se quemó el 12 de febrero.", "Unas 40 personas.", "FAKE_CODE."],
)
def test_verifier_rejects_other_numbers_formats_codes(cfg, extra):
    f = make_facts()
    al = A.evaluate(f, cfg)
    assert not SM.verify(SM.template_summary(f, al, "es") + " " + extra, f, al, "es")["ok"]


def test_verifier_requires_facts_and_length(cfg):
    f = make_facts()
    al = A.evaluate(f, cfg)
    assert not SM.verify("cc00001, 2.3 ha: cosecha el 12 feb 2026.", f, al, "es")["ok"]
    long = SM.template_summary(f, al, "en") + " ok" * 70
    assert any(p.startswith("too_long") for p in SM.verify(long, f, al, "en")["problems"])


# --- LLM path (mocked client) -------------------------------------------------------------


class FakeClient:
    def __init__(self, payload=None, exc=None, stop="end_turn"):
        self.payload, self.exc, self.stop, self.calls = payload, exc, stop, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            stop_reason=self.stop,
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=800, output_tokens=120),
        )


def test_fallback_when_api_key_missing(cfg, site, monkeypatch):
    monkeypatch.setattr(SM, "_api_key", lambda: None)
    f = make_facts()
    al = A.evaluate(f, cfg)
    r = SM.llm_summaries(f, al, site=site)
    assert r["renderer"] == "template" and r["fallback_reason"] == "no_api_key"
    assert r["es"] == SM.template_summary(f, al, "es")
    assert SM.llm_summary(f, al, "en", site=site) == SM.template_summary(f, al, "en")


def test_llm_output_accepted_when_verified(cfg, site):
    f = make_facts(fires=[fire("2026-02-15 01:37")])
    al = A.evaluate(f, cfg)
    assert [a["code"] for a in al] == ["FIRE_ACTIVE_OUTSIDE_HOURS"]
    out = {
        "es": "Segmento cc00001, 2.3 ha. Caña: confianza alta. Fecha: 12 feb 2026 ±6 días. "
        "1 detección VIIRS cerca del polígono, el 15 feb 2026 01:37. "
        "FIRE_ACTIVE_OUTSIDE_HOURS: fuego activo a las 01:37, fuera del horario permitido "
        "(09:00–17:00).",
        "en": "Segment cc00001, 2.3 ha. Cane: high confidence. Date: Feb 12, 2026 ±6 days. "
        "1 VIIRS detection near the polygon, on Feb 15, 2026 01:37. "
        "FIRE_ACTIVE_OUTSIDE_HOURS: fire active at 01:37, outside permitted hours "
        "(09:00–17:00).",
    }
    client = FakeClient(out)
    r = SM.llm_summaries(f, al, site=site, client=client)
    assert r["renderer"] == "llm", r["verifier"]
    assert r["usage"] == {"input_tokens": 800, "output_tokens": 120}
    call = client.calls[0]
    assert call["model"] == site.summary["model"]
    assert "{word_budget}" not in call["system"] and "At most 55 words" in call["system"]
    assert json.loads(call["messages"][0]["content"]).keys() == {"es", "en"}


def test_llm_output_with_extra_date_falls_back(cfg, site):
    f = make_facts()
    al = A.evaluate(f, cfg)
    es = SM.template_summary(f, al, "es") + " Antes se vio en pie el 1 feb 2026."
    r = SM.llm_summaries(f, al, site=site, client=FakeClient({"es": es, "en": "x"}))
    assert r["renderer"] == "template" and r["fallback_reason"] == "verifier_failed"
    assert not r["rejected"]["verifier"]["es"]["ok"]
    assert r["es"] == SM.template_summary(f, al, "es")


@pytest.mark.parametrize(
    "client, reason",
    [
        (FakeClient(exc=ConnectionError("offline")), "api_error: ConnectionError"),
        (FakeClient({"es": "a", "en": "b"}, stop="refusal"), "stop_reason: refusal"),
    ],
)
def test_llm_api_failure_falls_back(cfg, site, client, reason):
    f = make_facts()
    r = SM.llm_summaries(f, A.evaluate(f, cfg), site=site, client=client)
    assert r["renderer"] == "template" and r["fallback_reason"] == reason


# --- facts --------------------------------------------------------------------------------


def test_field_facts_from_outputs(site):
    row = {
        "seg_id": "cc00009", "area_ha": 1.234, "cane_pred": True, "p_cane": 0.91,
        "p_cane_radar": 0.4, "p_cane_optical": 0.91, "status": "harvested",
        "harvest_date": "2026-02-12", "harvest_after": pd.Timestamp("2026-02-15"),
        "date_uncertainty_days": 2.5, "hv_drop_db": -3.0, "coh_min": 0.2, "wc_class": 40,
        "wc_disagrees": False, "gate_confidence": "high", "confidence": "medium",
    }  # fmt: skip
    series = {
        "acq": pd.DataFrame(
            {"date": pd.to_datetime(["2025-11-23", "2025-11-28", "2026-07-01"]),
             "orbit": ["asc", "desc", "asc"], "hv_db": [-12.0, -12.5, -13.0]}
        ),
        "fires": pd.DataFrame(
            {"acq_utc": pd.to_datetime(["2026-02-14 07:37", "2026-02-14 19:30"], utc=True),
             "satellite": ["NOAA-20", "SNPP"], "daynight": ["N", "D"], "frp": [2.9, 5.0],
             "dist_edge_m": [0.0, 350.0]}
        ),
        "coh_min_date": pd.Timestamp("2026-02-15"),
    }  # fmt: skip
    f = SM.field_facts(row, series, site)
    assert f["status"] == "dated" and f["first_harvested_date"] == "2026-02-15"
    assert f["last_standing_date"] == "2026-02-10"
    assert f["acquisitions_used"] == {"asc": 1, "desc": 1}  # July is outside the season
    assert f["fires"] == [
        {"datetime_local": "2026-02-14 01:37", "sensor": "VIIRS NOAA-20", "frp": 2.9,
         "location": "inside_polygon", "overpass": "night"}
    ]  # fmt: skip
    assert f["gate_variant"] == "B" and f["coh_min_date"] == "2026-02-15"
    assert f["segments_covering"] is None
    # gate sure, radar P 0.4 < 0.5: medium, as the GeoJSON confidence of this row
    assert f["cane_confidence"] == "medium" and f["date_precision_days"] == 2.5
    assert "confidence" not in f


def test_reference_field_row_and_id(site, cfg):
    """A reference-field row (seg_id None) is summarised under its field id."""
    row = {
        "seg_id": None, "field_id": "p36", "municipality": "Tlaquiltenango", "area_ha": 21.43,
        "cane_pred": True, "p_cane": 0.7, "p_cane_radar": 0.2, "p_cane_optical": 0.7,
        "status": "harvested", "harvest_date": "2026-02-12",
        "harvest_after": pd.Timestamp("2026-02-15"), "date_uncertainty_days": 2.5,
        "wc_class": None, "wc_disagrees": None, "gate_confidence": "uncertain",
        "segments_covering": {"count": 23, "share_of_area": 0.945},
    }  # fmt: skip
    f = SM.field_facts(row, {}, site)
    assert f["segment_id"] is None and f["wc_disagrees"] is None
    assert f["segments_covering"] == {"count": 23, "share_of_area": 0.945}
    al = A.evaluate(f, cfg)
    text = SM.template_summary(f, al, "en")
    assert text.startswith("p36, 21.4 ha.") and SM.verify(text, f, al, "en")["ok"]
    assert "Cane: low confidence." in text and "Date: Feb 12, 2026 ±2.5 days." in text


def _script14():
    import importlib.util

    from zafra.paths import ROOT

    spec = importlib.util.spec_from_file_location(
        "s14", ROOT / "scripts" / "14_area_harvest_map.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "gate_conf, p_check",
    [("high", 0.8), ("high", 0.2), ("uncertain", 0.2), ("uncertain", 0.8), ("no_data", None)],
)
def test_cane_confidence_matches_script14(gate_conf, p_check):
    s14 = _script14()
    s14.SITE = load_site("morelos")
    p = float("nan") if p_check is None else p_check
    want = s14.harvest_confidence([gate_conf], [p], [True])[0]
    assert SM.cane_confidence(gate_conf, p_check) == want
