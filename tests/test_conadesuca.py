"""Tests for exercise 10: CONADESUCA report parser (zafra.conadesuca). The fixture is the
`pdftotext -layout` text of report 36 (week 42, zafra 2025/26): page 1 header and the
weekly / cumulative per-mill and per-state tables. No PDF needed."""

from datetime import date
from pathlib import Path

import pytest

from zafra import conadesuca as C

FIXTURE = Path(__file__).parent / "fixtures" / "conadesuca_report36_excerpt.txt"


@pytest.fixture(scope="module")
def text():
    return FIXTURE.read_text(encoding="utf-8")


def test_header(text):
    h = C.report_header(text)
    assert h["report"] == 36 and h["cycle_week"] == 42 and h["cycle"] == "2025/26"
    assert h["cutoff"] == date(2026, 7, 18)


def test_period_across_years():
    t = (
        "  Semana 14 del ciclo azucarero 2025/26\n     Reporte 8\n"
        "  Comprende el periodo del 28 de dic de 2025 al 03 de ene de 2026\n"
    )
    assert C.report_header(t)["cutoff"] == date(2026, 1, 3)


def test_cumulative_mill_and_region_rows(text):
    rows = C.section_rows(text, C.CUMULATIVE_MILL_REGION, ("Nacional", "Centro", "Emiliano Zapata"))
    ez = rows["Emiliano Zapata"]
    assert ez["marked"] and ez["area_ha"] == 9565 and ez["cane_t"] == 915365
    assert ez["field_yield_t_ha"] == 95.70 and ez["factory_yield_pct"] == 13.43
    assert ez["refined_sugar_t"] is None and ez["karbe_kg_tcn"] == 134.307
    assert rows["Centro"]["area_ha"] == 33301
    assert rows["Nacional"]["cane_t"] == 49605185


def test_state_rows(text):
    rows = C.section_rows(text, C.CUMULATIVE_STATE, ("Total nacional", "Morelos"))
    assert rows["Morelos"]["area_ha"] == 15787
    assert rows["Total nacional"]["area_ha"] == 736458


def test_short_row_is_an_error_not_a_guess(text):
    # the weekly Centro row has blank cells (9 values): it must not be parsed
    rows = C.section_rows(text, C.WEEKLY_MILL_REGION, ("Centro", "Emiliano Zapata"))
    assert isinstance(rows["Centro"], C.ParseError)
    assert all(rows["Emiliano Zapata"][c] is None for c in C.COLUMNS)


def test_absent_name_is_left_out(text):
    rows = C.section_rows(text, C.CUMULATIVE_MILL_REGION, ("Ingenio que no existe",))
    assert rows == {}


def test_changed_layout_fails(text):
    with pytest.raises(C.ParseError, match="units"):
        C.section(text.replace("kg/tCN", "kg/t"), C.CUMULATIVE_MILL_REGION)


def test_number():
    assert C.number("9,565") == 9565 and C.number("-1,000") == -1000
    assert C.number("13.43") == 13.43 and C.number("-") is None
    with pytest.raises(C.ParseError):
        C.number("9.565,0")


def test_footnotes(text):
    assert "*El ingenio concluyó su zafra 2025/26" in C.footnotes(text)
