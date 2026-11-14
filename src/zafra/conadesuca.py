"""Parse CONADESUCA zafra reports (weekly progress and final), from `pdftotext -layout` text.

    from zafra import conadesuca as C
    text = C.pdf_text(path)                       # pdftotext -layout; pdfplumber fallback
    head = C.report_header(text)                  # report number, cycle week, period, cut-off
    rows = C.section_rows(text, C.CUMULATIVE_MILL_REGION)
    rows["Emiliano Zapata"]                       # dict of the 12 columns, plus raw line

The weekly report tables "Información semanal / acumulada por ingenio y región cañera" and
"... por Entidad Federativa" share one layout: a name, then 12 columns (COLUMNS). A row is
parsed only when it has exactly 12 values and every value is a number or "-" (no data);
anything else raises ParseError, so a number is never guessed. Table numbers ("Cuadro 9")
change between reports, so sections are found by their title.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

COLUMNS = [
    "area_ha",  # Superficie industrializada (ha)
    "cane_t",  # Caña molida (t)
    "field_yield_t_ha",  # Rendimiento de campo (t/ha)
    "sugar_t",  # Producción de azúcar (t)
    "factory_yield_pct",  # Rendimiento de fábrica (%)
    "agroindustrial_yield_t_ha",  # Rendimiento agroindustrial (t azúcar/ha)
    "refined_sugar_t",  # Azúcar refinada (t)
    "standard_sugar_t",  # Azúcar estándar (t)
    "special_white_sugar_t",  # Azúcar blanco especial (t)
    "mascabado_t",  # Azúcar mascabado (t)
    "pol_below_99_2_t",  # Azúcar con pol menor a 99.2 (t)
    "karbe_kg_tcn",  # KARBE neto teórico (kg/tCN)
]
UNITS = "ha t t/ha t % t azúcar/ha t t t t t kg/tCN"

WEEKLY_MILL_REGION = r"Información semanal por ingenio y\d? región cañera"
CUMULATIVE_MILL_REGION = r"Información acumulada por ingenio y\d? región cañera"
WEEKLY_STATE = r"Información semanal por Entidad Federativa"
CUMULATIVE_STATE = r"Información acumulada por Entidad Federativa"

MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}  # fmt: skip


class ParseError(ValueError):
    pass


def pdf_text(pdf: Path) -> str:
    """Layout text of a PDF, pages separated by form feeds. pdftotext -layout when it is on
    PATH, else pdfplumber (optional, not a declared dependency)."""
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, check=True)
        return r.stdout.decode("utf-8")
    import pdfplumber  # fallback

    with pdfplumber.open(pdf) as doc:
        return "\f".join(p.extract_text(layout=True) or "" for p in doc.pages)


def number(tok: str) -> float | None:
    """'9,565' -> 9565.0, '13.43' -> 13.43, '-1,000' -> -1000.0, '-' -> None."""
    if tok == "-":
        return None
    if not re.fullmatch(r"-?\d{1,3}(,\d{3})*(\.\d+)?|-?\d+(\.\d+)?", tok):
        raise ParseError(f"not a number: {tok!r}")
    return float(tok.replace(",", ""))


def spanish_date(day: str, month: str, year: str) -> date:
    m = MONTHS.get(month.lower()[:3])
    if m is None:
        raise ParseError(f"unknown month: {month!r}")
    return date(int(year), m, int(day))


def long_date(s: str) -> date:
    """'27 de noviembre de 2025' -> date(2025, 11, 27)."""
    m = re.fullmatch(r"\s*(\d{1,2}) de (\w+) de (\d{4})\s*", s)
    if not m:
        raise ParseError(f"not a date: {s!r}")
    return spanish_date(*m.groups())


def report_header(text: str) -> dict:
    """Report number, cycle week, period text and cut-off date (last day of the period) from
    the first page of a weekly report."""
    first = text.split("\f", 1)[0]
    rep = re.search(r"^\s*Reporte (\d+)\s*$", first, re.M)
    week = re.search(r"Semana (\d+) del ciclo azucarero (\d{4}/\d{2})", first)
    period = re.search(r"Comprende el periodo (del .*\d{4})", first)
    if not (rep and week and period):
        raise ParseError("header: report number, week or period not found")
    ptxt = re.sub(r"\s+", " ", period.group(1))
    end = re.search(r"al (\d{1,2}) de (\w+) de (\d{4})$", ptxt)
    if not end:
        raise ParseError(f"period end not found: {ptxt!r}")
    return {
        "report": int(rep.group(1)),
        "cycle_week": int(week.group(1)),
        "cycle": week.group(2),
        "period": ptxt,
        "cutoff": spanish_date(*end.groups()),
    }


def section(text: str, title: str) -> list[str]:
    """Lines of the table whose title matches `title`, up to the next 'Información' title or,
    after the units line, the next 'Cuadro N' label. The units line must match UNITS, so a
    changed column layout fails loudly."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.search(title, ln)]
    if len(starts) != 1:
        raise ParseError(f"section {title!r}: found {len(starts)} times")
    out, units = [], []
    for ln in lines[starts[0] + 1 :]:
        if re.search(r"^\s*Información ", ln) or (units and re.search(r"Cuadro \d+\s*$", ln)):
            break
        if re.match(r"^\s*ha\s+t\s+t/ha", ln):
            units.append(ln)
        out.append(ln)
    if len(units) != 1 or " ".join(units[0].split()) != UNITS:
        raise ParseError(f"section {title!r}: units line not as expected")
    return out


def parse_row(line: str) -> dict:
    """'Emiliano Zapata*   9,565   915,365 ...' -> {name, marked, 12 columns, raw}. `marked`
    is the '*' after the name; its meaning is the report's footnote (see `footnotes`)."""
    m = re.match(r"^\s*(\S.*?\S\*?)\s{2,}(\S.*)$", line)
    if not m:
        raise ParseError(f"row: {line.strip()!r}")
    name, rest = m.groups()
    toks = rest.split()
    if len(toks) != len(COLUMNS):
        raise ParseError(f"row {name!r}: {len(toks)} values, expected {len(COLUMNS)}")
    row = {"name": name.rstrip("*").strip(), "marked": name.endswith("*")}
    row.update(zip(COLUMNS, (number(t) for t in toks), strict=True))
    row["raw"] = line.rstrip()
    return row


def section_rows(text: str, title: str, names: tuple[str, ...]) -> dict[str, dict | ParseError]:
    """Rows of `names` in the section `title` ('Emiliano Zapata' matches 'Emiliano Zapata*').
    A name absent from the table is left out; a row that does not parse maps to its error."""
    out: dict[str, dict | ParseError] = {}
    for ln in section(text, title):
        for n in names:
            if re.match(rf"^\s*{re.escape(n)}\*?\s{{2,}}", ln):
                if n in out:
                    out[n] = ParseError(f"{n!r} found twice in {title!r}")
                    continue
                try:
                    out[n] = parse_row(ln)
                except ParseError as e:
                    out[n] = e
    return out


def footnotes(text: str) -> list[str]:
    """Distinct '*' footnotes of the mill/state tables ('*El ingenio inició zafra 2025/26',
    '*El ingenio concluyó su zafra 2025/26', ...): what a row's '*' means in this report."""
    found = re.findall(r"^\s*\*\s*(?:El|La) (?:ingenio|entediad|entidad)[^\n]*", text, re.M)
    return sorted({" ".join(f.split()) for f in found})


def notes_mentioning(text: str, mill: str) -> list[str]:
    """Report notes (not table rows) that name `mill`, e.g. that its area was estimated by
    CONADESUCA because the mill sent only a factory daily. A note may run onto the next line."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if mill not in ln or re.match(rf"^\s*(\d+\s+)?{re.escape(mill)}\*?\s{{2,}}", ln):
            continue
        note = ln.strip()
        if i + 1 < len(lines) and not note.endswith(".") and lines[i + 1].strip():
            note += " " + lines[i + 1].strip()
        if "ingenio" in note.lower():
            out.append(" ".join(note.split()))
    return sorted(set(out))


# Final report ("Reporte final de producción de caña y azúcar")


def final_dates_row(text: str, mill: str) -> dict:
    """Cuadro 1: start, end, real operating days, 1st-estimate days, difference."""
    m = re.search(
        rf"^\s*\d+\s+{re.escape(mill)}\s{{2,}}(\d+ de \w+ de \d{{4}})\s{{2,}}"
        r"(\d+ de \w+ de \d{4})\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$",
        text,
        re.M,
    )
    if not m:
        raise ParseError(f"final report Cuadro 1: no row for {mill!r}")
    s, e, real, est, diff = m.groups()
    return {
        "start": long_date(s),
        "end": long_date(e),
        "days_real": int(real),
        "days_1st_estimate": int(est),
        "days_difference": int(diff),
        "raw": m.group(0).rstrip(),
    }


def final_precision_row(text: str, mill: str) -> dict:
    """Cuadro 8: real end date and % of the 1st estimate for area, cane and sugar."""
    m = re.search(
        rf"^\s*{re.escape(mill)}\s{{2,}}(\d+ de \w+ de \d{{4}})\s+"
        r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$",
        text,
        re.M,
    )
    if not m:
        raise ParseError(f"final report Cuadro 8: no row for {mill!r}")
    end, a, c, s = m.groups()
    return {
        "end": long_date(end),
        "area_pct_of_1st_estimate": float(a),
        "cane_pct_of_1st_estimate": float(c),
        "sugar_pct_of_1st_estimate": float(s),
        "raw": m.group(0).rstrip(),
    }


def karbe(text: str, mill: str) -> dict:
    """KARBE 5-zafra weighted average of `mill` and its rank among all mills in the table.
    The PDF lays three mill columns side by side and some names sit on another line than their
    value, so the rank is taken from all per-mill values (every 'ddd.ddd' after the results
    sentence), not from name-value pairs; the caller checks the count against the note."""
    m = re.search(r"agregado nacional (\d{3}\.\d{3}) kg/tCN", text)
    v = re.search(rf"^\s*{re.escape(mill)}\s+(\d{{3}}\.\d{{3}})\b", text, re.M)
    if not (m and v):
        raise ParseError(f"KARBE: national value or {mill!r} not found")
    values = [float(x) for x in re.findall(r"\b\d{3}\.\d{3}\b", text[m.end() :])]
    value = float(v.group(1))
    return {
        "value_kg_tcn": value,
        "rank": 1 + sum(x > value for x in values),
        "n_mills": len(values),
        "national_kg_tcn": float(m.group(1)),
        "raw": v.group(0).strip(),
    }
