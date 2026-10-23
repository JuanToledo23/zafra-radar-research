"""Download VIIRS 375 m active-fire detections (FIRMS area API) for the demo bounding box.

    python scripts/00_download_firms.py [--start 2024-11-01] [--end 2026-05-31]

Writes one combined CSV per satellite to data/firms/: SNPP, NOAA-20, NOAA-21.

API format checked against https://firms.modaps.eosdis.nasa.gov/api/area/ (2026-10-23):
    /api/area/csv/[MAP_KEY]/[SOURCE]/[west,south,east,north]/[DAY_RANGE]/[DATE]
DAY_RANGE is 1..5 (not 10) and returns DATE .. DATE+DAY_RANGE-1.
For each day the Standard Processing (SP) source is used if it covers that day, otherwise
Near Real-Time (NRT). Coverage comes from /api/data_availability/csv/[MAP_KEY]/[SOURCE].
"""

import argparse
import io
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from zafra import config
from zafra.paths import FIRMS_DIR

BASE = "https://firms.modaps.eosdis.nasa.gov/api"
BBOX = (-99.35, 18.50, -99.05, 18.75)  # west, south, east, north
MAX_DAYS = 5
SATELLITES = {"snpp": "VIIRS_SNPP", "noaa20": "VIIRS_NOAA20", "noaa21": "VIIRS_NOAA21"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default="2024-11-01", type=date.fromisoformat)
    p.add_argument("--end", default="2026-05-31", type=date.fromisoformat)
    return p.parse_args()


def get_csv(url: str, key: str) -> str:
    """GET a FIRMS CSV endpoint. The key is redacted from any error message."""
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(str(e).replace(key, "<MAP_KEY>")) from None
    text = r.text.strip()
    # FIRMS answers errors (bad key, bad source, quota) with HTTP 200 and a one-line message,
    # so anything whose first line is not a CSV header is treated as an error.
    if text and "," not in text.splitlines()[0]:
        raise RuntimeError(f"FIRMS error: {text[:200].replace(key, '<MAP_KEY>')}")
    return text


def availability(key: str, source: str) -> tuple[date, date] | None:
    text = get_csv(f"{BASE}/data_availability/csv/{key}/{source}", key)
    if not text:
        return None
    df = pd.read_csv(io.StringIO(text))
    row = df[df["data_id"] == source]
    if row.empty:
        return None
    return date.fromisoformat(row["min_date"].iloc[0]), date.fromisoformat(row["max_date"].iloc[0])


def plan_chunks(start: date, end: date, sp: tuple | None, nrt: tuple | None):
    """Yield (source_suffix, chunk_start, n_days) covering start..end, preferring SP."""

    def pick(d: date) -> str | None:
        if sp and sp[0] <= d <= sp[1]:
            return "SP"
        if nrt and nrt[0] <= d <= nrt[1]:
            return "NRT"
        return None

    d = start
    while d <= end:
        src = pick(d)
        n = 1
        while n < MAX_DAYS and d + timedelta(n) <= end and pick(d + timedelta(n)) == src:
            n += 1
        if src:
            yield src, d, n
        d += timedelta(n)


def fetch_satellite(key: str, name: str, prefix: str, start: date, end: date) -> pd.DataFrame:
    sp = availability(key, f"{prefix}_SP")
    nrt = availability(key, f"{prefix}_NRT")
    print(f"{name}: SP {sp or 'unavailable'}, NRT {nrt or 'unavailable'}")
    chunks = list(plan_chunks(start, end, sp, nrt))
    covered = sum(n for _, _, n in chunks)
    total = (end - start).days + 1
    if covered < total:
        print(f"  note: {total - covered} day(s) in range not covered by SP or NRT")

    bbox = ",".join(str(v) for v in BBOX)
    frames = []
    for suffix, d, n in tqdm(chunks, desc=name, unit="req"):
        source = f"{prefix}_{suffix}"
        text = get_csv(f"{BASE}/area/csv/{key}/{source}/{bbox}/{n}/{d.isoformat()}", key)
        df = pd.read_csv(io.StringIO(text)) if text else pd.DataFrame()
        if not df.empty:
            df["source"] = source
            frames.append(df)
        time.sleep(0.2)  # stay well under 5000 transactions / 10 min
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    args = parse_args()
    if args.end < args.start:
        sys.exit("--end must be on or after --start")
    key = config.firms_map_key()

    for name, prefix in SATELLITES.items():
        df = fetch_satellite(key, name, prefix, args.start, args.end)
        out = FIRMS_DIR / f"firms_viirs_{name}_{args.start}_{args.end}.csv"
        df.to_csv(out, index=False)
        print(f"  {len(df)} detections -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except config.MissingConfigError as e:
        sys.exit(f"error: {e}")
