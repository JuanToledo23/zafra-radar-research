"""List and download NISAR L2 GCOV scenes over the Zacatepec/Jojutla demo site.

Default: print a table of matching scenes and exit.
    python scripts/00_download_nisar.py
Download two specific scenes:
    python scripts/00_download_nisar.py --download --dates 2026-03-11,2026-05-10

Downloads are robust to dropped connections: each file is retried up to 5 times with
exponential backoff, partial files (.part) are resumed with an HTTP Range request when the
server answers 206, idle keep-alive connections are closed before every attempt, and the
final size is checked against the size reported by ASF. A per-file lock (`<file>.lock`,
flock-based) stops two concurrent runs from writing the same partial file.

Search parameters were checked against asf_search 14.1.0 (`inspect.signature(asf.search)`):
dataset, processingLevel, dataMaturity, flightDirection, relativeOrbit, frame and
intersectsWith are all native. For NISAR, asf_search rewrites `frame` from CENTER_ESA_FRAME
to the FRAME_NUMBER attribute (asf_search/CMR/translate.py::use_asf_frame).
"""

import argparse
import fcntl
import re
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import asf_search as asf
import requests
from asf_search.download.download import strip_auth_if_aws
from tqdm import tqdm

from zafra import config
from zafra.paths import NISAR_DIR

POINT_WKT = "POINT(-99.184 18.581)"
TRACK = 19
FRAME = 11
CHUNK = 1024 * 1024
MAX_RETRIES = 5
BACKOFF_BASE = 2  # seconds; wait 2, 4, 8, 16, 32 s before retries 1..5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default="2025-11-15", help="start date YYYY-MM-DD")
    p.add_argument("--end", default="2026-05-21", help="end date YYYY-MM-DD (inclusive)")
    p.add_argument("--download", action="store_true", help="download the .h5 files")
    p.add_argument("--dates", help="comma-separated acquisition dates YYYY-MM-DD to keep")
    p.add_argument("--limit", type=int, help="keep at most N scenes (earliest first)")
    return p.parse_args()


def search(start: str, end: str) -> list[asf.ASFProduct]:
    results = asf.search(
        dataset=asf.DATASET.NISAR,
        processingLevel="GCOV",
        dataMaturity="PROVISIONAL",
        flightDirection="ASCENDING",
        relativeOrbit=TRACK,
        frame=FRAME,
        intersectsWith=POINT_WKT,
        start=f"{start}T00:00:00Z",
        end=f"{end}T23:59:59Z",
    )
    # Belt and braces: re-check on returned properties. `dataMaturity` is not exposed as a
    # product property for NISAR in asf_search 14.1.0, so maturity is checked via the
    # collection short name (e.g. NISAR_L2_GCOV_PROVISIONAL_V1).
    products = [
        p
        for p in results
        if p.properties.get("pathNumber") == TRACK
        and p.properties.get("frameNumber") == FRAME
        and p.properties.get("flightDirection") == "ASCENDING"
        and p.properties.get("processingLevel") == "GCOV"
        and "PROVISIONAL" in (p.properties.get("collectionName") or "")
    ]
    return sorted(products, key=lambda p: p.properties["startTime"])


def h5_size(product: asf.ASFProduct) -> int | None:
    entry = product.properties.get("bytes", {}).get(product.properties["fileName"])
    return entry["bytes"] if entry else None


def acq_date(product: asf.ASFProduct) -> str:
    return product.properties["startTime"][:10]


def print_table(products: list[asf.ASFProduct]) -> None:
    print(f"{'date':<11} {'size_GB':>7}  granule")
    for p in products:
        size = h5_size(p)
        gb = f"{size / 1e9:7.2f}" if size else "    n/a"
        print(f"{acq_date(p):<11} {gb}  {p.properties['sceneName']}")
    total = sum(h5_size(p) or 0 for p in products)
    print(f"\n{len(products)} scene(s), {total / 1e9:.1f} GB total")


class SizeMismatch(Exception):
    """Finished stream does not match the size ASF reported; the file is deleted and retried."""


# Network failures worth retrying. RemoteDisconnected (a stale keep-alive closed by the
# server) surfaces as requests.ConnectionError.
RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ReadTimeout,
    SizeMismatch,
)


def log(msg: str) -> None:
    """Print without breaking an active tqdm bar."""
    tqdm.write(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr)


def drop_idle_connections(session: requests.Session) -> None:
    """Close pooled keep-alive connections so the next request opens a fresh one.

    HTTPAdapter.close() only clears the urllib3 pools; the session (and its Earthdata
    cookies) stays usable and new connections are opened on demand.
    """
    for adapter in session.adapters.values():
        adapter.close()


def fetch_once(url: str, tmp: Path, expected: int | None, session, desc: str) -> None:
    """One download attempt into `tmp`, resuming from its current size if possible."""
    have = tmp.stat().st_size if tmp.exists() else 0
    if expected is not None and have > expected:
        tmp.unlink()  # corrupt leftover, larger than the real file
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    # Same request asf_search's own download_url makes: strip auth on redirect to S3.
    hooks = {"response": strip_auth_if_aws}
    with session.get(url, stream=True, timeout=60, hooks=hooks, headers=headers) as r:
        if have and r.status_code == 416:
            # Range not satisfiable. If the partial is already complete, the size check below
            # accepts it; otherwise drop it so the next attempt starts clean instead of
            # asking for the same impossible range again.
            if expected is None or have != expected:
                tmp.unlink()
                raise SizeMismatch(f"HTTP 416 for range from byte {have}; deleted partial")
        else:
            r.raise_for_status()
            content_range = r.headers.get("Content-Range", "")
            if have and r.status_code == 206 and content_range.startswith(f"bytes {have}-"):
                mode = "ab"
                log(f"{desc}: resuming at {have / 1e9:.2f} GB (server sent 206)")
            else:
                if have:
                    log(f"{desc}: server ignored Range (HTTP {r.status_code}); restarting file")
                mode, have = "wb", 0
            total = expected or (have + int(r.headers.get("Content-Length", 0))) or None
            with (
                open(tmp, mode) as f,
                tqdm(
                    total=total,
                    initial=have,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))

    # Stream ended without a network error, but the byte count is wrong: don't trust the
    # partial, delete it and let the retry loop start this file again.
    size = tmp.stat().st_size if tmp.exists() else 0
    if expected is not None and size != expected:
        tmp.unlink(missing_ok=True)
        raise SizeMismatch(f"got {size} bytes, ASF reports {expected}; deleted partial")


@contextmanager
def download_lock(dest: Path):
    """Exclusive, non-blocking lock on `<file>.lock` so two runs never write the same .part.

    Uses flock: the kernel releases it when the process exits or crashes (even kill -9), so a
    stale lock can never block future runs. Yields False if another process holds it.
    The tiny .lock file is left in place on purpose: deleting it while another process may be
    opening it would let two processes lock two different files.
    """
    lock_path = dest.with_suffix(dest.suffix + ".lock")
    with open(lock_path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def is_complete(dest: Path, expected: int | None) -> bool:
    return dest.exists() and expected is not None and dest.stat().st_size == expected


def download(product: asf.ASFProduct, session: asf.ASFSession) -> bool:
    """Download one product. Returns False if skipped because another run holds its lock."""
    name = product.properties["fileName"]
    dest = NISAR_DIR / name
    expected = h5_size(product)
    if is_complete(dest, expected):
        print(f"skip (already complete): {name}")
        return True

    with download_lock(dest) as locked:
        if not locked:
            log(f"skip (another process is downloading it): {name}")
            return False
        # Re-check under the lock: the other run may have just finished this file.
        if is_complete(dest, expected):
            print(f"skip (already complete): {name}")
            return True

        tmp = dest.with_suffix(dest.suffix + ".part")
        url = product.properties["url"]
        desc = acq_date(product)

        for attempt in range(MAX_RETRIES + 1):
            drop_idle_connections(session)  # never reuse a keep-alive from the previous try
            try:
                fetch_once(url, tmp, expected, session, desc)
                break
            except RETRYABLE as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"{name}: giving up after {MAX_RETRIES} retries (partial, if any: {tmp})"
                    ) from e
                wait = BACKOFF_BASE ** (attempt + 1)  # 2, 4, 8, 16, 32 s
                # Drop query strings: redirected S3 URLs carry temporary signed tokens.
                msg = re.sub(r"\?[^\s'\"]*", "?...", str(e))
                log(
                    f"{desc}: {type(e).__name__}: {msg} -- "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {wait} s"
                )
                time.sleep(wait)

        tmp.rename(dest)
        print(f"done: {name} ({dest.stat().st_size} bytes, matches ASF size)")
        return True


def main() -> int:
    args = parse_args()
    for d in (args.start, args.end):
        date.fromisoformat(d)

    products = search(args.start, args.end)
    if args.dates:
        wanted = {d.strip() for d in args.dates.split(",") if d.strip()}
        for d in wanted:
            date.fromisoformat(d)
        products = [p for p in products if acq_date(p) in wanted]
        missing = wanted - {acq_date(p) for p in products}
        if missing:
            print(f"warning: no scene found for {', '.join(sorted(missing))}", file=sys.stderr)
    if args.limit is not None:
        products = products[: args.limit]

    print_table(products)
    if not args.download:
        print("\nList only. Re-run with --download to fetch (optionally --dates / --limit).")
        return 0

    username, password = config.earthdata_credentials()
    session = asf.ASFSession().auth_with_creds(username, password)
    print(f"\nDownloading to {NISAR_DIR}")
    locked_out = [p.properties["fileName"] for p in products if not download(p, session)]
    if locked_out:
        print(f"\n{len(locked_out)} file(s) skipped because another run is downloading them.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except config.MissingConfigError as e:
        sys.exit(f"error: {e}")
