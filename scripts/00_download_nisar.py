"""List and download NISAR L2 GCOV or GUNW scenes over a site (default: Morelos).

Default: print a table of matching GCOV ascending scenes and exit.
    python scripts/00_download_nisar.py
Download two specific scenes:
    python scripts/00_download_nisar.py --download --dates 2026-03-11,2026-05-10
Other products / passes (without --track/--frame, every track/frame covering the point is
listed; if exactly one is found it is used):
    python scripts/00_download_nisar.py --product GUNW --track 19 --frame 11
    python scripts/00_download_nisar.py --direction DESCENDING
Another site (configs/<site>.yaml; search point = AOI centre, dates = nisar.search):
    python scripts/00_download_nisar.py --site brazil --product GCOV --track 146 --frame 167

Output folder: the `dir` of the matching nisar.sources entry of the site config, otherwise
<nisar.root>/<product>_<direction>_t<track>f<frame>/. For Morelos that keeps GCOV ascending
track 19 frame 11 in data/nisar/, where the earlier exercises read it.
For GUNW, "date" is the reference (first) acquisition of the interferometric pair; --dates
matches it.

Downloads are robust to dropped connections: each file is retried up to 5 times with
exponential backoff, partial files (.part) are resumed with an HTTP Range request when the
server answers 206, idle keep-alive connections are closed before every attempt, and the
final size is checked against the size reported by ASF. A per-file lock (`<file>.lock`,
flock-based) stops two concurrent runs from writing the same partial file.

Search parameters were checked against asf_search 14.1.0 (`inspect.signature(asf.search)`):
dataset, processingLevel, dataMaturity, flightDirection, relativeOrbit, frame and
intersectsWith are all native. "GCOV" and "GUNW" are asf.PRODUCT_TYPE constants (CMR attribute
PROCESSING_TYPE); "ASCENDING"/"DESCENDING" are asf.FLIGHT_DIRECTION constants (CMR attribute
ASCENDING_DESCENDING). For NISAR, asf_search rewrites `frame` from CENTER_ESA_FRAME
to the FRAME_NUMBER attribute (asf_search/CMR/translate.py::use_asf_frame).
"""

import argparse
import fcntl
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import asf_search as asf
import requests
from asf_search.download.download import strip_auth_if_aws
from tqdm import tqdm

from zafra import config
from zafra.site import add_site_argument, load_site

MORELOS_POINT = (-99.184, 18.581)  # the search point used before site configs (exercise 5)
CHUNK = 1024 * 1024
MAX_RETRIES = 5
BACKOFF_BASE = 2  # seconds; wait 2, 4, 8, 16, 32 s before retries 1..5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_site_argument(p)
    p.add_argument("--product", choices=["GCOV", "GUNW"], default="GCOV")
    p.add_argument("--direction", choices=["ASCENDING", "DESCENDING"], default="ASCENDING")
    p.add_argument("--track", type=int, help="relative orbit; omit to search by point")
    p.add_argument("--frame", type=int, help="frame number; omit to search by point")
    p.add_argument("--out-dir", type=Path, help="default: data/nisar/<product>_<dir>_t<N>f<N>/")
    p.add_argument("--start", help="start date YYYY-MM-DD (default: site nisar.search)")
    p.add_argument("--end", help="end date YYYY-MM-DD, inclusive (default: site nisar.search)")
    p.add_argument("--download", action="store_true", help="download the .h5 files")
    p.add_argument("--dates", help="comma-separated acquisition dates YYYY-MM-DD to keep")
    p.add_argument("--limit", type=int, help="keep at most N scenes (earliest first)")
    return p.parse_args()


def search(
    point_wkt: str,
    product: str,
    direction: str,
    track: int | None,
    frame: int | None,
    start: str,
    end: str,
) -> list[asf.ASFProduct]:
    """PROVISIONAL products intersecting the point. track/frame of None means any.

    ASF matches on temporal overlap, so a GUNW pair that straddles `start` or `end` is
    included. Every BETA product seen so far has a PROVISIONAL reprocessing of the same
    acquisition(s), so BETA is never requested.
    """
    opts = {"relativeOrbit": track} if track is not None else {}
    if frame is not None:
        opts["frame"] = frame
    results = asf.search(
        dataset=asf.DATASET.NISAR,
        processingLevel=product,
        dataMaturity="PROVISIONAL",
        flightDirection=direction,
        intersectsWith=point_wkt,
        start=f"{start}T00:00:00Z",
        end=f"{end}T23:59:59Z",
        **opts,
    )
    # Belt and braces: re-check on returned properties. `dataMaturity` is not exposed as a
    # product property for NISAR in asf_search 14.1.0, so maturity is checked via the
    # collection short name (e.g. NISAR_L2_GCOV_PROVISIONAL_V1).
    products = [
        p
        for p in results
        if (track is None or p.properties.get("pathNumber") == track)
        and (frame is None or p.properties.get("frameNumber") == frame)
        and p.properties.get("flightDirection") == direction
        and p.properties.get("processingLevel") == product
        and "PROVISIONAL" in (p.properties.get("collectionName") or "")
    ]
    return sorted(products, key=lambda p: p.properties["startTime"])


def track_frame(product: asf.ASFProduct) -> tuple[int, int]:
    return product.properties["pathNumber"], product.properties["frameNumber"]


def h5_size(product: asf.ASFProduct) -> int | None:
    entry = product.properties.get("bytes", {}).get(product.properties["fileName"])
    return entry["bytes"] if entry else None


def acq_date(product: asf.ASFProduct) -> str:
    """Acquisition date; for GUNW, the reference (first) date of the pair."""
    return product.properties["startTime"][:10]


def date_label(product: asf.ASFProduct) -> str:
    """'YYYY-MM-DD' for GCOV, 'YYYY-MM-DD/YYYY-MM-DD' (reference/secondary) for GUNW.

    GUNW names carry four timestamps: reference start/stop, then secondary start/stop.
    """
    stamps = re.findall(r"_(\d{8})T\d{6}", product.properties["sceneName"])
    if product.properties["processingLevel"] == "GUNW" and len(stamps) >= 3:
        ref, sec = (date.fromisoformat(f"{d[:4]}-{d[4:6]}-{d[6:]}") for d in stamps[0:3:2])
        return f"{ref}/{sec}"
    return acq_date(product)


def print_table(products: list[asf.ASFProduct]) -> None:
    groups = defaultdict(list)
    for p in products:
        groups[track_frame(p)].append(p)
    width = max((len(date_label(p)) for p in products), default=10) + 1
    for (track, frame), group in sorted(groups.items()):
        print(f"\ntrack {track} frame {frame}")
        print(f"{'date':<{width}} {'size_GB':>7}  granule")
        for p in group:
            size = h5_size(p)
            gb = f"{size / 1e9:7.2f}" if size else "    n/a"
            print(f"{date_label(p):<{width}} {gb}  {p.properties['sceneName']}")
    total = sum(h5_size(p) or 0 for p in products)
    print(f"\n{len(products)} scene(s), {len(groups)} track/frame, {total / 1e9:.1f} GB total")


def default_out_dir(site, product: str, direction: str, track: int, frame: int) -> Path:
    for s in site.nisar.values():
        if (s.product, s.direction, s.track, s.frame) == (product, direction, track, frame):
            return s.dir
    return site.nisar_root / f"{product}_{direction}_t{track}f{frame}"


def search_point(site) -> str:
    lon, lat = MORELOS_POINT if site.name == "morelos" else site.center
    return f"POINT({lon} {lat})"


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


def download(product: asf.ASFProduct, session: asf.ASFSession, out_dir: Path) -> bool:
    """Download one product. Returns False if skipped because another run holds its lock."""
    name = product.properties["fileName"]
    dest = out_dir / name
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
        desc = date_label(product)

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
    site = load_site(args.site, register=False)
    args.start = args.start or site.nisar_search[0]
    args.end = args.end or site.nisar_search[1]
    for d in (args.start, args.end):
        date.fromisoformat(d)

    products = search(
        search_point(site),
        args.product,
        args.direction,
        args.track,
        args.frame,
        args.start,
        args.end,
    )
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

    groups = {track_frame(p) for p in products}
    if len(groups) > 1:
        sys.exit("error: several track/frame groups found; pick one with --track and --frame")
    if not products:
        print("nothing to download")
        return 0
    out_dir = args.out_dir or default_out_dir(site, args.product, args.direction, *groups.pop())
    out_dir.mkdir(parents=True, exist_ok=True)

    username, password = config.earthdata_credentials()
    session = asf.ASFSession().auth_with_creds(username, password)
    print(f"\nDownloading to {out_dir}")
    locked_out = [p.properties["fileName"] for p in products if not download(p, session, out_dir)]
    if locked_out:
        print(f"\n{len(locked_out)} file(s) skipped because another run is downloading them.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except config.MissingConfigError as e:
        sys.exit(f"error: {e}")
