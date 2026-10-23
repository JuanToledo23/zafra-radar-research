"""List and download NISAR L2 GCOV scenes over the Zacatepec/Jojutla demo site.

Default: print a table of matching scenes and exit.
    python scripts/00_download_nisar.py
Download two specific scenes:
    python scripts/00_download_nisar.py --download --dates 2026-03-11,2026-05-10

Search parameters were checked against asf_search 14.1.0 (`inspect.signature(asf.search)`):
dataset, processingLevel, dataMaturity, flightDirection, relativeOrbit, frame and
intersectsWith are all native. For NISAR, asf_search rewrites `frame` from CENTER_ESA_FRAME
to the FRAME_NUMBER attribute (asf_search/CMR/translate.py::use_asf_frame).
"""

import argparse
import sys
from datetime import date

import asf_search as asf
from asf_search.download.download import strip_auth_if_aws
from tqdm import tqdm

from zafra import config
from zafra.paths import NISAR_DIR

POINT_WKT = "POINT(-99.184 18.581)"
TRACK = 19
FRAME = 11
CHUNK = 1024 * 1024


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


def download(product: asf.ASFProduct, session: asf.ASFSession) -> None:
    name = product.properties["fileName"]
    dest = NISAR_DIR / name
    expected = h5_size(product)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        print(f"skip (already complete): {name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    # Same request asf_search's own download_url makes: strip auth on redirect to S3.
    url = product.properties["url"]
    hooks = {"response": strip_auth_if_aws}
    with session.get(url, stream=True, timeout=60, hooks=hooks) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or expected
        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=acq_date(product)
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                bar.update(len(chunk))

    if expected is not None and tmp.stat().st_size != expected:
        raise RuntimeError(
            f"{name}: size mismatch ({tmp.stat().st_size} != {expected}); "
            f"partial file kept at {tmp}"
        )
    tmp.rename(dest)


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
    for p in products:
        download(p, session)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except config.MissingConfigError as e:
        sys.exit(f"error: {e}")
