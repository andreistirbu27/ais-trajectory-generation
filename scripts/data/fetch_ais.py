"""
Download AIS zip files from NOAA for a date range, unzip, and optionally
keep a 1-day overlap buffer for cross-boundary vessel continuity.

Usage:
    # Download a week, unzip into data/raw/, delete zips:
    python3 scripts/fetch_ais.py --start 2024-02-01 --end 2024-02-07

    # Download without deleting zips (check space first):
    python3 scripts/fetch_ais.py --start 2024-02-01 --end 2024-02-07 --keep-zip

    # Dry-run: print URLs only, download nothing:
    python3 scripts/fetch_ais.py --start 2024-02-01 --end 2024-02-07 --dry-run

Recommended batch workflow (1-day overlap to preserve cross-boundary tracks):
    # Batch 1: Jan 1–7
    python3 scripts/fetch_ais.py --start 2024-01-01 --end 2024-01-07
    python3 scripts/prepare_data.py data/raw/AIS_2024_01_*.csv \\
        -o data/processed/batch_2024_01_w1.csv
    rm data/raw/AIS_2024_01_01.csv ... data/raw/AIS_2024_01_06.csv
    # keep 2024_01_07.csv → overlap for next batch

    # Batch 2: Jan 7–14  (day 7 already downloaded = overlap)
    python3 scripts/fetch_ais.py --start 2024-01-08 --end 2024-01-14
    python3 scripts/prepare_data.py data/raw/AIS_2024_01_07.csv data/raw/AIS_2024_01_08.csv \\
        ... data/raw/AIS_2024_01_14.csv -o data/processed/batch_2024_01_w2.csv
    rm data/raw/AIS_2024_01_07.csv ... data/raw/AIS_2024_01_13.csv
    # keep 2024_01_14.csv for next overlap

    # Merge all processed batches (dedup on MMSI+BaseDateTime happens in combine()):
    python3 scripts/prepare_data.py data/processed/batch_2024_01_*.csv \\
        -o data/processed/AIS_jan2024.csv
    # (or skip this and just train on all batch files directly)
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_day(d: date, out_dir: Path, keep_zip: bool, dry_run: bool) -> Path | None:
    url = BASE_URL.format(year=d.year, month=d.month, day=d.day)
    zip_path = out_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip"
    csv_path = out_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.csv"

    if csv_path.exists():
        print(f"  [skip] {csv_path.name} already exists")
        return csv_path

    if dry_run:
        print(f"  [dry]  would download {url}")
        return None

    print(f"  downloading {url} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f" {zip_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f" FAILED: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return None

    print(f"  unzipping {zip_path.name} ...", end="", flush=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            csv_names = [n for n in names if n.endswith(".csv")]
            if not csv_names:
                print(f" no CSV inside zip, contents: {names}")
                return None
            zf.extract(csv_names[0], out_dir)
            extracted = out_dir / csv_names[0]
            if extracted != csv_path:
                extracted.rename(csv_path)
            print(f" {csv_path.stat().st_size / 1e6:.1f} MB")
    except zipfile.BadZipFile as e:
        print(f" bad zip: {e}")
        return None
    finally:
        if not keep_zip and zip_path.exists():
            zip_path.unlink()

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA AIS daily zip files for a date range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--out-dir", default="data/raw", help="Directory to save CSV files (default: data/raw)")
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip files after extracting")
    parser.add_argument("--dry-run",  action="store_true", help="Print URLs only, download nothing")
    args = parser.parse_args()

    try:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
    except ValueError as e:
        print(f"Invalid date: {e}")
        sys.exit(1)

    if end < start:
        print("--end must be >= --start")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    days = list(daterange(start, end))
    n_days = len(days)
    total_gb_estimate = n_days * 0.4  # ~400 MB unzipped per day, rough estimate
    print(f"Fetching {n_days} day(s): {start} → {end}")
    print(f"Estimated unzipped size: ~{total_gb_estimate:.1f} GB (rough)")
    print(f"Output dir: {out_dir.resolve()}\n")

    downloaded = []
    failed = []
    for d in days:
        result = download_day(d, out_dir, args.keep_zip, args.dry_run)
        if result:
            downloaded.append(result)
        elif not args.dry_run:
            failed.append(d)

    if not args.dry_run:
        print(f"\nDone: {len(downloaded)} downloaded, {len(failed)} failed")
        if downloaded:
            print("\nNext step — process this batch:")
            csv_list = " ".join(str(p) for p in sorted(downloaded))
            print(f"  python3 scripts/prepare_data.py {csv_list} \\")
            print(f"      -o data/processed/AIS_{start}_{end}.csv")
        if failed:
            print(f"\nFailed dates: {', '.join(str(d) for d in failed)}")


if __name__ == "__main__":
    main()
