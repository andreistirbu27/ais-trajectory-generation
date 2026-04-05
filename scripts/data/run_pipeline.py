"""
Batch AIS data pipeline: download → process → merge.

Downloads weekly batches from NOAA with 1-day overlap (so vessels crossing week
boundaries are not split and dropped), processes each batch with prepare_data.py,
then stream-appends all batches into a single merged CSV without loading everything
into RAM at once.

Usage:
    # Dry run (print plan, download/process nothing):
    python3 scripts/data/run_pipeline.py \\
        --start 2024-01-01 --end 2024-03-31 \\
        --proc_dir data/processed/batches/q1 \\
        --out data/processed/AIS_2024_Q1.csv --dry_run

    # Full run — Q1 (saves to separate quarterly file):
    python3 scripts/data/run_pipeline.py \\
        --start 2024-01-01 --end 2024-03-31 \\
        --proc_dir data/processed/batches/q1 \\
        --out data/processed/AIS_2024_Q1.csv \\
        --batch_start_idx 0

    # Q2 with overlap (starts March 31) and offset batch IDs:
    python3 scripts/data/run_pipeline.py \\
        --start 2024-03-31 --end 2024-06-30 \\
        --proc_dir data/processed/batches/q2 \\
        --out data/processed/AIS_2024_Q2.csv \\
        --batch_start_idx 13

Overlap logic:
    Each batch starts on the last day of the previous batch.  prepare_data.py's
    combine() step deduplicates (MMSI, BaseDateTime) before segmentation, so the
    shared day's pings are counted once.  Vessels with pings spanning the boundary
    therefore accumulate enough points to survive the 50-point minimum filter.

Segment ID collision prevention:
    After processing, the MMSI column is prefixed with "b{NNN:03d}_" (e.g.
    "b000_368123450_0").  Use --batch_start_idx when running multiple quarters so
    IDs are globally unique in the final merged file.

Resumability:
    If proc_dir/batch_NNN.csv already exists, that batch is skipped.
    If a raw CSV already exists in raw_dir, that day's download is skipped.
    Safe to interrupt and re-run at any point.
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow importing download_day from fetch_ais.py (same directory)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
from fetch_ais import download_day  # noqa: E402


# ── Batch construction ─────────────────────────────────────────────────────────

def make_batches(start: date, end: date, batch_days: int):
    """
    Return list of day lists with 1-day overlap between adjacent batches.

    Each list entry is [day_0, day_1, ..., day_N] where day_N is shared with
    the next batch's day_0.  The last batch may be shorter than batch_days.
    """
    batches = []
    d = start
    while d < end:
        days = []
        for i in range(batch_days):
            candidate = d + timedelta(days=i)
            if candidate > end:
                break
            days.append(candidate)
        batches.append(days)
        d = days[-1]  # next batch starts on last day of this one = overlap
        if d == days[0]:  # safety: no progress, shouldn't happen
            break
    return batches


# ── Per-batch processing ───────────────────────────────────────────────────────

def _download_with_progress(d: date, out_dir: Path, keep_zip: bool) -> Path | None:
    """Download one day with a tqdm progress bar. Returns csv path or None."""
    url_template = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
    url      = url_template.format(year=d.year, month=d.month, day=d.day)
    zip_path = out_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip"
    csv_path = out_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.csv"

    if csv_path.exists():
        tqdm.write(f"    [skip] {csv_path.name} already exists")
        return csv_path

    pbar = tqdm(unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"    {d}", leave=False)

    def _reporthook(block_num, block_size, total_size):
        if total_size > 0 and pbar.total is None:
            pbar.total = total_size
        pbar.update(block_size)

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_reporthook)
        pbar.close()
        tqdm.write(f"    Downloaded {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        pbar.close()
        tqdm.write(f"    FAILED to download {d}: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return None

    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                tqdm.write(f"    No CSV inside {zip_path.name}")
                return None
            zf.extract(csv_names[0], out_dir)
            extracted = out_dir / csv_names[0]
            if extracted != csv_path:
                extracted.rename(csv_path)
        tqdm.write(f"    Unzipped → {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        tqdm.write(f"    Bad zip {zip_path.name}: {e}")
        return None
    finally:
        if not keep_zip and zip_path.exists():
            zip_path.unlink()

    return csv_path


def process_batch(batch_idx: int, days: list, raw_dir: Path, proc_dir: Path,
                  delete_raw: bool, dry_run: bool) -> Path | None:
    """
    Download + process one batch.  Returns path to batch CSV or None on failure.
    """
    batch_out = proc_dir / f"batch_{batch_idx:03d}.csv"

    if batch_out.exists():
        tqdm.write(f"\n  [batch {batch_idx:03d}] Already processed → {batch_out.name}  (skip)")
        return batch_out

    date_str = f"{days[0]} → {days[-1]}"
    tqdm.write(f"\n  [batch {batch_idx:03d}] {date_str}  ({len(days)} days)")
    tqdm.write("  " + "─" * 58)

    # 1. Download missing days
    raw_csvs = []
    for d in days:
        csv = raw_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.csv"
        if dry_run:
            tqdm.write(f"    [dry] would download {d}")
        else:
            result = _download_with_progress(d, raw_dir, keep_zip=False)
            if result:
                raw_csvs.append(result)
            else:
                tqdm.write(f"    WARNING: failed to download {d} — skipping day")

    if dry_run:
        tqdm.write(f"  [dry] would run: prepare_data.py {len(days)} files → {batch_out.name}")
        tqdm.write(f"  [dry] would prefix MMSI with b{batch_idx:03d}_")
        if delete_raw:
            tqdm.write(f"  [dry] would delete {len(days) - 1} raw CSVs (keep {days[-1]})")
        return None

    if not raw_csvs:
        tqdm.write(f"  No raw CSVs available for batch {batch_idx:03d} — skipping")
        return None

    # 2. Process with prepare_data.py
    tqdm.write(f"  Processing {len(raw_csvs)} day(s) → {batch_out.name} ...")
    cmd = [
        sys.executable,
        os.path.join(_scripts_dir, "prepare_data.py"),
        *[str(p) for p in sorted(raw_csvs)],
        "-o", str(batch_out),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        tqdm.write(f"  ERROR: prepare_data.py failed for batch {batch_idx:03d}")
        return None

    if not batch_out.exists():
        tqdm.write(f"  ERROR: expected output {batch_out} not created")
        return None

    # 3. Add batch prefix to MMSI column
    tqdm.write(f"  Prefixing MMSI with b{batch_idx:03d}_ ...")
    df = pd.read_csv(batch_out, low_memory=False)
    prefix = f"b{batch_idx:03d}_"
    df["MMSI"] = prefix + df["MMSI"].astype(str)
    df.to_csv(batch_out, index=False)
    n_vessels = df["MMSI"].nunique()
    tqdm.write(f"  ✓ Batch {batch_idx:03d} done — {len(df):,} rows, {n_vessels:,} vessels "
               f"({batch_out.stat().st_size / 1e6:.0f} MB)")
    del df

    # 4. Delete raw CSVs except last day (overlap buffer)
    if delete_raw:
        for d in days[:-1]:
            csv = raw_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.csv"
            if csv.exists():
                csv.unlink()
        tqdm.write(f"  Deleted {len(days) - 1} raw CSVs, kept {days[-1]} as overlap buffer")

    return batch_out


# ── Stream-append merge ────────────────────────────────────────────────────────

def merge_batches(batch_csvs: list, out_path: Path, dry_run: bool):
    """
    Write all batch CSVs to out_path one at a time.
    Peak RAM = one batch at a time (~450 MB), not all at once.
    """
    if dry_run:
        total = sum(p.stat().st_size for p in batch_csvs if p.exists())
        print(f"\n[dry] would stream-merge {len(batch_csvs)} batches → {out_path}")
        print(f"[dry] estimated output size: ~{total / 1e9:.1f} GB")
        return

    print(f"\nMerging {len(batch_csvs)} batch(es) → {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = True
    total_rows = 0
    for batch_csv in tqdm(sorted(batch_csvs), desc="Merging batches", unit="batch"):
        if not batch_csv.exists():
            tqdm.write(f"  WARNING: {batch_csv.name} not found — skipping")
            continue
        df = pd.read_csv(batch_csv, low_memory=False)
        df.to_csv(out_path, mode="a" if not first else "w", header=first, index=False)
        total_rows += len(df)
        tqdm.write(f"  Appended {batch_csv.name}: {len(df):,} rows")
        first = False
        del df

    if out_path.exists():
        print(f"\n✓ Done: {total_rows:,} rows → {out_path}")
        print(f"  {out_path.stat().st_size / 1e9:.2f} GB on disk")
    else:
        print("WARNING: no batches were merged")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download, process, and merge AIS data in weekly batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start",           required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",             required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--batch_days",      type=int, default=7,
                        help="Days per batch (default: 7)")
    parser.add_argument("--batch_start_idx", type=int, default=0,
                        help="Offset batch numbering to avoid ID collisions across quarters (default: 0)")
    parser.add_argument("--raw_dir",         default="data/raw",
                        help="Directory for downloaded raw CSVs (default: data/raw)")
    parser.add_argument("--proc_dir",        default="data/processed/batches",
                        help="Directory for per-batch processed CSVs")
    parser.add_argument("--out",             required=True,
                        help="Path for the final merged processed CSV")
    parser.add_argument("--delete_raw",      action="store_true",
                        help="Delete raw CSVs after each batch (keeps last day as overlap buffer)")
    parser.add_argument("--dry_run",         action="store_true",
                        help="Print plan only — download/process/merge nothing")
    args = parser.parse_args()

    try:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
    except ValueError as e:
        print(f"Invalid date: {e}")
        sys.exit(1)

    raw_dir  = Path(args.raw_dir)
    proc_dir = Path(args.proc_dir)
    out_path = Path(args.out)

    if not args.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

    batches = make_batches(start, end, args.batch_days)
    n_days_total = sum(len(b) for b in batches)

    print(f"\nAIS Batch Pipeline")
    print(f"  Date range      : {start} → {end}")
    print(f"  Batches         : {len(batches)}  ({args.batch_days} days each, 1-day overlap)")
    print(f"  Batch ID offset : {args.batch_start_idx} "
          f"(IDs: b{args.batch_start_idx:03d}_…b{args.batch_start_idx + len(batches) - 1:03d}_)")
    print(f"  Total days      : {n_days_total} (with overlap)")
    print(f"  Raw dir         : {raw_dir}")
    print(f"  Proc dir        : {proc_dir}")
    print(f"  Output          : {out_path}")
    print(f"  Delete raw      : {args.delete_raw}")
    if args.dry_run:
        print(f"  DRY RUN         : no files will be written\n")

    # Process all batches
    succeeded, failed_idxs = [], []
    pbar = tqdm(enumerate(batches), total=len(batches), desc="Batches", unit="batch")
    for i, days in pbar:
        batch_idx = args.batch_start_idx + i
        pbar.set_description(f"Batch {batch_idx:03d} ({days[0]} → {days[-1]})")
        result = process_batch(batch_idx, days, raw_dir, proc_dir, args.delete_raw, args.dry_run)
        if result:
            succeeded.append(result)
        elif not args.dry_run:
            failed_idxs.append(batch_idx)

    # Report failed batches before merge
    if failed_idxs and not args.dry_run:
        print(f"\n⚠  {len(failed_idxs)} batch(es) failed and will be missing from output:")
        for idx in failed_idxs:
            print(f"   batch_{idx:03d}.csv")
        print("   Re-run the same command to retry (skips already-processed batches).")

    if not args.dry_run and not succeeded:
        print("\nNo batches produced output — nothing to merge.")
        sys.exit(1)

    # For dry_run, use expected paths for reporting
    if args.dry_run:
        succeeded = [proc_dir / f"batch_{args.batch_start_idx + i:03d}.csv"
                     for i in range(len(batches))]

    merge_batches(succeeded, out_path, args.dry_run)

    if not args.dry_run and out_path.exists():
        print(f"\nNext step — train on this output:")
        print(f"  python3 scripts/train/train_disp.py \\")
        print(f"      --csv {out_path} \\")
        print(f"      --epochs 40 --val_frac 0.15 --seq_len 120 --stride 50 --num_layers 3")


if __name__ == "__main__":
    main()
