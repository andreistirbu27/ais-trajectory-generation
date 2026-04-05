#!/usr/bin/env python3
"""
Stream-append merge of quarterly AIS CSV files into one full-year dataset.

Reads each input CSV one at a time and appends rows to the output file.
Peak RAM usage = size of one quarterly CSV (~350 MB), not all combined.

Usage:
    python3 scripts/data/merge_quarters.py \\
        --inputs data/processed/AIS_2024_Q1.csv \\
                 data/processed/AIS_2024_Q2.csv \\
                 data/processed/AIS_2024_Q3.csv \\
                 data/processed/AIS_2024_Q4.csv \\
        --out data/processed/AIS_2024_full.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(
        description="Stream-merge quarterly AIS CSVs into one full-year file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Quarterly CSV files to merge (in order)")
    p.add_argument("--out", required=True,
                   help="Output path for merged CSV")
    args = p.parse_args()

    inputs   = [Path(f) for f in args.inputs]
    out_path = Path(args.out)

    # Validate inputs exist
    missing = [f for f in inputs if not f.exists()]
    if missing:
        print("ERROR: the following input files do not exist:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    total_size = sum(f.stat().st_size for f in inputs)
    print(f"\nMerging {len(inputs)} file(s) → {out_path}")
    print(f"  Total input size: {total_size / 1e9:.2f} GB")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first      = True
    total_rows = 0

    for csv_path in tqdm(inputs, desc="Merging quarters", unit="file"):
        size_mb = csv_path.stat().st_size / 1e6
        tqdm.write(f"\n  Reading {csv_path.name}  ({size_mb:.0f} MB) ...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_csv(out_path, mode="a" if not first else "w", header=first, index=False)
        total_rows += len(df)
        tqdm.write(f"  ✓ {csv_path.name}: {len(df):,} rows appended  "
                   f"(running total: {total_rows:,})")
        first = False
        del df

    if out_path.exists():
        print(f"\n✓ Done: {total_rows:,} rows → {out_path}")
        print(f"  {out_path.stat().st_size / 1e9:.2f} GB on disk")
        print(f"\nNext step — train on this dataset:")
        print(f"  python3 scripts/train/train_disp.py \\")
        print(f"      --csv {out_path} \\")
        print(f"      --epochs 40 --val_frac 0.15 --seq_len 120 --stride 50 --num_layers 3")
    else:
        print("WARNING: output file was not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
