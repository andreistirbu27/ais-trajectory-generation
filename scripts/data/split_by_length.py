#!/usr/bin/env python3
"""
Split a processed AIS CSV into two files by track length.

Tracks with fewer than --threshold pings go to --short_out (archive).
Tracks with >= --threshold pings go to --long_out (training).

Uses two passes so peak RAM = one chunk at a time, not the full file.

Usage:
    python3 scripts/data/split_by_length.py \\
        --csv data/processed/AIS_2024_full.csv \\
        --short_out data/processed/AIS_2024_le80.csv \\
        --long_out  data/processed/AIS_2024_gt80.csv \\
        --threshold 81
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(
        description="Split AIS CSV into short/long tracks by ping count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",       required=True, help="Input processed AIS CSV")
    p.add_argument("--short_out", required=True, help="Output path for short tracks (< threshold)")
    p.add_argument("--long_out",  required=True, help="Output path for long tracks (>= threshold)")
    p.add_argument("--threshold", type=int, default=81,
                   help="Min pings to go into long_out (default: 81)")
    p.add_argument("--chunksize", type=int, default=500_000,
                   help="Rows per chunk (default: 500000)")
    args = p.parse_args()

    csv_path  = Path(args.csv)
    short_out = Path(args.short_out)
    long_out  = Path(args.long_out)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    short_out.parent.mkdir(parents=True, exist_ok=True)
    long_out.parent.mkdir(parents=True, exist_ok=True)

    file_size_gb = csv_path.stat().st_size / 1e9
    print(f"\nInput: {csv_path}  ({file_size_gb:.1f} GB)")
    print(f"  Short (<{args.threshold} pings) → {short_out}")
    print(f"  Long  (≥{args.threshold} pings) → {long_out}")
    print(f"  Chunk size: {args.chunksize:,} rows\n")

    # ── Pass 1: count pings per MMSI ──────────────────────────────────────────
    print("Pass 1/2: counting pings per track ...")
    counts: dict = {}
    chunks = pd.read_csv(csv_path, usecols=["MMSI"], chunksize=args.chunksize,
                         low_memory=False)
    for chunk in tqdm(chunks, desc="  Pass 1", unit="chunk"):
        for mmsi, cnt in chunk["MMSI"].value_counts().items():
            counts[mmsi] = counts.get(mmsi, 0) + cnt

    short_ids = {m for m, c in counts.items() if c < args.threshold}
    long_ids  = {m for m, c in counts.items() if c >= args.threshold}

    total_tracks = len(counts)
    short_rows = sum(c for m, c in counts.items() if m in short_ids)
    long_rows  = sum(c for m, c in counts.items() if m in long_ids)
    total_rows = short_rows + long_rows

    print(f"\n  Total tracks : {total_tracks:,}")
    print(f"  Short tracks : {len(short_ids):,}  ({len(short_ids)/total_tracks*100:.1f}%)"
          f"  →  {short_rows:,} rows  ({short_rows/total_rows*100:.1f}%)")
    print(f"  Long  tracks : {len(long_ids):,}  ({len(long_ids)/total_tracks*100:.1f}%)"
          f"  →  {long_rows:,} rows  ({long_rows/total_rows*100:.1f}%)\n")

    # ── Pass 2: split rows into output files ──────────────────────────────────
    print("Pass 2/2: writing output files ...")
    first_short = first_long = True
    written_short = written_long = 0

    chunks = pd.read_csv(csv_path, chunksize=args.chunksize, low_memory=False)
    for chunk in tqdm(chunks, desc="  Pass 2", unit="chunk"):
        s = chunk[chunk["MMSI"].isin(short_ids)]
        l = chunk[chunk["MMSI"].isin(long_ids)]

        if not s.empty:
            s.to_csv(short_out, mode="a" if not first_short else "w",
                     header=first_short, index=False)
            written_short += len(s)
            first_short = False

        if not l.empty:
            l.to_csv(long_out, mode="a" if not first_long else "w",
                     header=first_long, index=False)
            written_long += len(l)
            first_long = False

    print(f"\n✓ Done")
    print(f"  {short_out.name}: {written_short:,} rows  "
          f"({short_out.stat().st_size / 1e9:.2f} GB)")
    print(f"  {long_out.name}:  {written_long:,} rows  "
          f"({long_out.stat().st_size / 1e9:.2f} GB)")
    print(f"\nNext: train on the long file:")
    print(f"  python3 scripts/train/train_disp.py \\")
    print(f"      --csv {long_out} \\")
    print(f"      --epochs 40 --val_frac 0.15 --seq_len 120 --stride 50 --num_layers 3 \\")
    print(f"      --out_dir runs/ais_transformer_v4")


if __name__ == "__main__":
    main()
