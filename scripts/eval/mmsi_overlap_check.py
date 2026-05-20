"""mmsi_overlap_check.py -- quantify train/val MMSI leakage between corpora.

The production v10 checkpoint was trained on `trajgen_128.npz` (unfiltered, ~207k
tracks) but evaluated on `trajgen_128_clean.npz` (E1-filtered, ~61k tracks).
The split inside each NPZ is MMSI-grouped by `train_val_split_gen` with seed=42,
so within a single NPZ there is no leakage. But the same physical MMSI can land
in *unfiltered-train* (because its dirty segments survived filtering for training)
and in *cleaned-val* (because some of its clean segments happen to be selected
for the cleaned-corpus val partition).

This script reports |train_unfiltered ∩ val_clean| as a fraction of |val_clean|.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_gen import _root_id, train_val_split_gen  # noqa: E402

DIRTY_NPZ = ROOT / "data/processed/trajgen_128.npz"
CLEAN_NPZ = ROOT / "data/processed/trajgen_128_clean.npz"
OUT       = ROOT / "results/mmsi_overlap.txt"

SEED = 42
VAL_FRAC = 0.15


def load_split(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    traj = data["trajectories"]
    vtypes = data["vessel_types"]
    tids = data["track_ids"]
    tr_traj, tr_vt, tr_ids, va_traj, va_vt, va_ids = train_val_split_gen(
        traj, vtypes, tids, val_frac=VAL_FRAC, seed=SEED,
    )
    train_mmsis = {_root_id(str(t)) for t in tr_ids}
    val_mmsis   = {_root_id(str(t)) for t in va_ids}
    return train_mmsis, val_mmsis, len(tr_ids), len(va_ids)


def main():
    print(f"Loading {DIRTY_NPZ.name} …")
    dirty_train_mmsis, dirty_val_mmsis, n_dt, n_dv = load_split(DIRTY_NPZ)
    print(f"  train tracks={n_dt:,}  val tracks={n_dv:,}")
    print(f"  unique train MMSIs={len(dirty_train_mmsis):,}  val MMSIs={len(dirty_val_mmsis):,}")

    print(f"Loading {CLEAN_NPZ.name} …")
    clean_train_mmsis, clean_val_mmsis, n_ct, n_cv = load_split(CLEAN_NPZ)
    print(f"  train tracks={n_ct:,}  val tracks={n_cv:,}")
    print(f"  unique train MMSIs={len(clean_train_mmsis):,}  val MMSIs={len(clean_val_mmsis):,}")

    # The load-bearing comparison: production v10 trained on dirty_train,
    # evaluated on clean_val. Quantify how many clean_val MMSIs leaked from
    # dirty_train.
    overlap_dirty_train_clean_val = dirty_train_mmsis & clean_val_mmsis
    pct_leak = 100.0 * len(overlap_dirty_train_clean_val) / max(1, len(clean_val_mmsis))

    # Sanity: within-corpus split is leak-free (should be zero by construction).
    within_dirty = dirty_train_mmsis & dirty_val_mmsis
    within_clean = clean_train_mmsis & clean_val_mmsis

    lines = []
    lines.append("MMSI overlap check (seed=42, val_frac=0.15, MMSI-grouped)")
    lines.append("=" * 68)
    lines.append(f"dirty_train ∩ dirty_val:  {len(within_dirty):>6,}  (must be 0)")
    lines.append(f"clean_train ∩ clean_val:  {len(within_clean):>6,}  (must be 0)")
    lines.append("")
    lines.append("Production-eval leakage (dirty_train_mmsis ∩ clean_val_mmsis):")
    lines.append(f"  dirty_train unique MMSIs: {len(dirty_train_mmsis):>7,}")
    lines.append(f"  clean_val   unique MMSIs: {len(clean_val_mmsis):>7,}")
    lines.append(f"  intersection             : {len(overlap_dirty_train_clean_val):>7,}")
    lines.append(f"  fraction of clean_val    : {pct_leak:>7.2f} %")
    out_text = "\n".join(lines)
    print()
    print(out_text)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(out_text + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
