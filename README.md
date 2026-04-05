# PRL: AIS Trajectory Generation

Transformer-based next-step prediction and synthetic maritime trajectory generation from AIS tracklines.

## Structure
- `scripts/data/`   data pipeline: fetch, prepare, batch pipeline, split utilities
- `scripts/train/`  training: displacement target (`train_disp.py`), velocity target (`train_vel.py`)
- `scripts/eval/`   evaluation: metrics, visualizations, diagnostics
- `src/`            reusable modules: data pipeline, model, metrics
- `configs/`        YAML experiment configs (one per run)
- `notebooks/`      exploratory notebooks
- `docs/references/` papers and links
- `data/`           datasets (not tracked)
- `runs/`           training outputs / checkpoints (not tracked)
- `outputs/`        evaluation plots and diagnostics (not tracked)

## Data Pipeline

```
Raw AIS CSVs (marinecadastre.gov / NOAA)
    ↓  scripts/data/fetch_ais.py          # download daily zip files
    ↓  scripts/data/run_pipeline.py       # batch download → filter → merge (weekly batches)
    ↓  scripts/data/merge_quarters.py     # merge quarterly CSVs into full year
    ↓  scripts/data/split_by_length.py    # split by track length (le80 / gt80)
    →  data/processed/AIS_2024_gt80.csv   # training-ready dataset
```

### Full-year pipeline (RAM-safe, ~16 GB)
```bash
bash scripts/data/run_year_pipeline.sh
```
Processes Q1–Q4 sequentially (one quarter at a time), then merges into `AIS_2024_full.csv`.
Add `--delete_raw` to free disk after each batch (already set in the shell script).

### Prepare a single date range
```bash
python3 scripts/data/run_pipeline.py \
    --start 2024-01-01 --end 2024-03-31 \
    --proc_dir data/processed/batches/q1 \
    --out data/processed/AIS_2024_Q1.csv \
    --batch_start_idx 0 --delete_raw
```

### Filter defaults (prepare_data.py)
Vessel types 60–89 (passenger/cargo/tanker), US coastal bbox lon `[-125, -60]` lat `[10, 55]`,
min 50 pts per track, max jump 2 km, max gap 60 min, turn segmentation at 150°, GPS median filter,
border truncation, min avg speed 1 km/h, min total distance 5 km.

### Split by track length
```bash
python3 scripts/data/split_by_length.py \
    --csv data/processed/AIS_2024_full.csv \
    --short_out data/processed/AIS_2024_le80.csv \
    --long_out  data/processed/AIS_2024_gt80.csv \
    --threshold 81
```

## Training

```bash
# From config (recommended)
python3 scripts/train/train_disp.py --config configs/12mo_seq120.yaml

# CLI — full options
python3 scripts/train/train_disp.py \
    --csv data/processed/AIS_2024_gt80.csv \
    --epochs 40 --val_frac 0.15 \
    --seq_len 120 --stride 50 \
    --num_layers 3 --lambda_smooth 5.0 \
    --out_dir runs/12mo_seq120
```

CLI args override config file values. Saves `best.pt`, `metrics.csv`, and `baseline.json` to `--out_dir`.

### Key design decisions

- **Displacement target** `[dlon_norm, dlat_norm]`, not absolute position.
  Recovered at eval: `pred_pos = input_pos + disp_scaler.inverse(pred)`.
- **Three scalers**: position (lon/lat), log(dt), displacement — fitted on train set only.
- **Vessel type conditioning**: learned embedding (28 AIS codes → 8-dim) added to all timesteps.
- **Causal mask**: position t cannot attend to future positions — no leakage.
- **Temporal gap mask**: positions with dt > `--max_gap_sec` (600 s) masked across gaps.
- **Loss**: `MSE + lambda_smooth × acceleration²`. t=0 excluded from causal loss.
- **Train/val split by root MMSI** (vessel identity, not segment) — prevents leakage.
- **Constant-velocity baseline** computed before training and saved to `baseline.json`.

### Input features (per timestep)

| Feature | Description | Normalised by |
|---------|-------------|---------------|
| `lon_norm`, `lat_norm` | absolute position | `pos` scaler |
| `log_dt_norm` | log(1 + seconds since last ping) | `logdt` scaler |
| `dlon_norm`, `dlat_norm` | displacement from previous step | `disp` scaler |

Plus vessel type embedding (8-dim, broadcast over all timesteps).

## Evaluation

```bash
# Model ADE/FDE vs CV baseline on exact val split
python3 scripts/eval/eval_checkpoint.py \
    --csv data/processed/AIS_2024_gt80.csv \
    --checkpoint runs/12mo_seq120/best.pt

# Plot training curves (can run mid-training)
python3 scripts/eval/plot_curves.py --csv runs/12mo_seq120/metrics.csv

# Visualize tracks and predictions
python3 scripts/eval/visualize.py \
    --csv data/processed/AIS_2024_Q1.csv \
    --checkpoint runs/12mo_seq120/best.pt \
    --out_dir outputs/viz

# Dataset diagnostics
python3 scripts/eval/diagnose_data.py --csv data/processed/AIS_2024_gt80.csv
```
