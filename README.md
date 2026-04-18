# PRL: AIS Trajectory Generation

Transformer-based maritime trajectory modeling from AIS data. Two approaches:

1. **Next-step prediction (v1)**: Causal transformer predicts next-position displacement from past trajectory. Matches but cannot beat constant-velocity baseline on post-turn-segmented tracks.

2. **Conditional trajectory generation (v2)**: Encoder-decoder transformer generates full 128-point routes given (start, end, vessel_type). Beats the great-circle baseline by 17% on ADE — learns real shipping lanes and navigation patterns.

## Structure
- `scripts/data/`   data pipeline: fetch, prepare, batch pipeline, split utilities, `prepare_trajgen.py` (NPZ dataset builder)
- `scripts/train/`  training: `train_disp.py` (v1 displacement), `train_vel.py` (v1 velocity), `train_gen.py` (v2 trajectory generation)
- `scripts/eval/`   evaluation: metrics, visualizations, diagnostics, `eval_gen.py` + `visualize_gen.py` (v2)
- `src/`            reusable modules: model, data, metrics (v1: `model.py`/`data.py`/`metrics.py`, v2: `*_gen.py`)
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

### v1 — Next-step prediction

```bash
python3 scripts/train/train_disp.py --config configs/12mo_seq120.yaml
```

CLI args override config file values. Saves `best.pt`, `metrics.csv`, and `baseline.json` to `--out_dir`.

**Key design decisions:**
- **Displacement target** `[dlon_norm, dlat_norm]`, not absolute position.
- **Three scalers**: position (lon/lat), log(dt), displacement — fitted on train set only.
- **Vessel type conditioning**: learned embedding (28 AIS codes → 8-dim) added to all timesteps.
- **Causal mask**: position t cannot attend to future positions.
- **Loss**: `MSE + lambda_smooth × acceleration²`. t=0 excluded.
- **Train/val split by root MMSI** (vessel identity, not segment) — prevents leakage.
- **Constant-velocity baseline** computed before training and saved to `baseline.json`.

**Input features per timestep:**

| Feature | Description | Normalised by |
|---------|-------------|---------------|
| `lon_norm`, `lat_norm` | absolute position | `pos` scaler |
| `log_dt_norm` | log(1 + seconds since last ping) | `logdt` scaler |
| `dlon_norm`, `dlat_norm` | displacement from previous step | `disp` scaler |

Plus vessel type embedding (8-dim, broadcast over all timesteps).

### v2 — Conditional trajectory generation

```bash
# Build dataset (resample tracks to 128 points, no turn segmentation)
python3 scripts/data/prepare_trajgen.py \
    --csv data/processed/AIS_2024_full.csv \
    --output data/processed/trajgen_128.npz

# Train
python3 scripts/train/train_gen.py --config configs/trajgen_mvp.yaml   # MVP
python3 scripts/train/train_gen.py --config configs/trajgen_v2.yaml    # stronger endpoint loss
python3 scripts/train/train_gen.py --config configs/trajgen_v3.yaml    # + scheduled sampling
```

**Architecture:** Encoder-decoder transformer. Encoder produces 3 conditioning tokens (start position, end position, vessel type). Decoder autoregressively generates 128 waypoints using cross-attention to the conditioning tokens.

**Key design decisions:**
- **No turn segmentation** — keeps full voyages with curves (the structure we want to learn).
- **Fixed 128-point resampling** — equal arc-length spacing via linear interpolation.
- **Displacement prediction** — decoder predicts normalized `[dlon, dlat]` deltas, accumulated to form trajectory.
- **Progress fraction** — decoder input includes `t/(T-1)` so the model knows how far along it is.
- **Teacher forcing** during training, autoregressive at inference.
- **Scheduled sampling** (v3) — with ramping probability, feeds model's own predictions instead of GT during training to reduce exposure bias.
- **Endpoint correction** — post-generation linear correction distributes endpoint error across all points.

**Loss:** `L_delta (Huber) + λ_end × L_endpoint (MSE) + λ_smooth × L_smooth (accel²)`

| Config | λ_end | λ_smooth | Epochs | Scheduled sampling |
|--------|-------|----------|--------|--------------------|
| MVP    | 10    | 1.0      | 50     | No                 |
| v2     | 50    | 0.5      | 100    | No                 |
| v3     | 50    | 0.5      | 100    | Yes (warmup 20, max 0.5) |

**Dataset:** 207K tracks from 2024 AIS data. 45% passenger, 24% cargo, 14% tanker. Median start-end distance 43 km.

**Results (MVP, 500 val trajectories):**

| Metric | Model | Great-Circle | |
|--------|-------|-------------|---|
| ADE | 6.8 km | 8.2 km | **17% better** |
| Endpoint error | 2.5 km → ~0 (after correction) | ~0 | |
| Path length ratio | 0.93 | 0.90 | |

## Evaluation

```bash
# --- v1 ---
python3 scripts/eval/eval_checkpoint.py \
    --csv data/processed/AIS_2024_gt80.csv \
    --checkpoint runs/12mo_seq120/best.pt

python3 scripts/eval/plot_curves.py --csv runs/12mo_seq120/metrics.csv

python3 scripts/eval/visualize.py \
    --csv data/processed/AIS_2024_Q1.csv \
    --checkpoint runs/12mo_seq120/best.pt \
    --out_dir outputs/viz

# --- v2 ---
python3 scripts/eval/eval_gen.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_mvp/best.pt \
    --no_frechet --n_eval 500

python3 scripts/eval/plot_curves_gen.py --csv runs/trajgen_mvp/metrics.csv

python3 scripts/eval/visualize_gen.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_mvp/best.pt \
    --out_dir outputs/viz_gen

# Dataset quality diagnostic plots
python3 scripts/eval/quick_data_viz.py \
    --data_npz data/processed/trajgen_128.npz \
    --out_dir outputs/data_viz
```
