# PRL: AIS Trajectory Generation

Transformer-based maritime trajectory modeling from AIS data. Six approaches:

1. **Next-step prediction (v1)**: Causal transformer predicts next-position displacement from past trajectory. Matches but cannot beat constant-velocity baseline on post-turn-segmented tracks.

2. **Conditional trajectory generation (v2)**: Encoder-decoder transformer generates full 128-point routes given (start, end, vessel_type). Beats the great-circle baseline by 17% on ADE — learns real shipping lanes and navigation patterns. Evolved through MVP → v2 → v3 → v4 (bigger model + bearing features + encoder) → v5 (v4 + scheduled sampling).

3. **Obstacle-conditioned generation (v6/v7)**: v5 architecture augmented with obstacle encoder tokens and a differentiable penetration penalty in the loss. Target application: storm / exclusion-zone avoidance. v6 had a data leak (per-sample fixed obstacles → penalty never fired); v7 fixes the leak but revealed a deeper design problem (obstacles are placed laterally so GT never goes around them, so `L_obstacle` is structurally zero). Parked until a detour-GT pipeline exists.

4. **Retrieval-augmented generation (v9)**: K nearest historical routes added as encoder tokens via `MeanPoolRouteEncoder`. Best overall ADE at epoch 38 (6207 m; beats v5 by 8%, retrieval-top-1 by 4%, GC by 25%), but mean-pool collapses route shape into a single vector per neighbor — retrieval-top-1 beats v9 by 2.5× on long routes (>500 km).

5. **Per-timestep retrieval + land-aware loss (v10)**: fixes v9's mean-pool bottleneck by giving the decoder `K * t_retr` per-step retrieval tokens (163 vs 8 memory tokens). Adds a differentiable SDF-based land penalty during training and a hard land projection at inference. Trained, 50 epochs. Closed 74% of the long-bucket gap to retrieval-top-1 (25964 m vs retrieval 18785 m, v9 46631 m), but overall ADE regressed to 6584 m (short-bucket 2117 → 4160 m) and land crossings stayed at 36.2% — the soft L_land cannot prevent segment crossings, only push waypoints off land. Hard projection at inference eliminates crossings but degrades ADE by 42% and breaks FDE=0 (it teleports endpoints off coastal ports). Motivated v11.

6. **Pointer transformer with water-valid candidate pool (v11)**: replaces v10's regression head with a pointer (masked softmax) over K=32 candidates generated per step via a Halton fan in the forward cone. Every candidate is water-point and water-segment filtered before the model sees it, so the trajectory is water-by-construction (0.00% crossings by the constraint, no soft penalty, no hard projection). Auxiliary offset head refines sub-cell position. Full causal mask (v10's `k_past=32` regressed short routes). Implemented and locally smoke-tested; training pending on school machines.

## Structure
- `scripts/data/`   data pipeline: fetch, prepare, batch pipeline, split utilities, `prepare_trajgen.py` (NPZ dataset builder), `build_land_sdf.py` (v10 land SDF raster)
- `scripts/train/`  training: `train_disp.py` (v1 displacement), `train_vel.py` (v1 velocity), `train_gen.py` (v2 trajectory generation, supports `--resume`), `train_gen_obs.py` (v6/v7 obstacle-conditioned), `train_gen_retrieval.py` (v9), `train_gen_v10.py` (v10), `train_gen_v11.py` (v11)
- `scripts/eval/`   evaluation: metrics, visualizations, diagnostics, `eval_gen.py`/`visualize_gen.py` (v2), `eval_obstacle_gen.py`/`visualize_obstacle_gen.py` (v6), `retrieval_baseline.py`/`eval_gen_retrieval.py` (v9), `eval_gen_v10.py`/`visualize_gen_v10.py` + `land_crossing_diagnostic.py` (v10), `eval_gen_v11.py`/`visualize_gen_v11.py` (v11)
- `src/`            reusable modules: `model.py`/`data.py`/`metrics.py` (v1), `*_gen.py` (v2), `*_gen_obs.py` (v6), `*_gen_retrieval.py` (v9), `model_gen_v10.py` + `land_mask.py` (v10), `model_gen_v11.py` + `data_gen_v11.py` + `candidates.py` (v11)
- `configs/`        YAML experiment configs grouped by approach (`displacement/`, `trajgen/`, `obstacle/`, `retrieval/`)
- `notebooks/`      exploratory notebooks
- `docs/references/` papers and links
- `data/`           datasets (not tracked). `data/processed/land_sdf_050deg.npz` and `land_sdf_005deg.npz` are v10/v11 SDF rasters.
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
python3 scripts/train/train_gen.py --config configs/trajgen_v4.yaml    # bigger + bearing + encoder
python3 scripts/train/train_gen.py --config configs/trajgen_v5.yaml    # v4 + scheduled sampling

# Resume any v2 run from its last checkpoint
python3 scripts/train/train_gen.py --config configs/trajgen_v5.yaml \
    --resume runs/trajgen_v5/best.pt
```

**Architecture:** Encoder-decoder transformer. Encoder produces 3 conditioning tokens (start position, end position, vessel type). Decoder autoregressively generates 128 waypoints using cross-attention to the conditioning tokens.

**Key design decisions:**
- **No turn segmentation** — keeps full voyages with curves (the structure we want to learn).
- **Fixed 128-point resampling** — equal arc-length spacing via linear interpolation.
- **Displacement prediction** — decoder predicts normalized `[dlon, dlat]` deltas, accumulated to form trajectory.
- **Progress fraction** — decoder input includes `t/(T-1)` so the model knows how far along it is.
- **Teacher forcing** during training, autoregressive at inference.
- **Scheduled sampling** (v3, v5) — with ramping probability, feeds the model's own predictions instead of GT during training to reduce exposure bias. Implemented as a **two-pass parallel TF** (Pass 1 no-grad over GT → Pass 2 over a mixed trajectory) so SS epochs cost ≈2× a TF epoch, not ~30×.
- **Endpoint correction** — post-generation linear correction distributes endpoint error across all points.

**Loss:** `L_delta (Huber) + λ_end × L_endpoint (MSE) + λ_smooth × L_smooth (accel²)`

| Config | d_model | Enc layers | Decoder input | λ_end | λ_smooth | Epochs | Scheduled sampling |
|--------|---------|------------|---------------|-------|----------|--------|--------------------|
| MVP    | 128     | 0          | 3 (no bearing)| 10    | 1.0      | 50     | No                 |
| v2     | 128     | 0          | 3             | 50    | 0.5      | 100    | No                 |
| v3     | 128     | 0          | 3             | 50    | 0.5      | 100    | Yes (warmup 20, max 0.5) |
| v4     | 256     | 2          | 5 (+ sin/cos bearing to end) | 10 | 1.0 | 60 | No |
| v5     | 256     | 2          | 5             | 10    | 1.0      | 60     | Yes (warmup 20, max 0.5) |

**Dataset:** 207K tracks from 2024 AIS data. 45% passenger, 24% cargo, 14% tanker. Median start-end distance 43 km.

**Results (MVP, 500 val trajectories):**

| Metric | Model | Great-Circle | |
|--------|-------|-------------|---|
| ADE | 6.8 km | 8.2 km | **17% better** |
| Endpoint error | 2.5 km → ~0 (after correction) | ~0 | |
| Path length ratio | 0.93 | 0.90 | |

### v6 — Obstacle-conditioned generation

```bash
# Train (v5 architecture + obstacle tokens + penetration loss)
python3 scripts/train/train_gen_obs.py --config configs/trajgen_v6_obstacle.yaml

# Evaluate: clean ADE + obstacle avoidance rate / clearance / path overhead
python3 scripts/eval/eval_obstacle_gen.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_v6_obstacle/best.pt

# Visualize: grid of GT, clean baseline, avoidance trajectory + obstacle circle
python3 scripts/eval/visualize_obstacle_gen.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_v6_obstacle/best.pt \
    --out_dir outputs/viz_obstacle
```

**Architecture changes (`src/model_gen_obs.py`):**
- Encoder memory extended from 3 to `3 + K` tokens (K=3 max). New token-type ID (4 total) distinguishes obstacles.
- `Linear(3 → d_model)` projects each obstacle `(center_lon, center_lat, radius)`.
- Padding mask `(B, 3+K)` flows through both the encoder (`src_key_padding_mask`) and decoder (`memory_key_padding_mask`), so padded obstacle slots never affect attention.

**Training data (`src/data_gen_obs.py`):**
- `ObstacleAugmenter` places 1..3 circular exclusion zones per trajectory, with probability `p_obstacle=0.5`. Zones are laterally offset from the GT route by at least `radius + margin`, so the GT is an "avoidance" example by construction.

**Loss** (key mechanism that forces avoidance, not just correlation):

`L = L_delta + λ_end · L_endpoint + λ_smooth · L_smooth + λ_obstacle · L_obstacle`

`L_obstacle = mean(max(0, r + margin − ‖pred_waypoint − obstacle_center‖)²)` — quadratic soft repulsion in degree space. Differentiable through the cumulative delta sum; gradient flows back to the decoder.

| Config | λ_obstacle | margin | p_obstacle | max_obstacles | radius (km) |
|--------|------------|--------|------------|---------------|-------------|
| v6     | 50         | 5 km   | 0.5        | 3             | 10–100      |

**Evaluation protocol**: for each val trajectory, (1) generate clean baseline, (2) place a 50 km blocking obstacle on the baseline's midpoint, (3) regenerate with that obstacle, (4) measure avoidance rate, min distance to center, clearance, path length overhead vs the clean baseline, and ADE vs GT (to check for regression vs v4/v5).

### v9 — Retrieval-augmented generation

```bash
# Zero-training retrieval-only baseline (the v9 gate)
python3 scripts/eval/retrieval_baseline.py \
    --data_npz data/processed/trajgen_128.npz \
    --n_eval 500

# Train v9 (KNN index built and cached on first run)
python3 scripts/train/train_gen_retrieval.py \
    --config configs/retrieval/trajgen_v9_retrieval.yaml

# Evaluate v9 (length-bucketed ADE + normalized ADE + land-crossing stats)
python3 scripts/eval/eval_gen_retrieval.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_v9_retrieval/best.pt \
    --knn_cache data/processed/trajgen_128_knn_k5.npz \
    --n_eval 500 --no_frechet
```

**Architecture** (`src/model_gen_retrieval.py`): v5 encoder-decoder + K retrieved-route tokens. Each retrieved `(T=128, 2)` trajectory is compressed to one `d_model` vector via `MeanPoolRouteEncoder`. Encoder memory = 3 base tokens + K = 5 retrieval tokens = 8 tokens total.

**Retrieval** (`src/data_gen_retrieval.py`): 5-D KNN over `(start_lon, start_lat, end_lon, end_lat, vtype)` with `vtype_weight=0.5`. KNN index is built once and cached to `data/processed/trajgen_128_knn_k5.npz`.

**Results (ep 38, stopped mid-training, 500 val, seed=42):**

| Method | ADE | FDE | normalized ADE |
|---|---:|---:|---:|
| **v9 (ep 38)** | **6207 m** | **0 m** | **0.0635** |
| Retrieval-top-1 | 6470 m | 5603 m | 0.0819 |
| Great-circle | 8226 m | 0.1 m | 0.0704 |
| v5 (ep 60) | 6753 m | 0 m | — |

**Length-bucketed ADE:**

| Bucket | n | v9 | retr-top1 | GC |
|---|---:|---:|---:|---:|
| short (<50 km) | 243 | **2117 m** | 3366 m | 2264 m |
| medium (50–500 km) | 250 | **9050 m** | 9143 m | 11598 m |
| long (>500 km) | 7 | 46631 m | **18785 m** | 94774 m |

v9 wins short + medium but loses long routes by 2.5× to zero-training retrieval-top-1. The mean-pool route encoder is the bottleneck: it compresses the full `(T=128, 2)` shape into one centroid-like vector before the decoder sees it, and on long routes the shape IS the signal. This motivated v10.

### v10 — Per-timestep retrieval + land-aware loss

```bash
# Build land SDF raster (one-shot, cached)
python3 scripts/data/build_land_sdf.py \
    --shoreline data/gshhg/GSHHS_i_L1.shp \
    --output data/processed/land_sdf_050deg.npz

# Train v10 (reuses v9 KNN cache)
python3 scripts/train/train_gen_v10.py \
    --config configs/trajgen/trajgen_v10.yaml

# Evaluate v10 (raw + hard-projection + retrieval-top-1 + GC)
python3 scripts/eval/eval_gen_v10.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_v10/best.pt \
    --knn_cache data/processed/trajgen_128_knn_k5.npz \
    --land_sdf data/processed/land_sdf_050deg.npz \
    --n_eval 500 --no_frechet
```

Three changes vs v9:

1. **Per-timestep retrieval** (`src/model_gen_v10.py::PerStepRouteEncoder`): each retrieved route is subsampled to `t_retr=32` points, each projected to `d_model` with per-route-index + per-step embeddings. Encoder memory = 3 base + `K * t_retr = 160` retrieval tokens = **163 tokens** (vs v9's 8). Decoder cross-attends to "neighbor k at step t" directly — no mean-pool collapse.
2. **Windowed causal mask** (`k_past=32`): decoder cannot attend beyond 32 steps of its own past. Reduces drift from stale self-state over 128-step AR rollout.
3. **Land-aware loss + hard inference projection** (`src/land_mask.py`):
   - Training: `L_land = mean(ReLU(sdf_km - 10)²)` via differentiable `F.grid_sample` on a precomputed 0.05° SDF raster; `λ_land=0.01`, `threshold=10 km` (discounts coastal raster noise — GT natural floor is 10.83%).
   - Inference: inside `generate()`, project any waypoint with `sdf > 10 km` to the nearest water cell via BFS (`LandMask.project_to_water`). Projected point fed back into the decoder so subsequent steps condition on the corrected trajectory. Guarantees zero land crossings in the post-processed output.

Land-crossing baselines (from v9 eval):

| | threshold 10 km | threshold 25 km |
|---|---:|---:|
| Ground truth | 10.83% | 1.81% |
| v9 (ep 38) | 12.81% | 2.10% |
| Retrieval-top-1 | 11.03% | 1.85% |
| Great-circle | 13.47% | 2.77% |

v10 success criteria: long-bucket ADE at least halfway between v9 (46631 m) and retrieval-top-1 (18785 m); v10 + hard projection land-crossing rate ≈ GT (≤ 11% at 10 km threshold).

**v10 outcome (50 epochs):**

| Metric | v10 raw | v10 + hard proj | v9 (ep 38) | retr-top-1 |
|---|---:|---:|---:|---:|
| ADE overall | 6584 m | ~9340 m | 6207 m | 6470 m |
| ADE short (<50 km) | 4160 m | — | 2117 m | 3366 m |
| ADE long (>500 km) | 25964 m | — | 46631 m | 18785 m |
| traj_crossing_rate | 36.2% | 0.00% | 39.8% | 35.6% |
| FDE | 0 m | non-zero | 0 m | 5603 m |

Closed 74% of the v9 → retrieval-top-1 long-bucket gap (the PerStepRouteEncoder worked), but regressed overall ADE because the short-bucket jumped from 2117 → 4160 m. Root causes: (a) the windowed mask `k_past=32` anchors too locally for short straight routes; (b) the soft `L_land` can push individual waypoints off land but cannot prevent segment crossings — 36% of trajectories still cross land because two consecutive water-valid waypoints can span a peninsula; (c) hard projection at inference breaks `FDE=0` because it teleports snapped endpoints off coastal ports. v11 addresses all three.

### v11 — Pointer transformer with water-valid candidate pool

```bash
# Train v11-lite (reuses v9 KNN cache + fine-resolution land SDF)
python3 scripts/train/train_gen_v11.py \
    --config configs/trajgen/trajgen_v11.yaml

# Resume v11
python3 scripts/train/train_gen_v11.py \
    --config configs/trajgen/trajgen_v11.yaml \
    --resume runs/trajgen_v11_lite/best.pt

# Eval v11 (strict 0-km land-crossing check — expect 0.00% by construction)
python3 scripts/eval/eval_gen_v11.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_v11_lite/best.pt \
    --knn_cache data/processed/trajgen_128_knn_k5.npz \
    --land_sdf data/processed/land_sdf_005deg.npz \
    --n_eval 500 --no_frechet

# Visualize v11 (GT + v11 + retrieval + GC, optional candidate fans)
python3 scripts/eval/visualize_gen_v11.py \
    --data_npz data/processed/trajgen_128.npz \
    --checkpoint runs/trajgen_v11_lite/best.pt \
    --knn_cache data/processed/trajgen_128_knn_k5.npz \
    --land_sdf data/processed/land_sdf_005deg.npz \
    --out_dir outputs/viz_v11_lite
```

Three changes vs v10:

1. **Pointer head over candidate pool** (`src/model_gen_v11.py::TrajectoryGeneratorV11`): at each step, K=32 candidates are sampled in the forward cone (`±60°` toward the destination, radius 3–25 km) via a pre-computed Halton 2D quasi-random grid. Each candidate is filtered through (i) bbox, (ii) water-point SDF lookup, (iii) 5-interior-sample segment-water check. Only valid candidates get non-`-inf` logits. Logit = `dot(candidate_embed, h_t) / sqrt(d_model)`. An auxiliary `offset_head` refines the selected candidate in normalized delta space (re-verified water-valid; discarded if offset pushes off water).

2. **GT-insertion training signal** (`src/candidates.py::insert_gt_into_candidates`): the ground-truth next step is injected into the candidate pool at slot 0 when the GT segment passes the filter. Otherwise the closest valid candidate is used as the target (`gt_index = argmin ||gt - cand||` over valid slots); if no valid candidate exists, `gt_index = -100` (CE ignore_index). Guarantees the pointer sees a feasible target on clean open-sea samples.

3. **Water-by-construction, no L_land** (`src/candidates.py`): because the candidate pool is pre-filtered, any trajectory the model can produce consists entirely of water-valid waypoints connected by water-valid segments. `traj_crossing_rate = 0.00%` at `threshold_km = 0` is a hard guarantee, not a soft target. No inference projection, no post-hoc fixup. Loss simplifies from `L_delta + L_end + L_smooth + L_land` to `L_CE + λ_offset·L_offset + λ_smooth·L_smooth`.

Inference uses a fallback cascade for empty candidate pools (rare): widen cone to ±120° → shrink `d_min_km` to 1 → snap to nearest water along the bearing-to-end ray → flag infeasible. The final step snaps to `end` directly if the segment is water (preserving FDE=0 for most samples).

v11 success gates:

1. **Strict** `traj_crossing_rate = 0.00%` at threshold 0 km (by construction — any non-zero value is a bug in the candidate filter).
2. Overall ADE < 6584 m (beat v10).
3. Short-bucket ADE < 4160 m (beat v10 short; full causal mask was the hypothesis).

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
