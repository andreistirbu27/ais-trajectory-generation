# TrAISformer Data Format

## Per-sample structure

Each pickled file (`ct_dma_train.pkl`, `_valid.pkl`, `_test.pkl`) is a Python list of trajectory dicts:

```python
[
    {
        "mmsi": <int>,
        "traj": np.ndarray of shape (N, 5),  # columns: lat, lon, sog, cog, unix_timestamp
    },
    ...
]
```

Pickled with `pickle.dump` (no compression).

## Column semantics (inferred from `datasets.py` and `trAISformer.py:151`)

| Index | Field | Storage | Notes |
|---|---|---|---|
| 0 | lat_normalized | float in [0, 1) | `(lat_deg - lat_min) / (lat_max - lat_min)` |
| 1 | lon_normalized | float in [0, 1) | `(lon_deg - lon_min) / (lon_max - lon_min)` |
| 2 | sog_normalized | float in [0, 1) | `sog_kn / 30.0` |
| 3 | cog_normalized | float in [0, 1) | `cog_deg / 360.0` |
| 4 | unix_timestamp | float seconds | Used only at eval time to scale times |

`datasets.py:66` clips values: `m_v[m_v > 0.9999] = 0.9999`. So strictly speaking values must lie in [0, 0.9999].

## Sampling cadence

- Constant 10-minute spacing between consecutive rows of `traj` (inferred from eval-time `np.arange(...) / 6` ⇒ 6 samples per hour).
- This is enforced upstream during csv→pkl conversion. AIS reports natively at variable cadence (every 6 s for ships in motion, every 3 min for moored ships, etc.) — their pipeline resamples to a uniform 10-min grid.

## Filtering applied by TrAISformer at load time (`trAISformer.py:82-87`)

```python
# Drop leading non-moving prefix
moving_threshold = 0.05    # normalized SOG, i.e. real SOG > 1.5 knots
moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
V["traj"] = V["traj"][moving_idx:, :]

# Drop NaN-containing trajectories and ones shorter than min_seqlen=36
Data[phase] = [x for x in pkl
               if not np.isnan(x["traj"]).any()
               and len(x["traj"]) > cf.min_seqlen]
```

So even after the pickle is built, the model further restricts to:
- Tracks ≥ 36 steps (= 6 hours @ 10 min)
- Tracks with no NaN values
- Tracks starting at the first SOG > 1.5 kn point

## Upstream csv2pkl.py (GeoTrackNet)

We need to read https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py to understand the full conversion. **TODO:** clone or fetch this; current notes file should be expanded with the actual conversion logic in week 2.

Conceptually, csv2pkl.py likely does:
1. Read raw AIS CSV (MMSI, timestamp, lat, lon, SOG, COG).
2. Group by MMSI.
3. Sort by timestamp.
4. Split into trajectories at large temporal gaps.
5. Resample each trajectory to fixed 10-minute grid (linear interpolation).
6. Filter by ROI bbox; clip lat/lon to ROI.
7. Normalize lat/lon/SOG/COG to [0,1).
8. Drop tracks shorter than threshold.
9. Pickle.

## Our conversion script — design notes for `scripts/paper/csv_to_traisformer_pkl.py`

### Inputs
- Source CSV: `data/processed/AIS_2024_full.csv` (US bbox, after `scripts/data/prepare_data.py` filtering).
- Target ROI: a sub-bbox of US to keep cell-count comparable to DMA. Candidate Gulf box: lat 28.0–30.5, lon -90.0 to -87.3 (2.5° × 2.7°, ~280 × 240 km @ lat 29).

### Pipeline
1. Filter CSV to target sub-bbox.
2. Group by MMSI (handle batch-prefixed IDs from `b{NNN}_` notation — extract real MMSI).
3. Sort by timestamp.
4. Split into trajectories at temporal gaps > 60 min (same as existing prepare_data.py).
5. **Resample to uniform 10-min grid** via linear interpolation on lat/lon, and use forward-fill or linear on SOG/COG (preserving the rotational nature of COG via sin/cos interpolation).
6. Clip to ROI bbox.
7. Compute normalized values: `lat_n = (lat - 28.0) / 2.5`, etc.
8. Filter by min length `min_seqlen=36`.
9. Train/val/test split (use MMSI-grouped split same as v10's `_root_id` policy).
10. Pickle to `data/processed/traisformer_us_gulf_{train,valid,test}.pkl`.

### Risks
- AIS sampling is irregular; we may lose 30–50% of tracks to resampling artifacts.
- COG interpolation must be circular-aware (sin/cos before interp, atan2 after).
- TrAISformer's `min_seqlen=36` means 6 hours of continuous data — tight constraint on US AIS, which has heavier vessel turnover.

### Verification (sanity checks before training)
- Visualize 20 random tracks pre/post resample; eyeball quality.
- Distribution of trajectory lengths should peak around 50–200 steps.
- `lat_normalized.min()` ≥ 0, `.max()` < 1. Same for lon, SOG, COG.
- No NaN, no Inf.
- Train/val/test MMSI overlap = 0.

## Verification of TrAISformer's bbox arithmetic

At eval (`trAISformer.py:128-129`):
```python
v_ranges = torch.tensor([2, 3, 0, 0])         # lat_range, lon_range, 0, 0
v_roi_min = torch.tensor([model.lat_min, -7, 0, 0])  # lat_min, lon_min, 0, 0
input_coords = (inputs * v_ranges + v_roi_min) * pi/180
```

So de-normalization uses `[2, 3]` for ranges — but the actual ranges are `lat_max - lat_min = 2.5` and `lon_max - lon_min = 2.7`. So they use **truncated integer ranges** at eval time, not the actual ranges. And lon_min is `-7` instead of `10.3`.

**This is suspicious.** Possible explanations:
1. They evaluate in a different ROI than they train on.
2. The eval code in the public repo was written for a different config and not updated.
3. There's an off-by-one or hardcoded-debug artefact.

If (1) or (2): their published numbers may be reproducible only if we replicate the same eval-time mismatch. If (3): their published numbers may be biased by a known factor. Document carefully in our reproduction; flag to readers.

## Decisions for our work

- We'll use **the actual `lat_max - lat_min`** as the range during de-normalization in our reproduction. If we can't match their numbers, we'll try the `[2, 3, ...]` truncation as a fallback to see if that's how they got there.
- Our US Gulf box will be computed correctly. We don't replicate their potential bug.
