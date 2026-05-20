# TrAISformer — Structured Summary

Source: Nguyen & Fablet, *TrAISformer: A Generative Transformer for AIS Trajectory Prediction*, IEEE Access 2024 (arXiv 2109.03958). Code: `paper/external/TrAISformer/`. License: CECILL-C.

This document captures what TrAISformer actually does (from reading their code, not just the abstract), and the implications for our comparison protocol.

---

## 1. Task

**Multi-step trajectory forecasting from history.** Given the first `init_seqlen=18` steps of an AIS trajectory (~3 hours at 10-minute spacing), roll out the next several hours. Their evaluation in `trAISformer.py:130` uses `max_seqlen = init_seqlen + 6*4 = 42`, i.e. they predict only **4 hours ahead** in the published code. The paper title and figure mention "up to ~10 hours", but the code as committed evaluates to 4 hours.

Inputs and outputs are both `[lat, lon, SOG, COG]` 4-tuples. SOG and COG are kept throughout — they are not auxiliary signals, they are part of the predicted token.

## 2. Architecture

GPT-style decoder-only Transformer, adapted from karpathy/minGPT.

- 8 layers, 8 heads.
- 4 independent embeddings: `n_lat_embd=256`, `n_lon_embd=256`, `n_sog_embd=128`, `n_cog_embd=128`. Concatenated to `n_embd = 896`.
- Inputs are *discretized* to cell IDs by `to_indexes()` (`models.py:245`): `(x * att_sizes).long()`, where `att_sizes = [lat_size, lon_size, sog_size, cog_size] = [250, 270, 30, 72]`.
- Output head is a **classification head** over `full_size = lat_size + lon_size + sog_size + cog_size = 622` classes — note this is **four independent classifications**, not a joint distribution.
- Learnable absolute positional embedding `pos_emb` of size `(1, max_seqlen, n_embd)`.
- Causal mask via `register_buffer("mask", torch.tril(...))`.

## 3. Loss

`models.py:323-380`:

```
L = lat_loss + lon_loss + sog_loss + cog_loss
```

Each term is per-token cross-entropy on the four independent classification heads. So implicitly the model assumes lat, lon, SOG, COG are **conditionally independent given the history embedding** — a strong factorization assumption.

**"Blur" loss** (`blur=True`, `blur_n=2`, `blur_loss_w=1.0`): an additional smoothness term. After softmax, they apply a fixed 3-tap conv (`Conv1d(1,1,3, weights=1/3 each)`) to the probability distribution, then compute NLL against the same target. The conv is repeated `blur_n=2` times. The effect is to encourage probability mass to spread to neighbouring cells — softening the target.

**This is the "novel multimodality loss" referenced in the paper.** It's a smoothing trick, not a structured prediction loss. Easy to ablate against; easy to dismiss as a regularizer.

## 4. Discretization scheme

For `dataset_name=ct_dma`:

| Axis | Range | Bins | Cell size |
|---|---|---:|---|
| lat | 55.5° – 58.0° (2.5°) | 250 | 0.01° ≈ 1.1 km |
| lon | 10.3° – 13.0° (2.7°) | 270 | 0.01° ≈ 0.6 km @ lat 56° |
| SOG | 0 – 30 kn | 30 | 1 knot |
| COG | 0° – 360° | 72 | 5° |

Inputs to the model are pre-normalized to [0, 1) per axis: `lat_normalized = (lat - lat_min) / (lat_max - lat_min)`. Bin index = `floor(x_norm * att_sizes)`.

This is a **tiny ROI**: Kattegat / Skagerrak, ~280 × 190 km. By contrast our US bbox is ~7000 × 5000 km. A direct port to US would either need ~70k × 50k bins (impossible) or much coarser cells (loss of accuracy).

## 5. Sampling

`sample_mode=pos_vicinity`, `top_k=10`, `r_vicinity=40` (`trAISformer.py:147`):

- At each step, look up `r_vicinity=40` cells around the previous lat / lon prediction (a hand-coded spatial locality prior).
- Within that window, pick top-K most probable cells, sample one.
- SOG and COG sampled with top-K=10 from the full distribution.

`n_samples=16` (`config_trAISformer.py:32`): for every test trajectory, sample 16 candidate rollouts. **Evaluation reports min(haversine error) across the 16 samples** (`trAISformer.py:155`), and only then averages across trajectories.

**Critical for fair comparison:** TrAISformer's reported ADE is "**best-of-16** trajectories per query". v10 produces a single trajectory per query. Comparing v10's single-shot ADE to TrAISformer's best-of-16 is an apples-to-pears comparison that **disadvantages v10**. We must either (a) also report best-of-16 for v10 with appropriate dropout-time sampling, or (b) compare both at single-shot (mean over samples for TrAISformer).

## 6. Evaluation metric

- Haversine distance per step, masked by padding.
- Averaged across batch and time → "prediction error curve".
- They mark 1h, 2h, 3h points on the plot. The fig in `figures/t18_3.png` is the cited result.
- **Units:** code reports km, paper converts to nm (1 nm ≈ 1.852 km).

## 7. Dataset

Danish Maritime Authority (DMA): https://dma.dk/safety-at-sea/navigational-information/ais-data

- ROI = Kattegat (~lat 55.5–58.0, lon 10.3–13.0).
- Sample rate: 10 minutes (inferred from `init_seqlen=18` ≈ 3 hours and the `/ 6` in the eval plot).
- Filter: `min_seqlen=36` (6 hours minimum trajectory length).
- Moving-threshold cleanup: trim leading non-moving prefix where `SOG_normalized < 0.05` (i.e. SOG < 1.5 knots).
- Format: pickled list of dicts. Each dict: `{"mmsi": int, "traj": ndarray(N, 5)}` where columns are `[lat_norm, lon_norm, sog_norm, cog_norm, unix_timestamp]`, lat/lon/sog/cog pre-normalized to [0,1) within ROI.
- Preprocessing code lives at https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py — referenced but not vendored. See [traisformer_data_format.md](traisformer_data_format.md) for our extraction.

A sample is shipped in `paper/external/TrAISformer/data/ct_dma/` (`ct_dma_train.pkl`, `ct_dma_valid.pkl`, `ct_dma_test.pkl`).

## 8. Baselines (in their paper)

From abstract / paper: they compare against AIS-specific prior work — GeoTrackNet (their own VRNN), LSTM baselines. Probably no Transformer comparison (they ARE the Transformer baseline).

## 9. Land/geographic handling

**None.** No SDF, no coastline, no projection. The model can predict on-land cells freely. The Kattegat ROI is mostly water so this is rarely visible in the published figures.

## 10. Bugs / oddities noted in code

- `trAISformer.py:129`: `v_roi_min = torch.tensor([model.lat_min, -7, 0, 0])` — uses `model.lat_min` for lat but hardcodes `-7` for lon. The Config has `lon_min=10.3`. The `-7` looks like leftover from a different ROI (maybe the original European-Atlantic ROI from their GeoTrackNet work). For ct_dma evaluation this would mean lon is wrongly de-normalized at eval time, producing systematic offset. **Worth confirming in their published numbers — if real, suggests their reported ADE may have a known unit/offset confound.**
- `models.py:269`: `mode=freq` path uses an external `partition_model` that's never set in the public config. Effectively dead code in the released repo.
- `dataset.py:66`: `m_v[m_v > 0.9999] = 0.9999` modifies the loaded array in-place each time `__getitem__` is called. Harmless because idempotent, but ugly.

## 11. Implications for our comparison

### Native task difference
TrAISformer takes 18 history points with (lat, lon, **SOG, COG**) and predicts 24 future points. v10 takes 1 start point, 1 end point, vessel type, and predicts 128 waypoints with no time information. **These solve different problems.**

### To make a fair benchmark, we need:
1. **Re-train TrAISformer on a US sub-ROI** matching their physical scale (~2.5° × 2.7° box). Candidate: Gulf approaches (lat 28–30.5, lon -90 to -87.3). Bin counts can be kept the same.
2. **Adapt our data pipeline** to also produce time-uniform 10-minute samples (we currently arc-length-resample, losing SOG/COG and time info). See `scripts/paper/csv_to_traisformer_pkl.py` (to be written).
3. **Define a shared comparison protocol.** Three candidates from the plan §3:
   - Protocol A — TrAISformer's native task, both models retrained.
   - Protocol B — v10's native task, TrAISformer adapted (give it 1-point "history").
   - Protocol C — K-anchor protocol. Strongest, but most work.

### Direct numerical comparison caveats
- TrAISformer's "best-of-16" inflates their headline. Document this; compute both mean and best-of-N for both models.
- Cross-ROI numbers don't transfer: a model trained on Kattegat can't predict Gulf traffic. Don't try.
- Their 4-hour eval horizon is a *short* horizon by our standards. v10's 128 waypoints over a 50-km median route corresponds to whatever wall-clock duration that takes — typically 2–8 hours. So the time scales partially overlap.

### Reproducibility plan (Week 2-3 in main timeline)
1. Verify environment: run their training on the shipped DMA sample, confirm we can match published prediction-error curve within tolerance. ~2 days.
2. Train on full DMA dump (once we acquire it — see [dma_data_access.md](dma_data_access.md)). ~3 days @ 1 GPU.
3. Write `scripts/paper/csv_to_traisformer_pkl.py` to convert our US AIS to their format. ~2 days.
4. Train TrAISformer on US Gulf sub-ROI. ~3 days @ 1 GPU.
5. Define shared eval harness `scripts/paper/eval_shared_protocol.py`. ~3 days.

## 12. Citation format (BibTeX skeleton)

```bibtex
@article{traisformer2024,
  title   = {{TrAISformer}: A Generative Transformer for {AIS} Trajectory Prediction},
  author  = {Nguyen, Duong and Fablet, Ronan},
  journal = {IEEE Access},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3349957},
  note    = {arXiv:2109.03958},
}
```
