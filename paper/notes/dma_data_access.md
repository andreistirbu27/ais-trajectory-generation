# Danish AIS Data (DMA) — Access Plan

> **2026-05-21 update — major scope reduction.**
> Sanity-check run revealed the TrAISformer "sample" pickle is **not a sample** — it's the complete Kattegat dataset they trained on:
> | split | tracks |
> |---|---|
> | `ct_dma_train.pkl` | 10,605 |
> | `ct_dma_valid.pkl` |  1,481 |
> | `ct_dma_test.pkl`  |  1,593 |
> | **total**          | **13,679** |
>
> That's enough to **reproduce TrAISformer's published Kattegat numbers** and to **retrain our paper-track methods on Danish data** with no download required. The cross-region experiment is therefore unblocked Week 1 — no waiting on DMA portal access, no schema-drift debugging, no 30-80 GB of disk.
>
> Sections "Concrete steps" below are kept as a fallback if reviewers ask for a bigger Danish corpus, but the **default plan now uses only the shipped 13,679-track Kattegat dataset.**

## Source

- URL: https://dma.dk/safety-at-sea/navigational-information/ais-data
- Provider: Danish Maritime Authority
- Same source TrAISformer used.

## What we know from TrAISformer's repo

- TrAISformer used a Kattegat sub-region: lat 55.5–58.0, lon 10.3–13.0.
- 10-minute resampling.
- 18 history + 24 future = 7 hours total per training sample (and they filter by `min_seqlen=36` ⇒ 6+ hours per source track).
- The pickles shipped under `paper/external/TrAISformer/data/ct_dma/` are the **full Kattegat dataset** (10,605 / 1,481 / 1,593 = 13,679 tracks), schema `(N, 6)` float32 = `[lat_norm, lon_norm, sog_norm, cog_norm, unix_ts, mmsi]` (see `traisformer_data_format.md`). Their `datasets.py` only reads `[:, :4]` for features and `[0, 4]` for the per-track start time; column 5 is a redundant per-row MMSI repeat. **No additional download is needed to reproduce TrAISformer or to run the cross-region experiment.**

## What we need

For our cross-region experiment, we want **comparable scale to our US clean corpus**: ~60k cleaned trajectories, post-filtering. Estimate: download enough source data such that after `prepare_trajgen.py`-equivalent filtering we have 30k+ Danish tracks.

Rough estimate: DMA publishes one ZIP per day per "area"; each daily ZIP is 50–500 MB. A full year of Denmark coverage is ~30–80 GB raw.

## Concrete steps (FALLBACK ONLY — not on the critical path)

The original plan to download a year of DMA data is no longer required for the headline cross-region experiment. The 13,679-track shipped Kattegat dataset is enough to train and evaluate on Danish waters at parity with TrAISformer. The download plan below is retained only as a fallback if reviewers ask for either (a) a larger Danish corpus, (b) a year that doesn't overlap TrAISformer's published era, or (c) coverage outside Kattegat.

## Concrete steps (week 4-5)

### 1. Access verification (day 1)
- Visit the URL, confirm the data dump is still public and freely downloadable.
- Document URL structure (per-day URLs? FTP? S3-like bucket?).
- Test-download one day's file and confirm format matches what `csv2pkl.py` expects.

### 2. Download (in the background, days 1-5)
- Pick 12 months: 2023-01 through 2023-12 (recent but not the current year, to match TrAISformer's published era).
- Use `wget`/`aria2c` in the background, retry on failures.
- Storage: `data/raw/dma_2023/` — bbox-trimmed before save to keep size bounded.

### 3. Preprocessing (week 5)
- Adapt our `scripts/data/prepare_trajgen.py` to read DMA CSV format. DMA columns differ from NOAA — read their schema doc first.
- Same filter cascade as US: bbox, vessel type 60–89 if available, min/max distance, etc.
- Output: `data/processed/trajgen_128_dma.npz` (paper-track artifact).
- Land filter: requires Danish coastline. GSHHS L1 already covers Europe — just rebuild the SDF for the Danish bbox.

### 4. Build paper-track artifacts (week 5)
- `data/processed/land_sdf_050deg_dma.npz` — SDF raster for the Danish ROI.
- `data/processed/water_graph_005deg_dma.npz` — water graph for WaterRouter.
- `data/processed/trajgen_128_dma_clean.npz` — E1-filtered.
- `data/processed/trajgen_128_dma_clean_knn_k5.npz` — KNN cache.

All under `data/processed/` (gitignored) but listed in `paper/notes/dma_data_access.md` for paper reproducibility.

## Risks

1. **Schema drift.** DMA CSVs may have different column names / encodings (Danish vessel-type codes, UTF-8 BOM, etc.). Expect 1-2 days of debugging.
2. **Coverage gaps.** Some days may be missing from the public archive. Document and use only complete months.
3. **Vessel type encoding.** DMA may use ITU-R M.1371 codes directly; our pipeline assumes 28-code AIS taxonomy. Should be compatible but verify.
4. **Storage.** 30-80 GB of raw plus processed artifacts. Confirm `/data/raw/dma_2023/` has enough free space. On the cluster, this is fine.
5. **Legal.** DMA license likely permits research use but **confirm before publication.** Add license confirmation to `paper/notes/dma_data_access.md` once verified.

## Decision points

- **If access is restricted or schema-incompatible:** fall back to using only TrAISformer's shipped sample as a sanity test, and label our cross-region experiment as "limited DMA sample" rather than "trained on DMA". Demote that experiment from headline to appendix.
- **If download is feasible but slow:** start the download Week 1 in the background using `aria2c`. Don't block the rest of the plan.

## Citation / acknowledgement plan

```
Acknowledgements:
We thank the Danish Maritime Authority for making AIS data publicly available
(https://dma.dk/safety-at-sea/navigational-information/ais-data).
```

Add to the paper's Acknowledgements section once submitted.
