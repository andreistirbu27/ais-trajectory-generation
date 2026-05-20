# Paper Track

This subtree contains everything for the **ICLR 2027 submission** (~Sep 2026 deadline). It is intentionally separate from `report/`, which holds the school report and is frozen.

## Branch

Paper work lives on the `paper` git branch. Never merge paper-track commits into `main`; the school report's `main` commits stay clean.

## Layout

```
paper/
  paper.tex                  Working draft (LaTeX, ICLR-style skeleton)
  bib/refs.bib               Bibliography (organized by plan §4 categories)
  figures/                   Paper figures (regenerated from results/paper/)
  tables/                    LaTeX tables (regenerated)
  notes/                     Planning notes, summaries, decision logs
  external/                  Third-party code as git submodules
    TrAISformer/             https://github.com/CIA-Oceanix/TrAISformer

configs/paper/               Configs ONLY for paper experiments
scripts/paper/               Scripts ONLY for paper experiments
results/paper/               CSVs from paper experiments
runs/paper/                  Training outputs from paper experiments (gitignored)
```

## Working principles

1. **Do not edit `report/`, `docs/decisions.md`, `docs/narrative.md`, or `docs/v13_proposal.md`** for paper work. Those belong to the school report. Cosmetic refinements (typos) to `report/` are allowed; new content is not.

2. **Do not edit `src/model_gen_v10.py`, `src/water_router.py`, `src/land_mask.py`, `src/model_gen_retrieval.py`, or `scripts/eval/eval_gen_v10_router.py`** in ways that change behaviour. The school report's headline numbers must remain reproducible from `runs/trajgen_v10/best.pt`. If you need to modify behaviour for the paper, fork into `src/model_gen_paper.py` etc.

3. **Place new modules** for paper-specific work under `src/` with clear names (`src/diffusion.py`, `src/model_gen_v13.py`, `src/traisformer_adapter.py`).

4. **Place new scripts** under `scripts/paper/` (`train_gen_v13.py`, `eval_shared_protocol.py`, `csv_to_traisformer_pkl.py`).

5. **Place new configs** under `configs/paper/` with subdirectories per experiment family (`v13/`, `traisformer_baseline/`, `v10_paper/`).

6. **Place experiment outputs** under `runs/paper/<run_name>/` and `results/paper/<table_name>.csv`. The existing `runs/` and `results/` belong to the school report.

## Cloning the repo from scratch

```bash
git clone <repo-url>
cd ais-trajectory-generation
git checkout paper
git submodule update --init --recursive
```

## Plan and decisions

The strategic plan is in `~/.claude/plans/i-need-strategic-research-elegant-shamir.md`. Locked scope decisions (2026-05-20):

- Venue: ICLR 2027
- Budget: full-time summer + multi-GPU cluster
- Framing: endpoint-conditioned route generation as primary contribution
- v13 (TrajDiff diffusion): in scope
- Region: US + Danish DMA cross-region transfer

## Notes index (under `paper/notes/`)

| File | Purpose |
|---|---|
| `traisformer_summary.md` | TrAISformer paper + code analysis; comparison protocol implications |
| `traisformer_data_format.md` | TrAISformer pickle format and our csv→pkl pipeline design |
| `v13_architecture.md` | TrajDiff diffusion model architecture sketch, refined against codebase |
| `dma_data_access.md` | Danish DMA data acquisition plan |
