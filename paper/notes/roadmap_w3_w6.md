# ICLR submission roadmap — W3–W6

Downstream of the W1/W2 work (done 2026-06-28: repo hygiene, reproducibility gate passed,
pinned deps + smoke tests, TrAISformer best-of-1 column + inference-latency table). Full
strategic plan lives at `~/.claude/plans/goal-audit-clean-organize-glittery-stroustrup.md`.
Target: complete supervisor-review draft well before the ICLR 2027 abstract (~mid/late Sept
2026 backstop). Conservative-hygiene constraint holds — no package refactor until after
acceptance.

## Status snapshot (what's already banked)
- US headline reproduced **exactly** (6282 m); DMA K=2 b16 reproduced within subsample noise.
- Exact params: v10 **5.84M** trainable (5.97M with buffers) vs TrAISformer **57.4M** → ~10×.
- Latency: v10 single-shot 569 / 75.8 ms-per-traj (B=1/32) vs TrAISformer×16 1312 / 365.
- Paper: abstract, related work, K-anchor protocol, main results, ablation ladder, transfer,
  stats all drafted; `tab:dma_k_anchor` now has b1+b16 rows; `tab:latency` added.

## W3 — Ablation gaps (credibility-hardening; mostly eval-only)
Reviewer-bait the paper currently narrates but does not tabulate.
- **Vessel-type ablation row** in `tab:ablation` (`paper/tables/ablation_ladder.tex`). The only
  item needing a *small retrain*: drop the vessel-type token, retrain a v10 variant, eval. Text
  at §5.4 already claims "~200 m ADE" — replace prose with a tabulated row.
- **Retrieval-K sensitivity** K∈{1,5,10}: eval-only on existing checkpoints (vary the retrieval
  bank at inference). Adds a small table or figure; supports "per-timestep retrieval earns its
  keep."
- **Failure-mode quantification**: bucket the worst-N high-ADE trajectories by cause (narrow
  strait / port maneuver / mid-route vessel-type change) and report counts. Turns §appendix
  prose into numbers. Eval-only.
- Optional: exact param counts already measured → drop into the HP table (`tab:hp`).

## W4 — Paper polish
- Apply the **ICLR style file** + proper title page (strip the leftover school-report
  `\thesistitle` scaffolding); confirm `\documentclass` switch from `article`.
- **PDF compiles end-to-end** with no missing refs (bib audit already clean: all 18 cites
  resolve in `paper/bib/refs.bib`).
- Tighten the novelty framing: lead with the **K-anchor protocol** (reusable benchmark) and the
  **differentiable-constraint-as-general-mechanism**, not just AIS numbers.
- Keep `paper/paper.tex` ↔ `paper/overleaf_bundle/paper.tex` byte-identical (`diff` empty).

## W5 — Anonymized release repo
- Stripped, anonymized bundle: code + configs + checkpoints + the NOAA→`trajgen_128_clean.npz`
  build script. A reviewer should be able to clone and run `python3 tests/test_smoke.py`.
- README with one-line repro commands for the headline tables (US + DMA K-anchor + latency).
- **Apply the carried TrAISformer patch at setup**: a fresh `git submodule update --init` gets
  upstream `433de39` (clean `trainers.py`); run
  `git -C paper/external/TrAISformer apply ../patches/0001-pytorch-compat-next-iter.patch`.
- Decide release vehicle (private GitHub vs scrubbed tarball) and confirm data/checkpoint
  license (NOAA-derived — expected fine).

## W6 — Supervisor draft v1 + feedback checkpoint
- Send the complete PDF + anonymized repo to the supervisor.
- Incorporate feedback; only then consider the *nice-to-have* baselines (LSTM seq2seq route
  generator; constant-velocity floor at large K) if schedule allows.

## Deferred / optional (not blocking submission)
- Full **n=500 DMA exact-match** rerun (~50 min) — only if a reviewer-facing exact 5758 is
  wanted; the spot-check already passed.
- `.gitignore` `*.txt`/`*.md` **footgun** tightening (scope ignores to `data/` etc.) — risky to
  change pre-deadline; do during release prep.
- Track the 3 `runs/paper/v13_trajdiff_*/config.yaml` for v13 reproducibility (v13 is the
  subordinate/exploratory model; bundle in the release, not now).
