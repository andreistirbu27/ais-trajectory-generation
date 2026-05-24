#!/usr/bin/env python3
"""
train_traisformer.py — Launcher for TrAISformer's upstream training script.

TrAISformer ships as a script-style codebase: `paper/external/TrAISformer/
trAISformer.py` reads `Config()` at module load and runs the whole training
under `if __name__ == "__main__":`. This wrapper:

1. Optionally overrides any `Config` field via CLI flags (so we can point
   it at our paper-track output dir, change the dataset, tweak epochs, etc.
   without modifying the upstream file).
2. cd's into the TrAISformer repo (its data/savedir paths are relative).
3. Runs their training entry point.

Two intended uses:

A. **DMA reproduction sanity check** — verify our env reproduces their
   published numbers using the shipped ct_dma pickles:

    paper/.venv/bin/python3 scripts/paper/train_traisformer.py \\
        --max_epochs 50

   That uses every default from config_trAISformer.py. Outputs land under
   paper/external/TrAISformer/results/<their auto-generated name>/.

B. **US sub-bbox retrain** (later) — point at converted US data + override
   bbox + bin sizes:

    paper/.venv/bin/python3 scripts/paper/train_traisformer.py \\
        --dataset_name ct_us_gulf \\
        --lat_min 30 --lat_max 32 --lon_min -85 --lon_max -82 \\
        --lat_size 250 --lon_size 270 --max_epochs 50

   Requires the US data to be pre-converted via
   `scripts/paper/csv_to_traisformer_pkl.py` and placed at
   `paper/external/TrAISformer/data/<dataset_name>/`.

Notes
-----
- TrAISformer's training script does its own checkpointing and eval at the
  end. We don't intercept; we just run their pipeline. The K-anchor adapter
  (separate script) loads the resulting checkpoint.
- We **do not** modify any file under paper/external/TrAISformer/. To change
  their config, pass overrides here. To change their model, fork their file
  into paper/external/TrAISformer/<something>_paper.py and route imports
  there.
- Their script uses logging into `cf.savedir`. We capture stdout via
  subprocess.PIPE so the wrapper's --log_to file also has everything.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIS_DIR = REPO_ROOT / "paper" / "external" / "TrAISformer"


# Fields that we expose for override. Maps CLI flag → (config attr, type).
OVERRIDES = {
    # Training schedule
    "max_epochs":       ("max_epochs",       int),
    "batch_size":       ("batch_size",       int),
    "learning_rate":    ("learning_rate",    float),
    # Sequence shape
    "init_seqlen":      ("init_seqlen",      int),
    "max_seqlen":       ("max_seqlen",       int),
    "min_seqlen":       ("min_seqlen",       int),
    # Dataset selection (must already exist under data/<dataset_name>/)
    "dataset_name":     ("dataset_name",     str),
    # Bounding box (must match how the pkls were normalized)
    "lat_min":          ("lat_min",          float),
    "lat_max":          ("lat_max",          float),
    "lon_min":          ("lon_min",          float),
    "lon_max":          ("lon_max",          float),
    # Discretisation
    "lat_size":         ("lat_size",         int),
    "lon_size":         ("lon_size",         int),
    "sog_size":         ("sog_size",         int),
    "cog_size":         ("cog_size",         int),
    # Model shape
    "n_layer":          ("n_layer",          int),
    "n_head":           ("n_head",           int),
    # Run-control
    "retrain":          ("retrain",          lambda s: s.lower() in ("1", "true", "t", "yes")),
    "tb_log":           ("tb_log",           lambda s: s.lower() in ("1", "true", "t", "yes")),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    p.add_argument("--dry_run", action="store_true",
                   help="Print the patched Config and the command, then exit.")
    p.add_argument("--log_to", default=None,
                   help="Tee stdout/stderr to this file (in addition to console).")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use for the subprocess. Defaults "
                        "to the current interpreter.")
    for cli_name, (attr, _) in OVERRIDES.items():
        p.add_argument(f"--{cli_name}",
                       help=f"Override Config.{attr} in config_trAISformer.py.")
    return p


def build_patch_script(args) -> str:
    """Generate a small Python snippet that imports their Config, applies
    overrides, then exec's their trAISformer.py with the patched module.

    We do this rather than editing their file so the upstream codebase stays
    pristine.
    """
    # Each line gets 8-space prefix because the template lives inside a
    # textwrap.dedent that strips 8 spaces uniformly (4 for the def, 4 for
    # being inside the dedent block). Mismatched indent => IndentationError.
    overrides_python_lines = []
    for cli_name, (attr, caster) in OVERRIDES.items():
        val = getattr(args, cli_name, None)
        if val is None:
            continue
        cast_val = caster(val)
        overrides_python_lines.append(f"        Config.{attr} = {cast_val!r}")
    overrides_block = "\n".join(overrides_python_lines) or "        pass  # no overrides"

    return textwrap.dedent(f"""\
        import os, sys, runpy
        # Make their repo importable + working dir.
        sys.path.insert(0, {str(TRAIS_DIR)!r})
        os.chdir({str(TRAIS_DIR)!r})

        from config_trAISformer import Config
        # ── overrides ───────────────────────────────────────────────
{overrides_block}
        # Recompute fields that depend on overridden values.
        Config.full_size  = Config.lat_size + Config.lon_size + Config.sog_size + Config.cog_size
        Config.n_embd     = Config.n_lat_embd + Config.n_lon_embd + Config.n_sog_embd + Config.n_cog_embd
        Config.datadir    = f"./data/{{Config.dataset_name}}/"
        Config.trainset_name = f"{{Config.dataset_name}}_train.pkl"
        Config.validset_name = f"{{Config.dataset_name}}_valid.pkl"
        Config.testset_name  = f"{{Config.dataset_name}}_test.pkl"
        Config.filename = (
            f"{{Config.dataset_name}}"
            f"-{{Config.mode}}-{{Config.sample_mode}}-{{Config.top_k}}-{{Config.r_vicinity}}"
            f"-blur-{{Config.blur}}-{{Config.blur_learnable}}-{{Config.blur_n}}-{{Config.blur_loss_w}}"
            f"-data_size-{{Config.lat_size}}-{{Config.lon_size}}-{{Config.sog_size}}-{{Config.cog_size}}"
            f"-embd_size-{{Config.n_lat_embd}}-{{Config.n_lon_embd}}-{{Config.n_sog_embd}}-{{Config.n_cog_embd}}"
            f"-head-{{Config.n_head}}-{{Config.n_layer}}"
            f"-bs-{{Config.batch_size}}"
            f"-lr-{{Config.learning_rate}}"
            f"-seqlen-{{Config.init_seqlen}}-{{Config.max_seqlen}}"
        )
        Config.savedir   = "./results/" + Config.filename + "/"
        Config.ckpt_path = os.path.join(Config.savedir, "model.pt")

        # Print the effective config for the log.
        print("=" * 70)
        print("Effective TrAISformer Config:")
        for k in sorted(vars(Config)):
            if k.startswith("__"): continue
            v = getattr(Config, k)
            if callable(v): continue
            print(f"  {{k}} = {{v!r}}")
        print("=" * 70)
        sys.stdout.flush()

        # Run their entry-point under __main__ semantics.
        runpy.run_path("trAISformer.py", run_name="__main__")
        """)


def main():
    args = build_parser().parse_args()

    if not TRAIS_DIR.exists():
        print(f"ERROR: TrAISformer repo not found at {TRAIS_DIR}")
        sys.exit(2)
    if not (TRAIS_DIR / "trAISformer.py").exists():
        print(f"ERROR: {TRAIS_DIR / 'trAISformer.py'} missing")
        sys.exit(2)

    patch = build_patch_script(args)

    if args.dry_run:
        print("=" * 70)
        print("Dry run — would execute the following inline script:")
        print("=" * 70)
        print(patch)
        print("=" * 70)
        return

    cmd = [args.python, "-u", "-c", patch]
    print(f"$ {' '.join(shlex.quote(c) for c in cmd[:3])} <patched-script>")
    if args.log_to:
        os.makedirs(os.path.dirname(args.log_to) or ".", exist_ok=True)
        with open(args.log_to, "ab", buffering=0) as logf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for chunk in iter(lambda: proc.stdout.read(4096), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logf.write(chunk)
            rc = proc.wait()
    else:
        rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
