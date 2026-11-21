#!/usr/bin/env python3
"""Step 13 — stability selection (CPSS) of one panel pair and its
candidate patterns.

Subcommands:

  prefilter  the cpss.prefilter patterns with the largest χ² over all genomes
  run        the fits of B pairs of half-samples with the final model's settings;
             --chunk C restricts it to one task (cpss.chunk pairs); a finished
             chunk is not run again
  select     every chunk must exist: selection frequencies, stable patterns, the
             Meinshausen–Bühlmann bound, the CPSS layer (layers_dir/cpss.csv) and
             the candidates (candidates_file: top total gain of the final model and
             the stable patterns)
  all        prefilter, run and select

The final model of 04 must exist. A model that is not evaluable exits with
status 3, as in 04. The rule is in lib/cpss.py. Outputs: paths_organism.cpss_dir
(prefilter.csv, pairs/, cpss.csv, cpss_summary.json), the CPSS layer and the
candidates.
"""
import argparse
import functools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import cpss, folds  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix  # noqa: E402


def _final(cv_dir: Path) -> tuple[dict, Path]:
    rec, model = cv_dir / "final" / "record.json", cv_dir / "final" / "model.ubj"
    if not rec.exists() or not model.exists():
        sys.exit(f"ERROR: the final model is missing ({cv_dir / 'final'}); run 04 final.")
    return json.loads(rec.read_text()), model


def prefilter(mm, out_dir: Path, cfg) -> pd.DataFrame:
    table = cpss.chi2_prefilter(mm, cfg["prefilter"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "prefilter.csv", index=False)
    return table


def _prefilter_table(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "prefilter.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found; run `prefilter` first.")
    return pd.read_csv(path)


def _fit_params(rec: dict, mm, n_prefilter: int) -> dict:
    """The final model's hyperparameters, colsample_bytree rescaled to the prefilter."""
    return {**rec["params"], "colsample_bytree": cpss.rescaled_colsample(
        rec["params"]["colsample_bytree"], mm.n_patterns, n_prefilter)}


def _key(rec: dict, mm, pre: pd.DataFrame, cfg):
    """pairs -> the key of a chunk's inputs (lib.cpss.chunk_key)."""
    return functools.partial(cpss.chunk_key, params=_fit_params(rec, mm, len(pre)),
                             n_trees=int(rec["n_trees"]), q=cfg["q"], seed=cfg["seed"],
                             pattern_ids=pre["pattern_id"].to_numpy(), y=mm.labels.astype(int),
                             matrix=mm.fingerprint)


def run(mm, cv_dir: Path, out_dir: Path, cfg, *, chunk=None, threads=1):
    rec, _ = _final(cv_dir)
    pre = _prefilter_table(out_dir)
    parts = cpss.chunks(cfg["n_pairs"], cfg["chunk"])
    if chunk is not None and not 0 <= chunk < len(parts):
        sys.exit(f"ERROR: --chunk must lie in 0 .. {len(parts) - 1}.")
    key = _key(rec, mm, pre, cfg)
    todo = [c for c in (range(len(parts)) if chunk is None else [chunk])
            if not cpss.chunk_is_current(cpss.chunk_file(out_dir, c), key(parts[c]))]
    if not todo:
        return
    x = mm.columns(pre["pattern_id"].to_numpy())
    params = _fit_params(rec, mm, len(pre))
    y = mm.labels.astype(int)
    for c in todo:
        t0 = time.time()
        cpss.run_chunk(x, y, parts[c], params=params, n_trees=int(rec["n_trees"]), q=cfg["q"],
                       seed=cfg["seed"], threads=threads, path=cpss.chunk_file(out_dir, c),
                       key=key(parts[c]))
        print(f"  ✓ chunk {c} ({len(parts[c])} pairs, {time.time() - t0:.0f} s)")


def select(mm, cv_dir: Path, out_dir: Path, cfg, *, top_gain: int, candidates_file: Path,
           layers_dir: Path) -> dict:
    rec, model = _final(cv_dir)
    pre = _prefilter_table(out_dir)
    pi = cpss.selection_frequency(out_dir, len(pre), n_pairs=cfg["n_pairs"], chunk=cfg["chunk"],
                                  key=_key(rec, mm, pre, cfg))
    table = pre[["pattern_id", "chi2"]].assign(
        n_selected=np.rint(pi * 2 * cfg["n_pairs"]).astype(int), pi=pi,
        stable=cpss.stable(pi, cfg["pi_threshold"]))
    table.to_csv(out_dir / "cpss.csv", index=False)
    cands = cpss.candidates(xgb.Booster(model_file=str(model)), table, top_gain)
    candidates_file.parent.mkdir(parents=True, exist_ok=True)
    cands.to_csv(candidates_file, index=False)
    layer = cpss.layer(cands["pattern_id"], table, cfg["pi_threshold"])
    layers_dir.mkdir(parents=True, exist_ok=True)
    layer.to_csv(layers_dir / "cpss.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: cfg[k] for k in ("prefilter", "n_pairs", "q", "pi_threshold", "seed")},
        "n_prefilter": int(len(pre)), "n_fits": 2 * cfg["n_pairs"],
        "final_colsample_bytree": rec["params"]["colsample_bytree"],
        "cpss_colsample_bytree": cpss.rescaled_colsample(rec["params"]["colsample_bytree"],
                                                         mm.n_patterns, len(pre)),
        "n_trees": int(rec["n_trees"]), "n_stable": int(table["stable"].sum()),
        "mb_bound": cpss.mb_bound(cfg["q"], cfg["pi_threshold"], len(pre)),
        "n_candidates": int(len(cands)),
        "candidates_by_source": cands["source"].value_counts().to_dict(),
    }
    (out_dir / "cpss_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="Stability selection of one panel pair.")
    ap.add_argument("command", choices=["prefilter", "run", "select", "all"])
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--threads", type=int, default=int(config["preprocessing"]["threads"]))
    args = ap.parse_args()
    cfg = config["cpss"]

    def path(key):
        return resolve_path(key, organism=args.organism, antibiotic=args.antibiotic,
                            config=config)

    cv_dir, out_dir = path("cv_dir"), path("cpss_dir")
    print(f"CPSS — {args.organism} / {args.antibiotic} — {args.command}")
    if not json.loads((cv_dir / "cv_design.json").read_text())["evaluable"]:
        print("Model not evaluable (see 04's seeds.csv): no CPSS.")
        sys.exit(folds.NOT_EVALUABLE)
    mm = ModelMatrix(path("matrix_dir"))
    if args.command in ("prefilter", "all"):
        t = prefilter(mm, out_dir, cfg)
        print(f"  prefilter: {len(t)} patterns, χ² {t['chi2'].iloc[0]:.1f} .. "
              f"{t['chi2'].iloc[-1]:.1f}")
    if args.command in ("run", "all"):
        run(mm, cv_dir, out_dir, cfg, chunk=args.chunk, threads=args.threads)
    if args.command in ("select", "all"):
        s = select(mm, cv_dir, out_dir, cfg, top_gain=config["candidates"]["top_gain"],
                   candidates_file=path("candidates_file"), layers_dir=path("layers_dir"))
        print(f"  stable {s['n_stable']} (π ≥ {cfg['pi_threshold']}, MB bound "
              f"{s['mb_bound']:.2f}); candidates {s['n_candidates']} {s['candidates_by_source']}")


if __name__ == "__main__":
    main()
