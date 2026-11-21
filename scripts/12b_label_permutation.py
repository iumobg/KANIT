#!/usr/bin/env python3
"""Step 12b — label permutation test of one panel pair, and its
correction across models.

Subcommands:

  run      null predictions: every outer fold of the first lineage-aware repeat
           refitted on permuted labels. --fold K and --chunk C restrict it to one
           task (label_permutation.chunk permutations of one fold); a finished
           chunk is not run again
  metrics  every chunk of every fold must exist; null AUCs, p and z
  all      run and metrics
  across   Benjamini–Hochberg over every evaluable model of the panel; models with
           q ≥ alpha are flagged permutation_not_significant
           (cross_model_dir/label_permutation.csv)

A model that is not evaluable exits with status 3, as in 04. The rule is in
lib/label_permutation.py. Outputs: paths_organism.label_permutation_dir (null/,
label_permutation_null.csv, label_permutation.json).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import folds, panel  # noqa: E402
from lib import label_permutation as lp  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix  # noqa: E402


def _evaluable(cv_dir: Path) -> bool:
    return bool(json.loads((cv_dir / "cv_design.json").read_text())["evaluable"])


def run(mm, cv_dir, out_dir, cfg, *, fold=None, chunk=None, threads=1):
    fold_list = [f for f in lp.folds_of(mm, cv_dir) if fold is None or f.fold == fold]
    if not fold_list:
        sys.exit(f"ERROR: no outer fold {fold} in the first lineage-aware repeat.")
    parts = lp.chunks(cfg["n_permutations"], cfg["chunk"])
    if chunk is not None and not 0 <= chunk < len(parts):
        sys.exit(f"ERROR: --chunk must lie in 0 .. {len(parts) - 1}.")
    wanted = range(len(parts)) if chunk is None else [chunk]
    labels = mm.labels.astype(int)
    for f in fold_list:
        todo = [c for c in wanted if not lp.chunk_is_current(
            lp.chunk_file(out_dir, f.fold, c),
            lp.chunk_key(f, labels, parts[c], cfg["seed"], mm.fingerprint))]
        if not todo:
            continue
        refit = lp.Refitter(mm, f, threads=threads)
        for c in todo:
            t0 = time.time()
            lp.run_chunk(refit, parts[c], seed=cfg["seed"], path=lp.chunk_file(out_dir, f.fold, c))
            print(f"  ✓ fold {f.fold} chunk {c} ({len(parts[c])} permutations, "
                  f"{time.time() - t0:.0f} s)")


def metrics(mm, cv_dir, out_dir, cfg) -> dict:
    fold_list = lp.folds_of(mm, cv_dir)
    null = lp.null_auc(mm, fold_list, out_dir, n_permutations=cfg["n_permutations"],
                       chunk=cfg["chunk"], seed=cfg["seed"])
    s = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         **lp.summary(lp.observed_auc(mm, fold_list), null, seed=cfg["seed"])}
    pd.DataFrame({"permutation": np.arange(null.size), "auc": null}).to_csv(
        out_dir / "label_permutation_null.csv", index=False)
    (out_dir / "label_permutation.json").write_text(json.dumps(s, indent=2) + "\n")
    return s


def across(config) -> pd.DataFrame:
    cfg = config["label_permutation"]
    pairs = panel.included_pairs(resolve_path("panel_dir", config=config) / "panel_decisions.csv")
    rows, missing = [], []
    for org, ab in pairs:
        cv_dir = resolve_path("cv_dir", organism=org, antibiotic=ab, config=config)
        res = resolve_path("label_permutation_dir", organism=org, antibiotic=ab,
                           config=config) / "label_permutation.json"
        if not (cv_dir / "cv_design.json").exists():
            missing.append(f"{org}/{ab} (no cross-validation)")
        elif not _evaluable(cv_dir):
            rows.append({"organism": org, "antibiotic": ab, "status": "not_evaluable"})
        elif not res.exists():
            missing.append(f"{org}/{ab}")
        else:
            s = json.loads(res.read_text())
            rows.append({"organism": org, "antibiotic": ab, "status": "tested",
                         **{k: s[k] for k in ("auc_observed", "null_mean", "null_sd", "z",
                                              "n_permutations", "p")}})
    if missing:
        sys.exit(f"ERROR: label permutation missing for {len(missing)} model(s), "
                 f"e.g. {missing[:5]}")
    cols = ["organism", "antibiotic", "status", "auc_observed", "null_mean", "null_sd", "z",
            "n_permutations", "p"]
    table = lp.across_models(pd.DataFrame(rows, columns=cols), cfg["alpha"])
    out = resolve_path("cross_model_dir", config=config)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "label_permutation.csv", index=False)
    return table


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="Label permutation test of one panel pair.")
    ap.add_argument("command", choices=["run", "metrics", "all", "across"])
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--threads", type=int, default=int(config["preprocessing"]["threads"]))
    args = ap.parse_args()
    cfg = config["label_permutation"]
    if args.command == "across":
        t = across(config)
        print(f"LABEL PERMUTATION — {int((t['status'] == 'tested').sum())} models tested, "
              f"{int((t['flag'] == lp.FLAG).sum())} flagged {lp.FLAG}")
        return

    def path(key):
        return resolve_path(key, organism=args.organism, antibiotic=args.antibiotic,
                            config=config)

    cv_dir, out_dir = path("cv_dir"), path("label_permutation_dir")
    print(f"LABEL PERMUTATION — {args.organism} / {args.antibiotic} — {args.command}")
    if not _evaluable(cv_dir):
        print("Model not evaluable (see 04's seeds.csv): no label permutation.")
        sys.exit(folds.NOT_EVALUABLE)
    mm = ModelMatrix(path("matrix_dir"))
    if args.command in ("run", "all"):
        run(mm, cv_dir, out_dir, cfg, fold=args.fold, chunk=args.chunk, threads=args.threads)
    if args.command in ("metrics", "all"):
        s = metrics(mm, cv_dir, out_dir, cfg)
        print(f"  observed AUC {s['auc_observed']:.3f}; null {s['null_mean']:.3f} ± "
              f"{s['null_sd']:.3f}; z {s['z']:.1f}; p {s['p']:.4f}")


if __name__ == "__main__":
    main()
