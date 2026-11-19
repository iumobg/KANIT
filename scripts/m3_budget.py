#!/usr/bin/env python3
"""Unitig–k-mer comparison, arms B and C — the feature budget of one panel pair (lib/budget.py).

Chooses the K features with the largest χ² in every training set of the model (each
outer fold of arm A's cross-validation design, then all genomes) and writes:

  matrix_dir/                  the final model's features (a model matrix)
  matrix_dir/cv_union/         the outer folds' features and unit_columns.npz
  matrix_dir/selection.csv.gz  every training set's choice
  cv_dir/                      arm A's folds.csv, seeds.csv, fold_composition.csv and
                               cv_design.json: the arms share their folds

Features: arm A's patterns (m3.representation = unitig, arm B) or the k-mers of the
organism's k-mer store (kmer, arm C). Run with the arm's overlay (config/m3_B.yml or
config/m3_C.yml): it moves matrix_dir and cv_dir and names arm A's as source_matrix_dir and
source_cv_dir. A model that is not evaluable exits with status 3, as in 04.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import budget, folds  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix, Store  # noqa: E402
from lib.run_metadata import peak_rss_gb  # noqa: E402

DESIGN_FILES = ("folds.csv", "seeds.csv", "fold_composition.csv", "cv_design.json")


def _source(src: ModelMatrix, representation: str, config: dict, organism: str):
    """Feature blocks, column getter and members of the chosen features, and a summary."""
    if representation == "unitig":
        mem = src.members()
        n_members = pd.read_csv(src.dir / "patterns.csv")["n_members"].to_numpy()

        def members(chosen):
            m = mem[mem["pattern_id"].isin(chosen)]
            return (np.column_stack([m["unitig_index"].to_numpy(),
                                     np.searchsorted(chosen, m["pattern_id"].to_numpy())]),
                    n_members[chosen])
        info = {k: src.summary.get(k) for k in ("store_dir", "store_source_sha256", "min_support")}
        return budget.pattern_blocks(src), src.columns, members, {**info, "source": str(src.dir)}

    if representation == "kmer":
        store_dir = resolve_path("kmer_store_dir", organism=organism, config=config)
        store = Store(store_dir)
        pos = {g: i for i, g in enumerate(store.genomes)}
        ids = src.genomes["Genome ID"].astype(str)
        missing = [g for g in ids if g not in pos]
        if missing:
            sys.exit(f"ERROR: {len(missing)} genome(s) not in {store_dir}, e.g. {missing[:3]}")
        cols = np.array([pos[g] for g in ids], dtype=np.int64)
        ms = int(config["unitig"]["min_support"])

        def members(chosen):
            return np.column_stack([chosen, np.arange(chosen.size)]), np.ones(chosen.size)
        info = {"store_dir": str(store_dir), "min_support": ms,
                "store_source_sha256": store.summary.get("source_sha256"),
                "source": str(store_dir), "k": store.summary.get("k")}
        return (budget.store_blocks(store, cols, ms),
                lambda chosen: budget.store_columns(store, chosen, cols), members, info)
    sys.exit(f"ERROR: m3.representation must be unitig or kmer, not {representation!r}")


def run(organism: str, antibiotic: str, config: dict) -> dict:
    m3 = config.get("m3") or {}
    k, representation = int(m3["feature_budget"]), str(m3["representation"])
    kw: dict[str, Any] = {"organism": organism, "antibiotic": antibiotic, "config": config}
    src = ModelMatrix(resolve_path("source_matrix_dir", **kw))
    src_cv = resolve_path("source_cv_dir", **kw)
    matrix_dir, cv_dir = resolve_path("matrix_dir", **kw), resolve_path("cv_dir", **kw)
    for mine, theirs in ((matrix_dir, src.dir), (cv_dir, src_cv)):
        if mine.resolve() == Path(theirs).resolve():
            sys.exit(f"ERROR: {mine} is arm A's; run with an arm's overlay (config/m3_*.yml)")
    design = json.loads((src_cv / "cv_design.json").read_text())
    if not design["evaluable"]:
        print("Model not evaluable (arm A's design); no budget.")
        sys.exit(folds.NOT_EVALUABLE)

    t0 = time.time()
    ids = src.genomes["Genome ID"].astype(str).to_numpy()
    sets = budget.training_sets(src_cv / "folds.csv", src_cv / "seeds.csv", ids)
    names = [name for name, _ in sets]
    topk = budget.TopK(src.labels, np.stack([m for _, m in sets]), k)
    blocks, get, members, info = _source(src, representation, config, organism)
    for fid, bits in blocks:
        topk.update(fid, bits)
    sel = topk.result(names)
    seconds_select = round(time.time() - t0, 1)

    final_ids = np.sort(sel.loc[sel["set"] == budget.FINAL, "feature_id"].to_numpy())
    units = sel[sel["set"] != budget.FINAL]
    union_ids = np.unique(units["feature_id"].to_numpy())
    common = {**info, "representation": representation, "feature_budget": k,
              "n_features_source": topk.n_features, "seconds_select": seconds_select}
    matrix_dir.mkdir(parents=True, exist_ok=True)
    sel.to_csv(matrix_dir / "selection.csv.gz", index=False)
    union = matrix_dir / budget.UNION
    shutil.rmtree(union, ignore_errors=True)
    mem, n_mem = members(union_ids)
    budget.write_matrix(union, src.genomes, get(union_ids), union_ids, mem, n_mem,
                        {**common, "selection": "union of the outer folds"})
    unit_cols: dict[str, Any] = {
        str(u): np.searchsorted(union_ids, np.sort(g["feature_id"].to_numpy()))
        for u, g in units.groupby("set")}
    np.savez(union / budget.UNIT_COLUMNS, **unit_cols)
    mem, n_mem = members(final_ids)
    s = budget.write_matrix(matrix_dir, src.genomes, get(final_ids), final_ids, mem, n_mem,
                            {**common, "selection": budget.FINAL})
    cv_dir.mkdir(parents=True, exist_ok=True)
    for name in DESIGN_FILES:
        shutil.copy2(src_cv / name, cv_dir / name)
    return {"n_features_source": topk.n_features, "n_final": s["n_patterns"],
            "n_union": int(union_ids.size), "n_sets": len(names),
            "seconds": round(time.time() - t0, 1)}


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="Feature budget of one panel pair (unitig–k-mer comparison).")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    args = ap.parse_args()
    if "m3" not in config:
        sys.exit("ERROR: no m3 block; run with an arm's overlay (KANIT_CONFIG_OVERLAY)")
    print(f"FEATURE BUDGET — {args.organism} / {args.antibiotic} — "
          f"{config['m3']['representation']}, K = {config['m3']['feature_budget']}")
    r = run(args.organism, args.antibiotic, config)
    print(f"  {r['n_features_source']:,} features → final {r['n_final']:,}, union of "
          f"{r['n_sets'] - 1} outer folds {r['n_union']:,}; {r['seconds']} s; "
          f"peak {peak_rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
