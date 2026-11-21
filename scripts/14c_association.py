#!/usr/bin/env python3
"""Step 14c — the patterns nominated by association of one panel pair (a planned
secondary analysis).

The patterns that pass the pyseer layer but are not candidates, at most
association.max_patterns with the smallest p values (lib/pattern_sets.py). This step
writes the set and two of its layers: pyseer, which every pattern of the set passes by the
way it is chosen, and CPSS, from the model's stability selection (a stable pattern is a
candidate, so none of the set passes it). Steps 09, 10, 12 and 14b compute the other
layers and the grades with --set association; the grades are not comparable with those of
the candidates.

Inputs: candidates_file, pyseer_dir/pyseer_tested.csv, cpss_dir/cpss.csv. Outputs
(paths_organism.association_dir): association.csv, layers/pyseer.csv, layers/cpss.csv,
association_summary.json. A model that is not evaluable exits with status 3, as in 04.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import cpss, folds, pattern_sets  # noqa: E402
from lib import pyseer_lmm as pl  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402


def select(path, config: dict) -> dict:
    """Writes the association set of a model and its pyseer and CPSS layers;
    ``path(key)`` resolves a paths_organism key of the model."""
    ps = pattern_sets.paths(pattern_sets.ASSOCIATION, path)
    candidates = pattern_sets.read_patterns(path("candidates_file"))
    tested = pd.read_csv(path("pyseer_dir") / "pyseer_tested.csv", keep_default_na=False,
                         na_values=[""])
    max_patterns = int(config["association"]["max_patterns"])
    chosen = pattern_sets.association(tested, candidates, max_patterns)
    ids = chosen["pattern_id"].tolist()
    pyseer, threshold = pl.results(tested, ids, config["pyseer"]["alpha"])
    if not pyseer["passes"].all():
        raise ValueError("a pattern of the association set does not pass the pyseer layer")
    table = pd.read_csv(path("cpss_dir") / "cpss.csv")
    layer = cpss.layer(ids, table, config["cpss"]["pi_threshold"])
    ps["layers_dir"].mkdir(parents=True, exist_ok=True)
    chosen.to_csv(ps["patterns"], index=False)
    pyseer.to_csv(ps["layers_dir"] / "pyseer.csv", index=False)
    layer.to_csv(ps["layers_dir"] / "cpss.csv", index=False)
    significant = tested.loc[tested["significant"].astype(str).str.lower().isin(
        {"true", "1", "1.0"}), "pattern_id"].astype(int)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_patterns": max_patterns, "n_tested": int(len(tested)),
        "bonferroni_threshold": threshold, "n_significant": int(len(significant)),
        "n_significant_candidates": int(significant.isin(set(candidates)).sum()),
        "n_significant_not_candidates": int((~significant.isin(set(candidates))).sum()),
        "n_patterns": int(len(chosen)), "n_cpss_passing": int(layer["passes"].sum()),
        "largest_p": float(chosen["lrt_pvalue"].max()) if len(chosen) else None,
    }
    (ps["patterns"].parent / "association_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="Patterns nominated by association of one pair.")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    args = ap.parse_args()

    def path(key):
        return resolve_path(key, organism=args.organism, antibiotic=args.antibiotic,
                            config=config)

    print(f"ASSOCIATION — {args.organism} / {args.antibiotic}")
    if not json.loads((path("cv_dir") / "cv_design.json").read_text())["evaluable"]:
        print("Model not evaluable (see 04's seeds.csv): no association set.")
        sys.exit(folds.NOT_EVALUABLE)
    s = select(path, config)
    print(f"  {s['n_patterns']} of the {s['n_significant_not_candidates']} significant "
          f"patterns that are not candidates (at most {s['max_patterns']}; Bonferroni "
          f"{s['bonferroni_threshold']:.2e})")


if __name__ == "__main__":
    main()
