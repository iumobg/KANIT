#!/usr/bin/env python3
"""Step 14b — grades of one panel pair's candidate patterns.

Every candidate pattern is graded from its CARD state (09) and the four
statistical layers, in both CARD modes (allele_aware and homolog_only); its
member unitigs take its grade. config grading.primary_mode names the mode reported
as ``grade``. The rule is in lib/grading.py. With --set association, the same for the
patterns nominated by association (lib/pattern_sets.py), from their own
layers and CARD layer.

Inputs: candidates_file; card_layer_dir/card_patterns.csv and card_unitigs.csv
(09); layers_dir/{prevalence,mda,cpss,pyseer}.csv, each with pattern_id and
passes. Outputs (paths_organism.grades_dir): grades_patterns.csv,
grades_unitigs.csv, grades_summary.json.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import grading, pattern_sets  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="Grades of one panel pair's candidate patterns.")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--set", choices=pattern_sets.SETS, default=pattern_sets.CANDIDATES,
                    help="the patterns (lib/pattern_sets.py)")
    args = ap.parse_args()
    org, ab, primary = args.organism, args.antibiotic, config["grading"]["primary_mode"]

    def path(key):
        return resolve_path(key, organism=org, antibiotic=ab, config=config)

    ps = pattern_sets.paths(args.set, path)
    patterns = set(pattern_sets.read_patterns(ps["patterns"]))
    card_dir = ps["card_layer_dir"]
    card_patterns = pd.read_csv(card_dir / "card_patterns.csv", keep_default_na=False)
    card_unitigs = pd.read_csv(card_dir / "card_unitigs.csv", keep_default_na=False)
    if set(card_patterns["pattern_id"].astype(int)) != patterns:
        sys.exit(f"ERROR: the CARD layer was not computed for the current "
                 f"{pattern_sets.LABEL[args.set]} patterns; rerun 09.")
    layers = {layer: grading.read_layer(ps["layers_dir"] / f"{layer}.csv", layer)
              for layer in grading.LAYERS}

    graded = grading.grade_patterns(card_patterns, layers, primary)
    unitigs = grading.grade_unitigs(card_unitigs, graded)
    summ = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "organism": org, "antibiotic": ab, "set": args.set, **grading.summary(graded, primary),
            "n_unitigs": len(unitigs)}

    out = ps["grades_dir"]
    out.mkdir(parents=True, exist_ok=True)
    graded.to_csv(out / "grades_patterns.csv", index=False)
    unitigs.to_csv(out / "grades_unitigs.csv", index=False)
    (out / "grades_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(f"Grades — {org} / {ab} ({primary}): {summ['patterns_by_grade'][primary]}")


if __name__ == "__main__":
    main()
