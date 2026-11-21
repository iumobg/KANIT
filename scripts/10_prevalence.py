#!/usr/bin/env python3
"""Step 10 — prevalence layer of one panel pair's candidate patterns.

Prevalence among the resistant and the susceptible genomes of the model, Δ and
its direction, and a Fisher exact test with Benjamini–Hochberg over the
candidates. The rule is in lib/prevalence.py. With --set association, the same for
the patterns nominated by association, adjusted over them
(lib/pattern_sets.py).

Inputs: candidates_file, the model matrix. Output: layers_dir/prevalence.csv.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import pattern_sets  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix  # noqa: E402
from lib.prevalence import prevalence_layer  # noqa: E402


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="Prevalence layer of one panel pair's candidates.")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--set", choices=pattern_sets.SETS, default=pattern_sets.CANDIDATES,
                    help="the patterns (lib/pattern_sets.py)")
    args = ap.parse_args()
    org, ab, cfg = args.organism, args.antibiotic, config["prevalence"]

    def path(key):
        return resolve_path(key, organism=org, antibiotic=ab, config=config)

    ps = pattern_sets.paths(args.set, path)
    patterns = pattern_sets.read_patterns(ps["patterns"])
    layer = prevalence_layer(ModelMatrix(path("matrix_dir")), patterns,
                             min_delta=cfg["min_delta"], alpha=cfg["alpha"])
    out = ps["layers_dir"]
    out.mkdir(parents=True, exist_ok=True)
    layer.to_csv(out / "prevalence.csv", index=False)
    (out / "prevalence_summary.json").write_text(json.dumps({
        "organism": org, "antibiotic": ab, "set": args.set, "n_patterns": len(layer),
        "n_passes": int(layer["passes"].sum()), "min_delta": cfg["min_delta"],
        "alpha": cfg["alpha"]}, indent=2) + "\n")
    print(f"Prevalence — {org} / {ab}: {int(layer['passes'].sum())} of {len(layer)} "
          f"{pattern_sets.LABEL[args.set]} patterns pass")


if __name__ == "__main__":
    main()
