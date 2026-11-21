#!/usr/bin/env python3
"""Step 12 — permutation importance (MDA) of one panel pair's candidate patterns
.

The lineage-aware outer-fold models of 04 (every repeat; all must be finished)
predict their own test folds while each candidate pattern is permuted within the
test folds, R times; the sensitivity analysis permutes clusters of candidates
linked by |r| ≥ mda.cluster_r together. A model that is not evaluable exits with
status 3, as in 04. The rule is in lib/mda.py. With --set association, the same for the
patterns nominated by association, adjusted over them (lib/pattern_sets.py).

Inputs: candidates_file, the model matrix, cv_dir. Outputs (layers_dir):
mda.csv (the layer), mda_clusters.csv (sensitivity analysis), mda_summary.json.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import folds, mda, pattern_sets  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix  # noqa: E402
from lib.run_metadata import peak_rss_gb  # noqa: E402


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="MDA layer of one panel pair's candidates.")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--set", choices=pattern_sets.SETS, default=pattern_sets.CANDIDATES,
                    help="the patterns (lib/pattern_sets.py)")
    args = ap.parse_args()
    org, ab, cfg = args.organism, args.antibiotic, config["mda"]

    def path(key):
        return resolve_path(key, organism=org, antibiotic=ab, config=config)

    cv_dir = path("cv_dir")
    if not json.loads((cv_dir / "cv_design.json").read_text())["evaluable"]:
        print("Model not evaluable (see 04's seeds.csv): no MDA.")
        sys.exit(folds.NOT_EVALUABLE)
    ps = pattern_sets.paths(args.set, path)
    patterns = pattern_sets.read_patterns(ps["patterns"])
    mm = ModelMatrix(path("matrix_dir"))
    t0 = time.time()
    if patterns:
        design = mda.Design(mda.load_fold_models(mm, cv_dir), cfg["n_permutations"],
                            cfg["seed"], mm.labels.astype(int))
        layer, clusters = mda.mda_layer(mm, design, patterns, alpha=cfg["alpha"],
                                        r_min=cfg["cluster_r"])
        fit = {"n_fold_models": len(design.models),
               "auc_observed_by_repeat": {str(r): a for r, a in sorted(design.auc.items())},
               "auc_observed": design.observed}
    else:                       # an empty association set: nothing to permute
        layer, clusters = mda.empty_layer()
        fit = {"n_fold_models": 0, "auc_observed_by_repeat": {}, "auc_observed": None}

    out = ps["layers_dir"]
    out.mkdir(parents=True, exist_ok=True)
    layer.to_csv(out / "mda.csv", index=False)
    clusters.to_csv(out / "mda_clusters.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "organism": org, "antibiotic": ab, **{k: cfg[k] for k in sorted(cfg)},
        **fit, "set": args.set, "n_patterns": len(layer),
        "n_passing": int(layer["passes"].sum()),
        "n_clusters": int(clusters["cluster"].nunique()),
        "n_clusters_passing": int(clusters.drop_duplicates("cluster")["passes"].sum()),
        "seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    (out / "mda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"MDA — {org} / {ab}: {summary['n_passing']} of {len(layer)} "
          f"{pattern_sets.LABEL[args.set]} patterns pass "
          f"(R = {cfg['n_permutations']}); clusters {summary['n_clusters_passing']} of "
          f"{summary['n_clusters']}")


if __name__ == "__main__":
    main()
