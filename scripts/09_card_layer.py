#!/usr/bin/env python3
"""Step 09 — CARD layer of the candidate unitigs of one panel pair.

Every member unitig of every candidate pattern is located in up to three
carrier genomes of the model and compared with their RGI hits (08); the result
is one state per unitig and per pattern, in two modes: ``allele_aware`` (the
primary mode) and ``homolog_only`` (its sensitivity analysis). The rule is in
lib/card_layer.py. With --set association, the same for the patterns nominated by
association (lib/pattern_sets.py).

Inputs: candidates_file (column pattern_id), the model matrix, the unitig store,
02d's QC table, the organism's rgi_hits.csv, CARD's rRNA references.
Outputs (paths_organism.card_layer_dir): card_unitigs.csv, card_patterns.csv,
card_summary.json.
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import card_layer, panel, pattern_sets, registry  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix, Store  # noqa: E402


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="CARD layer of one panel pair's candidates.")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--set", choices=pattern_sets.SETS, default=pattern_sets.CANDIDATES,
                    help="the patterns (lib/pattern_sets.py)")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="CSV with a pattern_id column (default: the file of --set)")
    args = ap.parse_args()
    org, ab, card = args.organism, args.antibiotic, config["card"]

    def path(key, **kw):
        return resolve_path(key, organism=org, config=config, **kw)

    ps = pattern_sets.paths(args.set, lambda key: path(key, antibiotic=ab))
    cand_file = args.candidates or ps["patterns"]
    patterns = pd.read_csv(cand_file)["pattern_id"].astype(int).tolist()
    targets = registry.card_drug_classes(ab)
    if not targets:
        sys.exit(f"ERROR: {ab} has no CARD drug class terms in the antibiotic registry.")
    hits = pd.read_csv(path("rgi_dir") / "rgi_hits.csv", dtype={"genome_id": str, "aro": str,
                                                               "model_id": str})
    hits["snps"] = hits["snps"].fillna("")
    qc = pd.read_csv(panel.input_paths(org, config)["qc_table"], dtype={"genome_id": str})
    fna_dir = path("raw_genomes_dir")

    @lru_cache(maxsize=16)
    def contigs_of(genome):
        return card_layer.read_fasta(fna_dir / f"{genome}.fna")

    ref_dna = (card_layer.rrna_references(PROJECT_ROOT / card["card_json"])
               if (hits["model_type"] == "rrna_variant").any() else {})
    unitigs, pats, summary = card_layer.annotate(
        patterns, ModelMatrix(path("matrix_dir", antibiotic=ab)), Store(path("unitig_store_dir")),
        qc, hits, contigs_of, targets, near_universal_share=card["near_universal"],
        max_located=card["max_located_genomes"], min_overlap=card["min_overlap"],
        ref_dna=ref_dna)

    out = ps["card_layer_dir"]
    out.mkdir(parents=True, exist_ok=True)
    unitigs.to_csv(out / "card_unitigs.csv", index=False)
    pats.to_csv(out / "card_patterns.csv", index=False)
    (out / "card_summary.json").write_text(json.dumps(
        {**summary, "organism": org, "antibiotic": ab, "set": args.set,
         "candidates_file": str(cand_file)},
        indent=2) + "\n")
    print(f"CARD layer — {org} / {ab}: {summary['n_patterns']} patterns, "
          f"{summary['n_unitigs']} unitigs; {summary.get('patterns_by_state', {})}")


if __name__ == "__main__":
    main()
