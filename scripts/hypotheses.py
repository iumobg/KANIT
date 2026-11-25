#!/usr/bin/env python3
"""Hypothesis tests on the knowledge base.

Reads kb_dir/kanit.sqlite, CARD's ARO index (next to card.card_json, or --aro-index)
and, for H7 and H6, every organism's AMRFinderPlus calls
(external_dir/amrfinder_calls.csv and amrfinder_genomes.csv, written by step 16). An
organism without them is left out of H7 and H6 and listed in the summary; one whose
reference misses a model genome stops the run. So does an ARO index from another CARD
release than the knowledge base's (check_index()): reference databases ignore an
overlay's paths_prefix, so a knowledge base built elsewhere meets the local CARD. The
tests are in lib/hypotheses.py.

Outputs (cross_model_dir/hypotheses/): one CSV per table and hypotheses.json with
every criterion, statistic and verdict, and under sensitivity_label_permutation the
same summaries without the models the label permutation flags.
"""
import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import hypotheses as hy  # noqa: E402
from lib import registry  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402
from lib.databases import card_version  # noqa: E402

CALL_COLUMNS = ["genome_id", "element_symbol", "type", "subtype", "scope", "class", "subclass"]


def _plain(x):
    """JSON-safe values: numpy scalars to Python, NaN to null."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def references(conn, config) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """AMRFinderPlus calls of every organism that has them, and those that do not."""
    out, missing = {}, []
    for org in [r[0] for r in conn.execute("SELECT DISTINCT organism_id FROM model ORDER BY 1")]:
        d = resolve_path("external_dir", organism=org, config=config)
        calls_f, genomes_f = d / "amrfinder_calls.csv", d / "amrfinder_genomes.csv"
        if not calls_f.exists() or not genomes_f.exists():
            missing.append(org)
            continue
        analysed = set(pd.read_csv(genomes_f, dtype={"genome_id": str})["genome_id"])
        needed = {r[0] for r in conn.execute(
            "SELECT DISTINCT mg.genome_id FROM model_genome mg JOIN model m USING (model_id) "
            "WHERE m.organism_id = ?", (org,))}
        if needed - analysed:
            sys.exit(f"ERROR: AMRFinderPlus did not analyse {len(needed - analysed)} model "
                     f"genome(s) of {org}, e.g. {sorted(needed - analysed)[:3]}")
        calls = pd.read_csv(calls_f, dtype={"genome_id": str}, keep_default_na=False)
        absent = sorted(set(CALL_COLUMNS) - set(calls.columns))
        if absent:
            sys.exit(f"ERROR: {calls_f}: columns missing: {absent}")
        out[org] = calls[CALL_COLUMNS]
    return out, missing


def check_index(conn, aro_file: Path, aro_index: pd.DataFrame, kb_card: str | None) -> None:
    """Stop unless the ARO index is from the CARD release the knowledge base was built
    with: the version of the card.json next to it, when there is one, and the AMR gene
    families of every CARD term the knowledge base holds."""
    hint = f"pass the ARO index of CARD {kb_card or '(not recorded)'} with --aro-index"
    card_json = aro_file.parent / "card.json"
    if kb_card and card_json.exists() and (v := card_version(card_json)) != kb_card:
        sys.exit(f"ERROR: {aro_file.parent} holds CARD {v}, the knowledge base was built "
                 f"with CARD {kb_card}; {hint}")
    bad = hy.index_mismatches(conn, aro_index)
    if bad:
        sys.exit(f"ERROR: {aro_file} lacks or files under other AMR gene families {len(bad)} "
                 f"CARD term(s) of the knowledge base, e.g. {bad[:3]}; {hint}")


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="Hypothesis tests on the knowledge base.")
    ap.add_argument("--kb", type=Path, default=None,
                    help="knowledge base (default: paths_organism.kb_dir/kanit.sqlite)")
    ap.add_argument("--aro-index", type=Path, default=None,
                    help="CARD's ARO index of the release the knowledge base was built with "
                         "(default: aro_index.tsv next to card.card_json)")
    ap.add_argument("--n-mc", type=int, default=10_000, help="Monte Carlo draws of H3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    kb_file = args.kb or resolve_path("kb_dir", config=config) / "kanit.sqlite"
    if not kb_file.exists():
        sys.exit(f"ERROR: knowledge base not found: {kb_file}")
    aro_file = args.aro_index or (PROJECT_ROOT / Path(config["card"]["card_json"]).parent /
                                  "aro_index.tsv")
    if not aro_file.exists():
        sys.exit(f"ERROR: CARD's ARO index not found: {aro_file}")
    aro_index = pd.read_csv(aro_file, sep="\t", dtype=str)
    with sqlite3.connect(f"file:{kb_file}?mode=ro", uri=True) as conn:
        kb_version, kb_card = conn.execute(
            "SELECT kb_version, card_version FROM release").fetchone()
        check_index(conn, aro_file, aro_index, kb_card)
        idx = hy.card_family_index(aro_index)
        reference, missing = references(conn, config)
        keywords = registry.load_amrfinder_keywords()
        res = hy.run(conn, reference, idx, keywords, n_mc=args.n_mc, seed=args.seed)
        sens = hy.sensitivity(conn, reference, idx, keywords, n_mc=args.n_mc, seed=args.seed)
    out = resolve_path("cross_model_dir", config=config) / "hypotheses"
    out.mkdir(parents=True, exist_ok=True)
    for name, t in res["tables"].items():
        t.to_csv(out / f"{name}.csv", index=False)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "kb_version": kb_version, "kb_file": str(kb_file), "aro_index": str(aro_file),
               "reference_missing": missing, "n_mc": args.n_mc, "seed": args.seed,
               **res["summary"], "sensitivity_label_permutation": sens}
    (out / "hypotheses.json").write_text(json.dumps(_plain(summary), indent=2) + "\n")
    s = res["summary"]
    print(f"H1 {s['H1']['supported']} · H2 {s['H2']['supported']} (median "
          f"{s['H2']['median']}) · H3 {s['H3']['supported']} (p {s['H3']['p']}) · H7 "
          f"{s['H7'].get('0.1', {}).get('supported_primary')}"
          + (f" · no AMRFinderPlus reference for {missing}" if missing else ""))
    print(f"Without the flagged models: {sens['status']}"
          + (f", {len(sens['excluded_models'])} excluded" if sens["excluded_models"] else ""))


if __name__ == "__main__":
    main()
