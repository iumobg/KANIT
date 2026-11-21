#!/usr/bin/env python3
"""Step 18 — genomic context of an organism's candidate unitigs from NCBI
. Context is not evidence: no grade depends on it.

Subcommands:

  query  searches the unitigs that are not yet in the cache (remote blastn, then
         Entrez for the best hit of each); it needs internet, so it runs on a
         login node, and it can be stopped and run again; a batch that NCBI keeps
         failing is left for a second pass after the others, and the step fails
         only if that pass fails too
  build  the context table from the cache alone (unitig_context.csv)
  all    query and build

Inputs: the CARD layer of every panel model of the organism (09's
card_unitigs.csv: the candidate unitigs and their sequences). The rules are in
lib/ncbi_context.py. Outputs: paths_organism.context_dir (cache/, unitig_context.csv).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import knowledge_base as kb  # noqa: E402
from lib import ncbi_context as nc  # noqa: E402
from lib import panel, registry  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402


def candidate_members(organism: str, config: dict) -> pd.DataFrame:
    """model_id, pattern_id, unitig_id and canonical sequence of every candidate unitig."""
    d = panel.read_panel(resolve_path("panel_dir", config=config) / "panel_decisions.csv")
    parts = []
    for ab in sorted(d.loc[(d["organism"] == organism) & (d["decision"] == panel.INCLUDED),
                           "antibiotic"]):
        f = resolve_path("card_layer_dir", organism=organism, antibiotic=ab,
                         config=config) / "card_unitigs.csv"
        if not f.exists():
            continue                                   # not evaluable: no candidates
        cu = pd.read_csv(f, usecols=["pattern_id", "sequence"])
        ids = [kb.unitig_id(s) for s in cu["sequence"]]
        parts.append(pd.DataFrame({"model_id": f"{organism}__{ab}", "pattern_id": cu["pattern_id"],
                                   "unitig_id": [i for i, _ in ids],
                                   "sequence": [c for _, c in ids]}))
    cols = ["model_id", "pattern_id", "unitig_id", "sequence"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def entrez_query(taxids: list[int]) -> str:
    """nt records of the organism: at or below any of its taxa."""
    q = " OR ".join(f"txid{t}[Organism:exp]" for t in taxids)
    return q if len(taxids) == 1 else f"({q})"


def blast_params(organism: str, cfg: dict) -> dict:
    return {"ENTREZ_QUERY": entrez_query(registry.organism_taxids(organism)),
            "WORD_SIZE": cfg["word_size"],
            "HITLIST_SIZE": cfg["max_target_seqs"], "EXPECT": cfg["evalue"]}


def search(client: nc.Ncbi, cache: Path, batch: pd.DataFrame, params: dict) -> bool:
    """Searches one batch; False when NCBI kept failing, so that it is tried again later."""
    try:
        nc.query_batch(client, cache, batch, params)
    except Exception as e:
        if not (nc.transient(e) or isinstance(e, nc.SearchFailed)):
            raise
        print(f"  ✗ BLAST: {type(e).__name__}: {e}; {len(batch)} unitigs left for later",
              flush=True)
        return False
    return True


def query(organism: str, config: dict, out_dir: Path, client: nc.Ncbi) -> dict:
    cfg = config["context"]
    params = blast_params(organism, cfg)
    todo = nc.select_unitigs(candidate_members(organism, config), cfg["max_members"])
    cache = out_dir / "cache"
    done = set()
    for meta in (cache / "blast").glob("*/meta.json"):
        done |= set(json.loads(meta.read_text())["unitig_ids"])
    pending = todo[~todo["unitig_id"].isin(done)]
    size = cfg["batch_size"]
    left = [pending.iloc[s:s + size] for s in range(0, len(pending), size)]
    n_done = 0
    for _ in range(2):                  # the batches NCBI failed get a second pass
        failed = []
        for batch in left:
            if search(client, cache, batch, params):
                n_done += len(batch)
                print(f"  ✓ BLAST {n_done}/{len(pending)}", flush=True)
            else:
                failed.append(batch)
        left = failed
    n_fetched = 0
    for meta in sorted((cache / "blast").glob("*/meta.json")):
        hits = nc.parse_xml((meta.parent / "report.xml").read_text())
        for acc, a, b in hits.loc[hits["rank"] == 1, ["accession", "hit_start", "hit_stop"]
                                  ].itertuples(index=False):
            f = nc.genbank_file(cache, acc, int(a), int(b))
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(client.genbank(acc, int(a), int(b)))
                n_fetched += 1
    if left:
        sys.exit(f"ERROR: NCBI failed {sum(map(len, left))} unitig(s) of {organism} twice; "
                 f"the others are cached: run the CONTEXT entry again with -resume.")
    return {"n_unitigs": len(todo), "n_searched_now": len(pending), "n_genbank_now": n_fetched}


def build(organism: str, out_dir: Path, config: dict | None = None,
          allow_incomplete: bool = True) -> pd.DataFrame:
    """The context table from the cache; with ``config``, every unitig to be searched
    must be in it unless ``allow_incomplete``."""
    t = nc.build_table(out_dir / "cache", organism)
    if config is not None:
        want = nc.select_unitigs(candidate_members(organism, config),
                                 config["context"]["max_members"])
        missing = sorted(set(want["unitig_id"]) - set(t["unitig_id"]))
        if missing and not allow_incomplete:
            sys.exit(f"ERROR: {len(missing)} candidate unitig(s) of {organism} were not "
                     f"searched yet; run the CONTEXT entry (18 query) first.")
    t.to_csv(out_dir / "unitig_context.csv", index=False)
    hits = t[t["n_hits"] > 0] if "n_hits" in t else t.iloc[0:0]
    (out_dir / "context_summary.json").write_text(json.dumps({
        "organism": organism, "n_unitigs": len(t), "n_with_hits": len(hits),
        "mean_plasmid_share": None if hits.empty else round(float(hits["plasmid_share"].mean()), 4),
        "nt_releases": sorted(t["nt_release"].dropna().astype(str).unique()) if "nt_release" in t else []},
        indent=2) + "\n")
    return t


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="NCBI context of an organism's candidate unitigs.")
    ap.add_argument("command", choices=["query", "build", "all"])
    ap.add_argument("--organism", required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="build although some candidate unitigs were not searched")
    args = ap.parse_args()
    out_dir = resolve_path("context_dir", organism=args.organism, config=config)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"NCBI CONTEXT — {args.organism} — {args.command}")
    if args.command in ("query", "all"):
        n = config["ncbi"]
        client = nc.Ncbi(n['entrez_email'], n.get('api_key') or '',
                         off_hours_only=config["context"]["off_hours_only"])
        print(f"  {query(args.organism, config, out_dir, client)}")
    if args.command in ("build", "all"):
        t = build(args.organism, out_dir, config, allow_incomplete=args.allow_incomplete)
        print(f"  {len(t)} unitigs, {int((t['n_hits'] > 0).sum())} with a hit")


if __name__ == "__main__":
    main()
