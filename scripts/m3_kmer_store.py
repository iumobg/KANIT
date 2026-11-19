#!/usr/bin/env python3
"""Unitig–k-mer comparison, arm C — the organism's k-mer store (lib/kmers.py).

Reads the unitig sequences of the full presence table (unitig_store_dir/call/unitigs.rtab),
counts in the assemblies the genomes carrying each of their canonical k-mers and writes
the k-mers passing the frequency filter to kmer_store_dir, in the unitig store's format.

Run with arm C's overlay (config/m3_C.yml), where unitig_store_dir names this store for the
steps after it and source_unitig_store_dir the unitig store it is built from.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import kmers  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402
from lib.run_metadata import peak_rss_gb  # noqa: E402


def unitig_store(organism: str, config: dict) -> Path:
    """The unitig store the k-mer store is built from."""
    key = ("source_unitig_store_dir" if "source_unitig_store_dir" in config["paths_organism"]
           else "unitig_store_dir")
    return resolve_path(key, organism=organism, config=config)


def main():
    config = load_config()
    default_org, _ = get_target(config=config)
    ucfg = config["unitig"]
    ap = argparse.ArgumentParser(description="Build an organism's k-mer store.")
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--workers", type=int, default=int(config["preprocessing"]["threads"]))
    ap.add_argument("--k", type=int, default=int(ucfg["k"]))
    ap.add_argument("--min-support", type=int, default=int(ucfg["min_support"]))
    ap.add_argument("--rtab", type=Path, help="presence table whose unitigs give the k-mers "
                    "(default: <unitig store>/call/unitigs.rtab)")
    args = ap.parse_args()

    src = unitig_store(args.organism, config)
    out = resolve_path("kmer_store_dir", organism=args.organism, config=config)
    if out.resolve() == src.resolve():
        sys.exit(f"ERROR: the k-mer store would overwrite the unitig store ({src})")
    import pandas as pd
    genomes = pd.read_csv(src / "genomes.csv", dtype=str)["Genome ID"].tolist()
    gdir = resolve_path("raw_genomes_dir", organism=args.organism, config=config)
    print(f"K-MER STORE — {args.organism}: {len(genomes):,} genomes, k={args.k}, "
          f"min_support={args.min_support}, {args.workers} workers")
    s = kmers.build_store(src, [gdir / f"{g}.fna" for g in genomes], out, k=args.k,
                          min_support=args.min_support, workers=args.workers, rtab=args.rtab)
    print(f"  {s['n_unitigs_read']:,} unitigs → {s['n_kmers_graph']:,} k-mers → "
          f"{s['n_unitigs']:,} kept; {s['seconds']}; peak {peak_rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
