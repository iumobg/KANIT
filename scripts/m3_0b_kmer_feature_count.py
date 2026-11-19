#!/usr/bin/env python3
"""M3-0b — how wide and how heavy would the k-mer arm be?

The unitig–k-mer comparison needs the k-mer arm's
dimensionality before any code is written. It is already determined by the finished
unitig run: in a compacted de Bruijn graph every canonical k-mer sits in exactly one
unitig, so the number of distinct 31-mers in a set of unitigs is

    sum over unitigs of (length - k + 1)

and, if every k-mer of a unitig carries the unitig's presence vector, the number of
non-zero entries of the uncollapsed k-mer matrix is

    sum over unitigs of (length - k + 1) * (genomes carrying that unitig)

That second number is what decides whether a k-mer model fits a node at all. Both are
read off the store and the pair matrices; nothing is recomputed.

⚠️ The per-k-mer presence assumption is exactly what M3-0a measures. If it fails, a
k-mer inside a kept unitig can fail the frequency filter and a k-mer inside a dropped
one can pass it, so these counts are estimates in both directions, not bounds.

Outputs (default results/tables/): m3_0b_kmer_feature_count.csv, m3_0b_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lib import panel  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402

# CSR cost XGBoost pays per stored entry: float32 value + uint32 column index.
BYTES_PER_NNZ = 8


def unitig_lengths(store_dir: Path, block: int = 1 << 24) -> np.ndarray:
    """Length of every unitig in the store, in store index order.

    Counted from the newline positions of each block rather than by iterating over
    the lines: a large store holds tens of millions of unitigs, and a Python list
    that long costs more than the int64 array it ends up as.
    """
    parts: list[np.ndarray] = []
    carry = 0                                    # bytes of the line still open
    with open(store_dir / "sequences.txt", "rb") as f:
        while chunk := f.read(block):
            a = np.frombuffer(chunk, dtype=np.uint8)
            nl = np.flatnonzero(a == 0x0A)
            if nl.size == 0:
                carry += a.size
                continue
            parts.append(np.asarray([carry + int(nl[0])], dtype=np.int64))
            if nl.size > 1:
                parts.append((np.diff(nl) - 1).astype(np.int64))
            carry = a.size - int(nl[-1]) - 1
    if carry:                                    # a last line without a newline
        parts.append(np.asarray([carry], dtype=np.int64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def pair_counts(matrix_dir: Path, lengths: np.ndarray, k: int) -> dict:
    """k-mer width and nnz of one pair, next to the unitig arm's own figures."""
    summary = json.loads((matrix_dir / "matrix_summary.json").read_text())
    members = np.load(matrix_dir / "members.npy")           # (n_kept, 2)
    patterns = pd.read_csv(matrix_dir / "patterns.csv")
    n_present = patterns["n_present"].to_numpy(dtype=np.int64)

    unitig_idx, pattern_id = members[:, 0], members[:, 1]
    if unitig_idx.size and unitig_idx.max() >= lengths.size:
        raise IndexError(f"{matrix_dir}: members.npy points past the store "
                         f"({unitig_idx.max()} >= {lengths.size} unitigs); "
                         "store and matrix are from different runs")
    # a unitig of length < k holds no k-mer; unitig-caller should emit none
    kmers = np.clip(lengths[unitig_idx] - k + 1, 0, None)
    carriers = n_present[pattern_id]

    return {
        "n_genomes": int(summary["n_genomes"]),
        "min_support": int(summary["min_support"]),
        "n_unitigs_kept": int(unitig_idx.size),
        "n_patterns": int(summary["n_patterns"]),
        "n_kmer_features": int(kmers.sum()),
        "nnz_kmer": int((kmers * carriers).sum()),
        "nnz_unitig": int(n_present.sum()),
        "n_unitigs_below_k": int((lengths[unitig_idx] < k).sum()),
        "median_unitig_length": float(np.median(lengths[unitig_idx])) if unitig_idx.size else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--organism", default=None, help="only this organism (default: all)")
    args = ap.parse_args()

    config = load_config()
    k = int(config["unitig"]["k"])
    panel_csv = resolve_path("panel_dir", config=config) / "panel_decisions.csv"
    if not panel_csv.exists():
        sys.exit(f"ERROR: no panel at {panel_csv}.")
    pairs = [(o, a) for o, a in panel.included_pairs(panel_csv)
             if args.organism in (None, o)]
    if not pairs:
        sys.exit("ERROR: the panel includes no matching pair.")

    rows, store_level = [], []
    for org in sorted({o for o, _ in pairs}):
        store_dir = resolve_path("unitig_store_dir", organism=org, config=config)
        if not (store_dir / "sequences.txt").exists():
            print(f"  ! no store for {org} ({store_dir})")
            continue
        print(f"  reading {org} store ...", flush=True)
        lengths = unitig_lengths(store_dir)
        store_level.append({
            "organism": org,
            "n_unitigs_store": int(lengths.size),
            # every canonical k-mer of the organism that passed the store filter
            "n_kmers_store": int(np.clip(lengths - k + 1, 0, None).sum()),
            "median_unitig_length": float(np.median(lengths)) if lengths.size else 0.0,
        })
        for o, ab in [(o, a) for o, a in pairs if o == org]:
            matrix_dir = resolve_path("matrix_dir", organism=o, antibiotic=ab, config=config)
            if not (matrix_dir / "matrix_summary.json").exists():
                print(f"  ! no matrix for {o}/{ab}")
                continue
            rows.append({"organism": o, "antibiotic": ab, **pair_counts(matrix_dir, lengths, k)})

    if not rows:
        sys.exit("ERROR: no pair measured; has the unitig run written the matrices?")

    table = pd.DataFrame(rows)
    table["kmer_per_pattern"] = table["n_kmer_features"] / table["n_patterns"]
    table["nnz_ratio"] = table["nnz_kmer"] / table["nnz_unitig"]
    table["kmer_csr_gib"] = table["nnz_kmer"] * BYTES_PER_NNZ / 2**30
    table["unitig_csr_gib"] = table["nnz_unitig"] * BYTES_PER_NNZ / 2**30

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "m3_0b_kmer_feature_count.csv", index=False)
    heaviest = table.loc[table["nnz_kmer"].idxmax()]
    summary = {
        "k": k,
        "bytes_per_nnz_assumed": BYTES_PER_NNZ,
        "n_pairs_measured": int(len(table)),
        "store_level": store_level,
        "kmer_per_pattern": {
            "min": float(table["kmer_per_pattern"].min()),
            "median": float(table["kmer_per_pattern"].median()),
            "max": float(table["kmer_per_pattern"].max()),
        },
        "nnz_ratio": {
            "min": float(table["nnz_ratio"].min()),
            "median": float(table["nnz_ratio"].median()),
            "max": float(table["nnz_ratio"].max()),
        },
        "heaviest_pair": {
            "organism": heaviest["organism"], "antibiotic": heaviest["antibiotic"],
            "n_kmer_features": int(heaviest["n_kmer_features"]),
            "nnz_kmer": int(heaviest["nnz_kmer"]),
            "kmer_csr_gib": float(heaviest["kmer_csr_gib"]),
            "unitig_csr_gib": float(heaviest["unitig_csr_gib"]),
        },
        "assumes": "every k-mer of a unitig carries the unitig's presence vector (M3-0a)",
    }
    (args.out / "m3_0b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{len(table)} pair(s), k={k}")
    print(f"  k-mer features per pattern: median {table['kmer_per_pattern'].median():.1f}x")
    print(f"  nnz k-mer / nnz unitig:     median {table['nnz_ratio'].median():.1f}x")
    print(f"  heaviest: {heaviest['organism']}/{heaviest['antibiotic']} "
          f"{heaviest['n_kmer_features']:,} k-mer features, "
          f"{heaviest['kmer_csr_gib']:,.1f} GiB CSR "
          f"(unitig arm {heaviest['unitig_csr_gib']:,.1f} GiB)")
    print(f"  -> {args.out / 'm3_0b_kmer_feature_count.csv'}")


if __name__ == "__main__":
    main()
