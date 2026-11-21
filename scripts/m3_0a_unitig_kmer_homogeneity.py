#!/usr/bin/env python3
"""M3-0a — do all 31-mers of a unitig carry the same presence/absence vector?

This is the pivot of the unitig–k-mer comparison's design. Our frequency filter is defined on the 31-mer; the unitig arm applies it
to whole unitigs. The two are the same filter only if a unitig is homogeneous, i.e.
every k-mer inside it appears in exactly the genomes the unitig appears in.

The answer is not in the documentation. unitig-caller's README says call mode "builds
a Bifrost DBG and calls the colours for each unitig within" without saying how the
colours of a unitig's k-mers are combined, and in Bifrost colours attach to k-mers,
not to unitigs. So we measure it on our own data.

What this does:

  * reservoir-samples unitigs of at least --min-length from the organism store,
  * takes up to --kmers-per-unitig of each one's 31-mers and canonicalises them,
  * samples --n-genomes assemblies and enumerates their canonical 31-mers,
  * for each sampled unitig, compares its k-mers' measured presence vectors with
    each other (internal homogeneity) and with the store's own row (store fidelity).

Reading the result:

  homogeneous and faithful  -> the filter is literally the same in both arms, and
                               the collapse trap is real: the k-mer arm must not
                               collapse, or the comparison is null by construction.
  inhomogeneous             -> raw 31-mers carry strictly more information than
                               unitigs, the trap disappears, and the k-mer arm's
                               feature count has to be measured, not derived (M3-0b).

⚠️ Homogeneity over a genome subset is necessary, not sufficient: a unitig called
homogeneous here may still split over genomes that were not sampled. Raise
--n-genomes to tighten the bound.

Outputs (default results/tables/): m3_0a_homogeneity_<organism>.csv,
m3_0a_summary_<organism>.json

Usage:
    python scripts/m3_0a_unitig_kmer_homogeneity.py --organism ecoli
    python scripts/m3_0a_unitig_kmer_homogeneity.py --organism ecoli \
        --n-unitigs 5000 --n-genomes 600 --threads 16
"""
import argparse
import gzip
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lib import matrix_store, panel  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402

# base -> 2 bits; everything else (N, IUPAC codes) maps to 4 and breaks the run
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i
    _LUT[_b + 32] = _i          # lowercase (soft-masked assemblies)

_CTX: dict[str, Any] = {}          # sorted query codes and k, set per worker


# ---------------------------------------------------------------------------
# k-mer encoding
# ---------------------------------------------------------------------------
def canonical_codes(v: np.ndarray, k: int) -> np.ndarray:
    """Canonical code of every k-mer of a run of 2-bit bases.

    The code packs the k bases two bits each, most significant first; the canonical
    form is the smaller of the k-mer and its reverse complement, so a genome and its
    reverse strand give the same code. k <= 32 keeps this inside a uint64.
    """
    n = v.size - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    v64 = v.astype(np.uint64)
    two, three = np.uint64(2), np.uint64(3)
    fwd = np.zeros(n, dtype=np.uint64)
    rev = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        fwd <<= two
        fwd |= v64[j:j + n]
        rev <<= two
        rev |= three - v64[k - 1 - j:k - 1 - j + n]
    return np.minimum(fwd, rev)


def _acgt_runs(v: np.ndarray, k: int):
    """Maximal runs of ACGT long enough to hold a k-mer."""
    bad = np.flatnonzero(v > 3)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), bad + 1))
    ends = np.concatenate((bad, np.asarray([v.size], dtype=np.int64)))
    for s, e in zip(starts, ends, strict=True):
        if e - s >= k:
            yield v[s:e]


def _contigs(path: Path):
    """Sequence of each FASTA record, as bytes. k-mers never span records."""
    opener = gzip.open if path.suffix == ".gz" else open
    parts: list[bytes] = []
    with opener(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if parts:
                    yield b"".join(parts)
                    parts = []
            else:
                parts.append(line.strip())
    if parts:
        yield b"".join(parts)


# ---------------------------------------------------------------------------
# per-genome scan (worker)
# ---------------------------------------------------------------------------
def _init_worker(query_sorted: np.ndarray, k: int) -> None:
    """Hand the workers the query set. Set through an initializer rather than a module
    global so it survives the spawn start method as well as fork."""
    _CTX["q"], _CTX["k"] = query_sorted, k


def _scan(task: tuple[int, str]) -> tuple[int, np.ndarray]:
    """Which query codes occur in one assembly."""
    slot, path = task
    q, k = _CTX["q"], _CTX["k"]
    hit = np.zeros(q.size, dtype=bool)
    for seq in _contigs(Path(path)):
        v = _LUT[np.frombuffer(seq, dtype=np.uint8)]
        for run in _acgt_runs(v, k):
            codes = canonical_codes(run, k)
            if codes.size == 0:
                continue
            pos = np.searchsorted(q, codes)
            np.clip(pos, 0, q.size - 1, out=pos)
            hit[pos[q[pos] == codes]] = True
    return slot, hit


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_unitigs(store_dir: Path, *, n_wanted: int, min_length: int,
                   rng: np.random.Generator) -> list[tuple[int, str]]:
    """Reservoir sample of (store index, sequence) over unitigs >= min_length.

    One streaming pass, so the whole sequences.txt is never held in memory.
    """
    reservoir: list[tuple[int, str]] = []
    seen = 0
    with open(store_dir / "sequences.txt", "rb") as f:
        for i, line in enumerate(f):
            seq = line.strip()
            if len(seq) < min_length:
                continue
            seen += 1
            if len(reservoir) < n_wanted:
                reservoir.append((i, seq.decode("ascii")))
            else:
                j = int(rng.integers(seen))
                if j < n_wanted:
                    reservoir[j] = (i, seq.decode("ascii"))
    print(f"  {seen:,} unitig(s) of >= {min_length} bp; sampled {len(reservoir):,}")
    return sorted(reservoir)


def build_queries(sampled, k: int, per_unitig: int):
    """Canonical codes of the k-mers we will look for, with their unitig and offset."""
    codes, owners, offsets, skipped = [], [], [], []
    for slot, (idx, seq) in enumerate(sampled):
        v = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        if (v > 3).any():                     # ambiguous base: cannot encode cleanly
            skipped.append(idx)
            continue
        all_codes = canonical_codes(v, k)
        if all_codes.size == 0:
            skipped.append(idx)
            continue
        if all_codes.size <= per_unitig:
            take = np.arange(all_codes.size)
        else:                                 # evenly spaced, so both ends are tested
            take = np.unique(np.linspace(0, all_codes.size - 1, per_unitig).astype(np.int64))
        codes.append(all_codes[take])
        owners.append(np.full(take.size, slot, dtype=np.int64))
        offsets.append(take)
    if not codes:
        sys.exit("ERROR: no sampled unitig could be encoded.")
    return (np.concatenate(codes), np.concatenate(owners),
            np.concatenate(offsets), skipped)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def compare(hits, owners, offsets, sampled, store, genome_slots, n_slots):
    """Per sampled unitig: internal agreement of its k-mers, and the store's row."""
    rows = []
    presence = store.presence
    n_genomes = len(store.genomes)
    for slot, (idx, seq) in enumerate(sampled):
        mine = np.flatnonzero(owners == slot)
        if mine.size == 0:
            continue
        block = hits[mine]                                  # k-mers x sampled genomes
        uniq = np.unique(block, axis=0)
        # worst disagreement between two k-mers of this unitig, in genomes
        spread = 0
        if uniq.shape[0] > 1:
            spread = int(max((uniq[a] != uniq[b]).sum()
                             for a in range(uniq.shape[0])
                             for b in range(a + 1, uniq.shape[0])))
        store_row = np.unpackbits(np.asarray(presence[idx]))[:n_genomes][genome_slots]
        store_row = store_row.astype(bool)
        # the k-mer vector the unitig would get if we took the majority call
        modal = block.mean(axis=0) >= 0.5
        rows.append({
            "unitig_index": int(idx),
            "length": len(seq),
            "n_kmers_tested": int(mine.size),
            "first_offset": int(offsets[mine].min()),
            "last_offset": int(offsets[mine].max()),
            "n_distinct_kmer_patterns": int(uniq.shape[0]),
            "homogeneous": bool(uniq.shape[0] == 1),
            "max_pairwise_hamming": spread,
            "store_vs_modal_hamming": int((store_row != modal).sum()),
            "faithful": bool(uniq.shape[0] == 1 and np.array_equal(store_row, block[0])),
            "store_n_present_sampled": int(store_row.sum()),
            "kmer_n_present_sampled": int(modal.sum()),
            "n_genomes_sampled": int(n_slots),
        })
    return pd.DataFrame(rows)


def measure(organism: str, args, config, k: int) -> dict | None:
    """One organism, start to finish. None when there is nothing to measure yet."""
    store_dir = resolve_path("unitig_store_dir", organism=organism, config=config)
    if not (store_dir / "sequences.txt").exists():
        print(f"  ! no unitig store at {store_dir}; skipped")
        return None
    genomes_dir = resolve_path("raw_genomes_dir", organism=organism, config=config)

    rng = np.random.default_rng(args.seed)
    store = matrix_store.Store(store_dir)
    print(f"  store: {store.n_unitigs:,} unitigs x {len(store.genomes):,} genomes, k={k}")

    sampled = sample_unitigs(store_dir, n_wanted=args.n_unitigs,
                             min_length=max(args.min_length, k), rng=rng)
    codes, owners, offsets, skipped = build_queries(sampled, k, args.kmers_per_unitig)

    # a canonical k-mer belongs to one node of a de Bruijn graph, so duplicates here
    # would mean the store is not what we assume; keep the first and report them
    order = np.argsort(codes, kind="stable")
    uniq_codes, first = np.unique(codes[order], return_index=True)
    n_duplicate = int(codes.size - uniq_codes.size)
    keep = order[first]
    query_sorted = codes[keep]
    owners, offsets = owners[keep], offsets[keep]
    print(f"  {query_sorted.size:,} distinct query k-mer(s)"
          + (f"; {n_duplicate:,} duplicate(s) dropped" if n_duplicate else ""))

    # genomes: only those whose assembly is on disk
    store_slot = {gid: i for i, gid in enumerate(store.genomes)}
    pool_ids = [store.genomes[i] for i in rng.permutation(len(store.genomes))]
    chosen, missing = [], 0
    for gid in pool_ids:
        if len(chosen) >= args.n_genomes:
            break
        path = genomes_dir / f"{gid}.fna"
        if not path.exists():
            path = genomes_dir / f"{gid}.fna.gz"
        if path.exists():
            chosen.append((gid, path))
        else:
            missing += 1
    if not chosen:
        print(f"  ! no assembly found under {genomes_dir}; skipped")
        return None
    if missing:
        print(f"  ! {missing} assembly file(s) missing, skipped")
    genome_slots = np.asarray([store_slot[gid] for gid, _ in chosen], dtype=np.int64)
    print(f"  scanning {len(chosen):,} assemblies on {args.threads} thread(s) ...", flush=True)

    hits = np.zeros((query_sorted.size, len(chosen)), dtype=bool)
    tasks = [(slot, str(path)) for slot, (_, path) in enumerate(chosen)]
    if args.threads > 1:
        with Pool(args.threads, initializer=_init_worker,
                  initargs=(query_sorted, k)) as pool:
            for done, (slot, hit) in enumerate(pool.imap_unordered(_scan, tasks), 1):
                hits[:, slot] = hit
                if done % 25 == 0:
                    print(f"    {done}/{len(tasks)}", flush=True)
    else:
        _init_worker(query_sorted, k)
        for done, task in enumerate(tasks, 1):
            slot, hit = _scan(task)
            hits[:, slot] = hit
            if done % 25 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)

    table = compare(hits, owners, offsets, sampled, store, genome_slots, len(chosen))
    if table.empty:
        print("  ! no sampled unitig could be compared; skipped")
        return None
    args.out.mkdir(parents=True, exist_ok=True)
    # one file per organism: the job script measures every organism of the panel
    csv_path = args.out / f"m3_0a_homogeneity_{organism}.csv"
    table.to_csv(csv_path, index=False)

    n = len(table)
    n_hom = int(table["homogeneous"].sum())
    n_faithful = int(table["faithful"].sum())
    summary = {
        "organism": organism,
        "k": k,
        "seed": args.seed,
        "n_unitigs_tested": n,
        "n_unitigs_skipped": len(skipped),
        "n_kmers_tested": int(table["n_kmers_tested"].sum()),
        "n_duplicate_query_kmers": n_duplicate,
        "n_genomes_sampled": len(chosen),
        "n_genomes_in_store": len(store.genomes),
        "homogeneous_frac": n_hom / n if n else float("nan"),
        "faithful_frac": n_faithful / n if n else float("nan"),
        "max_pairwise_hamming": {
            "max": int(table["max_pairwise_hamming"].max()) if n else 0,
            "median_of_inhomogeneous": (
                float(table.loc[~table["homogeneous"], "max_pairwise_hamming"].median())
                if n_hom < n else 0.0),
        },
        "store_vs_modal_hamming_max": int(table["store_vs_modal_hamming"].max()) if n else 0,
        "caveat": "homogeneity is measured over the sampled genomes only",
    }
    (args.out / f"m3_0a_summary_{organism}.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print(f"\n  {n:,} unitig(s) tested, {summary['n_kmers_tested']:,} k-mer(s), "
          f"{len(chosen):,} of {len(store.genomes):,} genomes")
    print(f"  internally homogeneous: {n_hom:,}/{n:,} ({100 * n_hom / n:.1f}%)")
    print(f"  equal to the store row: {n_faithful:,}/{n:,} ({100 * n_faithful / n:.1f}%)")
    if n_hom < n:
        print("  -> unitigs are NOT homogeneous: raw 31-mers carry more information "
              "than unitigs, and the collapse trap does not apply")
    else:
        print("  -> unitigs are homogeneous over the sampled genomes: the k-mer arm "
              "must not pattern-collapse")
    print(f"  -> {csv_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--organism", default="all",
                    help="organism slug, or 'all' for every organism of the panel")
    ap.add_argument("--n-unitigs", type=int, default=2000)
    ap.add_argument("--n-genomes", type=int, default=300)
    ap.add_argument("--min-length", type=int, default=100,
                    help="shortest unitig to sample; 100 bp holds 70 31-mers")
    ap.add_argument("--kmers-per-unitig", type=int, default=40)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()

    config = load_config()
    k = int(config["unitig"]["k"])
    if k > 32:
        sys.exit(f"ERROR: k={k} does not fit a uint64 code (max 32).")

    if args.organism == "all":
        panel_csv = resolve_path("panel_dir", config=config) / "panel_decisions.csv"
        if not panel_csv.exists():
            sys.exit(f"ERROR: no panel at {panel_csv}; name an organism instead.")
        organisms = sorted({org for org, _ in panel.included_pairs(panel_csv)})
        if not organisms:
            sys.exit("ERROR: the panel includes no pair.")
    else:
        organisms = [args.organism]

    done = []
    for org in organisms:
        print(f"\n===== {org} =====")
        summary = measure(org, args, config, k)
        if summary is not None:
            done.append(summary)
    if not done:
        sys.exit("ERROR: no organism could be measured.")

    if len(done) > 1:
        print("\n===== all organisms =====")
        for s in done:
            print(f"  {s['organism']:24} homogeneous {s['homogeneous_frac']:6.1%}  "
                  f"faithful {s['faithful_frac']:6.1%}  "
                  f"({s['n_unitigs_tested']:,} unitigs, {s['n_genomes_sampled']:,} genomes)")


if __name__ == "__main__":
    main()
