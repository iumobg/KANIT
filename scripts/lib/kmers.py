"""Canonical k-mers and the organism k-mer store (unitig–k-mer comparison, arm C).

The k-mer store has the format of the unitig store (lib/matrix_store.py), so that every
step reading a store reads it unchanged; a "unitig" of this store is one canonical k-mer:

    genomes.csv          Genome ID, in bit order (the unitig store's order)
    sequences.txt        line i = canonical k-mer i (codes ascending)
    presence.bin         uint8, shape (n_kmers, ceil(n_genomes / 8))
    store_summary.json   shapes, filter counts, source, timings

The k-mers are those of the organism's compacted de Bruijn graph: every k-mer of every
genome sits in exactly one unitig, so the unitig sequences of the full presence table
hold the organism's k-mers. Their presence is counted from the assemblies, not taken
from the unitigs: a unitig's k-mers do not always share its presence (M3-0a).

Building takes two passes over the assemblies: the first counts the genomes carrying
each k-mer, the second writes the presence of the k-mers passing the frequency filter
(present in at least ``min_support`` genomes and absent from at least as many).
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib import matrix_store

# base -> 2 bits; everything else (N, IUPAC codes, separators) maps to 4 and breaks k-mers
LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    LUT[_b] = _i
    LUT[_b + 32] = _i            # lowercase (soft-masked assemblies)
_ACGT = np.frombuffer(b"ACGT", dtype=np.uint8)
_SEP = b"N"

_CTX: dict[str, Any] = {}        # per worker: the sorted codes searched and k


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def codes(seq: bytes, k: int) -> np.ndarray:
    """Canonical code of every k-mer of ``seq`` made only of A, C, G and T, in order.

    The code packs the k bases two bits each, most significant first; the canonical form
    is the smaller of the k-mer and its reverse complement. Any other byte breaks the
    sequence, so records joined with ``N`` give no k-mer across the join. k <= 32.
    """
    if not 1 <= k <= 32:
        raise ValueError("k must lie in [1, 32]")
    v = LUT[np.frombuffer(seq, dtype=np.uint8)]
    n = v.size - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    bad = v > 3
    cum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(bad, dtype=np.int64)))
    ok = cum[k:] == cum[:n]                  # window i = v[i:i+k] holds no bad base
    if not ok.any():
        return np.empty(0, dtype=np.uint64)
    v64 = np.where(bad, 0, v).astype(np.uint64)
    two, three = np.uint64(2), np.uint64(3)
    fwd = np.zeros(n, dtype=np.uint64)
    rev = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        fwd <<= two
        fwd |= v64[j:j + n]
        rev <<= two
        rev |= three - v64[k - 1 - j:k - 1 - j + n]
    return np.minimum(fwd, rev)[ok]


def decode(c: np.ndarray, k: int) -> np.ndarray:
    """k-mer sequences of codes, as a (len(c), k) array of ASCII bytes."""
    c = np.asarray(c, dtype=np.uint64)
    shifts = (2 * (k - 1 - np.arange(k))).astype(np.uint64)
    return _ACGT[((c[:, None] >> shifts) & np.uint64(3)).astype(np.intp)]


def records(path: Path) -> Iterator[bytes]:
    """Sequence of each FASTA record, as bytes."""
    opener: Callable[..., Any] = gzip.open if Path(path).suffix == ".gz" else open
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


def genome_codes(path: Path, k: int) -> np.ndarray:
    """Distinct canonical k-mer codes of one assembly, ascending."""
    return np.unique(codes(_SEP.join(records(path)), k))


def unitig_codes(rtab: Path, k: int, block_bytes: int = 1 << 28) -> tuple[np.ndarray, int]:
    """Distinct canonical k-mer codes of the unitigs of a presence table (its first
    column), ascending, and the number of unitigs read."""
    parts: list[np.ndarray] = []
    n_unitigs, first = 0, True
    for lines in matrix_store._iter_lines(Path(rtab), block_bytes):
        if first:                            # the header line
            lines, first = lines[1:], False
        seqs = [ln[:ln.find(b"\t")] for ln in lines if ln.strip()]
        n_unitigs += len(seqs)
        if seqs:
            parts.append(np.unique(codes(_SEP.join(seqs), k)))
    out = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
    return out, n_unitigs


# ---------------------------------------------------------------------------
# per-genome work (pool workers)
# ---------------------------------------------------------------------------
def _init(sorted_codes: np.ndarray, k: int) -> None:
    _CTX["q"], _CTX["k"] = sorted_codes, k


def _hits(path: str) -> np.ndarray:
    """Indices into the sorted codes of the k-mers one assembly carries (unique)."""
    q, k = _CTX["q"], _CTX["k"]
    c = genome_codes(Path(path), k)
    if c.size == 0 or q.size == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(q, c)
    np.clip(pos, 0, q.size - 1, out=pos)
    return pos[q[pos] == c]


def _row(path: str) -> bytes:
    """Packed presence row of one assembly over the sorted codes."""
    q = _CTX["q"]
    row = np.zeros(matrix_store._nbytes(q.size) * 8, dtype=bool)
    row[_hits(path)] = True
    return np.packbits(row).tobytes()


def _map(fn: Callable[[str], Any], paths: list[str], sorted_codes: np.ndarray, k: int,
         workers: int) -> Iterable[Any]:
    """``fn`` over the assemblies, in order; workers share the codes through fork."""
    if workers <= 1:
        _init(sorted_codes, k)
        yield from map(fn, paths)
        return
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else None)
    with ctx.Pool(workers, initializer=_init, initargs=(sorted_codes, k)) as pool:
        yield from pool.imap(fn, paths, chunksize=1)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def build_store(unitig_store_dir: Path, genome_paths: list[Path], out_dir: Path, *, k: int,
                min_support: int, workers: int = 1, rtab: Path | None = None,
                block_bytes: int = 1 << 14) -> dict:
    """Build the k-mer store of an organism next to its unitig store.

    ``genome_paths`` are the assemblies in the unitig store's genome order. The presence
    is first written genome by genome (``presence_by_genome.bin``) and then transposed
    ``block_bytes`` bytes of k-mers at a time into ``presence.bin``.
    """
    unitig_store_dir, out_dir = Path(unitig_store_dir), Path(out_dir)
    t: dict[str, float] = {}
    t0 = time.time()
    usum = json.loads((unitig_store_dir / "store_summary.json").read_text())
    genomes = pd.read_csv(unitig_store_dir / "genomes.csv", dtype=str)["Genome ID"].tolist()
    n = len(genomes)
    if len(genome_paths) != n:
        raise ValueError(f"{len(genome_paths)} assemblies for {n} genomes of the unitig store")
    missing = [str(p) for p in genome_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} assemblies missing, e.g. {missing[:3]}")
    if n >= np.iinfo(np.uint16).max:
        raise ValueError("more genomes than a uint16 count holds")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in genome_paths]

    rtab = Path(rtab) if rtab else unitig_store_dir / "call" / "unitigs.rtab"
    universe, n_unitigs = unitig_codes(rtab, k)
    t["universe"] = time.time() - t0

    t1 = time.time()
    counts = np.zeros(universe.size, dtype=np.uint16)
    for idx in _map(_hits, paths, universe, k, workers):
        counts[idx] += 1                     # idx holds each k-mer once
    t["count"] = time.time() - t1
    keep = (counts >= min_support) & (n - counts.astype(np.int64) >= min_support)
    kept = universe[keep]
    n_absent = int((counts == 0).sum())
    del universe, counts

    t1 = time.time()
    width = matrix_store._nbytes(kept.size)
    tmp = out_dir / "presence_by_genome.bin"
    by_genome = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n, max(width, 1)))
    for g, row in enumerate(_map(_row, paths, kept, k, workers)):
        by_genome[g, :width] = np.frombuffer(row, dtype=np.uint8)
    by_genome.flush()
    t["presence"] = time.time() - t1

    t1 = time.time()
    with open(out_dir / "presence.bin", "wb") as pres:
        for b0 in range(0, width, block_bytes):
            b1 = min(b0 + block_bytes, width)
            bits = np.unpackbits(np.asarray(by_genome[:, b0:b1]), axis=1)    # (n, 8w)
            n_rows = min(8 * (b1 - b0), kept.size - 8 * b0)
            pres.write(np.packbits(bits[:, :n_rows].T, axis=1).tobytes())
    del by_genome
    tmp.unlink()
    with open(out_dir / "sequences.txt", "wb") as seqs:
        for s in range(0, kept.size, 1 << 20):
            block = decode(kept[s:s + (1 << 20)], k)
            nl = np.full((block.shape[0], 1), ord("\n"), dtype=np.uint8)
            seqs.write(np.hstack([block, nl]).tobytes())
    pd.DataFrame({"Genome ID": genomes}).to_csv(out_dir / "genomes.csv", index=False)
    t["write"] = time.time() - t1

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature": "kmer", "k": k,
        "unitig_store_dir": str(unitig_store_dir), "source_rtab": str(rtab),
        "source_sha256": usum.get("source_sha256"),
        "n_genomes": n, "n_unitigs_read": n_unitigs,
        "n_kmers_graph": int(keep.size), "n_kmers_in_no_genome": n_absent,
        # the unitig store's keys, so that matrix_store.Store reads this store
        "n_unitigs_seen": int(keep.size), "n_unitigs": int(kept.size),
        "min_support": min_support, "presence_shape": [int(kept.size), matrix_store._nbytes(n)],
        "seconds": {key: round(v, 1) for key, v in t.items()},
        "workers": workers, "host_cpus": os.cpu_count(),
    }
    (out_dir / "store_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
