"""Feature budget of the unitig–k-mer comparison (arms B and C).

A model of arm B or C sees only the ``K`` features with the largest χ² against the label,
chosen inside each of its training sets: every outer fold's training genomes, and all
genomes for the final model. Choosing on all genomes for an outer fold would let the
test fold pick the features.

Features come from a *source*: the patterns of arm A's model matrix (arm B), or the
canonical k-mers of the organism's k-mer store that pass the model's frequency filter
(arm C; k-mers are not collapsed). χ² and its tie rule (smaller feature id first) are
those of the CPSS prefilter (lib/cpss.py), so arm B's final selection is that prefilter.

A budgeted model writes, in the format of lib/matrix_store.py:

    <matrix_dir>/                 the final model's K features
    <matrix_dir>/cv_union/        the features chosen by any outer fold, and
        unit_columns.npz          per outer fold, its columns in cv_union
    <matrix_dir>/selection.csv.gz every training set's choice with its χ²

``SubsetMatrix`` shows an outer fold its own columns of cv_union as a model matrix.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib import folds, matrix_store
from lib.matrix_store import ModelMatrix, Store

FINAL = "final"
UNION = "cv_union"
UNIT_COLUMNS = "unit_columns.npz"


# ---------------------------------------------------------------------------
# training sets
# ---------------------------------------------------------------------------
def training_sets(folds_csv, seeds_csv, ids) -> list[tuple[str, np.ndarray]]:
    """(name, training mask over ``ids``) of every evaluable outer fold, then the final
    model's (all genomes). Names are those of 04_nested_cv's units."""
    ids = np.asarray(ids, dtype=str)
    ft = pd.read_csv(folds_csv, dtype={"genome_id": str})
    seeds = pd.read_csv(seeds_csv)
    out: list[tuple[str, np.ndarray]] = []
    for s in seeds.itertuples():
        if not s.evaluable:
            continue
        part = ft[(ft["arm"] == s.arm) & (ft["repeat"] == s.repeat)]
        fold_of = pd.Series(part["fold"].to_numpy(), index=part["genome_id"]).reindex(ids)
        if fold_of.isna().any():
            raise ValueError(f"folds of {s.arm} repeat {s.repeat} do not cover the genomes")
        f = fold_of.to_numpy(dtype=int)
        out += [(folds.unit_name(str(s.arm), int(s.repeat), int(k)), f != k)
                for k in np.unique(f)]
    out.append((FINAL, np.ones(ids.size, dtype=bool)))
    return out


# ---------------------------------------------------------------------------
# χ² top K of many training sets in one pass
# ---------------------------------------------------------------------------
def chi2(a, c, n_r, n_s) -> np.ndarray:
    """χ² of the 2×2 table (carriers a of n_r resistant, c of n_s susceptible), as in
    lib/cpss.chi2_prefilter; 0 where a margin is empty."""
    a, c = np.asarray(a, dtype=np.int64), np.asarray(c, dtype=np.int64)
    b, d = n_r - a, n_s - c
    n = n_r + n_s
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, n * (a * d - b * c).astype(float) ** 2 / denom, 0.0)


class TopK:
    """The ``k`` features with the largest χ² in each training set, ties to the smaller
    feature id. Feed blocks of features with ``update``; the counts of all sets come from
    one matrix product per block (exact: counts stay far below 2**24)."""

    def __init__(self, y, masks: np.ndarray, k: int):
        y = np.asarray(y).astype(bool)
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 2 or masks.shape[1] != y.size:
            raise ValueError("masks must be (n_sets, n_genomes)")
        self.k, self.n_sets = int(k), masks.shape[0]
        res, sus = masks & y, masks & ~y
        self.n_r = res.sum(axis=1).astype(np.int64)
        self.n_s = sus.sum(axis=1).astype(np.int64)
        self.m = np.concatenate([res, sus]).T.astype(np.float32)          # (n, 2S)
        self.ids = [np.empty(0, dtype=np.int64) for _ in range(self.n_sets)]
        self.score = [np.empty(0) for _ in range(self.n_sets)]
        self.a = [np.empty(0, dtype=np.int64) for _ in range(self.n_sets)]
        self.c = [np.empty(0, dtype=np.int64) for _ in range(self.n_sets)]
        self.n_features = 0

    def update(self, ids, bits) -> None:
        """One block: feature ids (b,) and their 0/1 presence over the genomes (b, n)."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return
        self.n_features += int(ids.size)
        counts = np.rint(np.asarray(bits, dtype=np.float32) @ self.m).astype(np.int64)
        a_all, c_all = counts[:, :self.n_sets], counts[:, self.n_sets:]
        for s in range(self.n_sets):
            x = chi2(a_all[:, s], c_all[:, s], self.n_r[s], self.n_s[s])
            take = np.ones(ids.size, dtype=bool)
            if self.ids[s].size >= self.k:                 # only what can enter
                take = x >= self.score[s][-1]
                if not take.any():
                    continue
            cid = np.concatenate([self.ids[s], ids[take]])
            csc = np.concatenate([self.score[s], x[take]])
            ca = np.concatenate([self.a[s], a_all[take, s]])
            cc = np.concatenate([self.c[s], c_all[take, s]])
            order = np.lexsort((cid, -csc))[:self.k]
            self.ids[s], self.score[s] = cid[order], csc[order]
            self.a[s], self.c[s] = ca[order], cc[order]

    def result(self, names) -> pd.DataFrame:
        """set, rank, feature_id, chi2, present_resistant, present_susceptible."""
        if len(names) != self.n_sets:
            raise ValueError("one name per training set")
        return pd.concat([pd.DataFrame({
            "set": name, "rank": np.arange(1, self.ids[s].size + 1),
            "feature_id": self.ids[s], "chi2": self.score[s],
            "present_resistant": self.a[s], "present_susceptible": self.c[s]})
            for s, name in enumerate(names)], ignore_index=True)


# ---------------------------------------------------------------------------
# feature sources
# ---------------------------------------------------------------------------
def pattern_blocks(mm: ModelMatrix, block: int = 1 << 15) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(pattern ids, presence over the model's genomes) of a model matrix, in id order."""
    pmap = np.memmap(mm.dir / "patterns.bin", dtype=np.uint8, mode="r",
                     shape=tuple(mm.summary["patterns_shape"]))
    for s in range(0, mm.n_patterns, block):
        e = min(s + block, mm.n_patterns)
        yield (np.arange(s, e, dtype=np.int64),
               np.unpackbits(np.asarray(pmap[s:e]), axis=1)[:, :mm.n_genomes])


def store_blocks(store: Store, cols, min_support: int,
                 block: int = 1 << 15) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(store indices, presence over the genomes ``cols``) of the store's features that
    pass the frequency filter over those genomes, in index order."""
    cols = np.asarray(cols, dtype=np.int64)
    n = cols.size
    for s in range(0, store.n_unitigs, block):
        e = min(s + block, store.n_unitigs)
        bits = np.unpackbits(np.asarray(store.presence[s:e]), axis=1)[:, cols]
        cnt = bits.sum(axis=1, dtype=np.int64)
        keep = np.flatnonzero((cnt >= min_support) & (n - cnt >= min_support))
        if keep.size:
            yield keep + s, bits[keep]


def store_columns(store: Store, ids, cols) -> np.ndarray:
    """Dense 0/1 block (len(cols) genomes × len(ids) features) of store features."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.zeros((len(cols), 0), dtype=np.uint8)
    rows = np.asarray(store.presence[ids])
    return np.ascontiguousarray(np.unpackbits(rows, axis=1)[:, np.asarray(cols)].T)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_matrix(out_dir, genomes: pd.DataFrame, x: np.ndarray, source_ids, members,
                 n_members, summary: dict) -> dict:
    """A model matrix of the given features without collapsing them.

    ``x`` is (n_genomes × n_features) 0/1 in feature order; ``source_ids`` names each
    feature in its source; ``members`` (store index, feature id) and ``n_members`` follow
    lib/matrix_store's members.npy and patterns.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.uint8)
    n, p = x.shape
    if n != len(genomes):
        raise ValueError("x needs one row per genome")
    genomes[["Genome ID", "label", "lineage"]].to_csv(out / "genomes.csv", index=False)
    (out / "patterns.bin").write_bytes(np.packbits(x.T, axis=1).tobytes() if p else b"")
    xb = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.uint8,
                                   shape=(n, matrix_store._nbytes(p)))
    if p:
        xb[:] = np.packbits(x, axis=1)
    xb.flush()
    del xb
    pd.DataFrame({"pattern_id": np.arange(p), "n_present": x.sum(axis=0, dtype=np.int64),
                  "n_members": np.asarray(n_members, dtype=np.int64),
                  "source_id": np.asarray(source_ids, dtype=np.int64)}).to_csv(
        out / "patterns.csv", index=False)
    np.save(out / "members.npy", np.asarray(members, dtype=np.int64).reshape(-1, 2))
    labels = genomes["label"].astype(int)
    s = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **summary,
         "n_genomes": n, "n_resistant": int((labels == 1).sum()),
         "n_susceptible": int((labels == 0).sum()), "n_patterns": p,
         "n_unitigs_kept": int(np.asarray(members).reshape(-1, 2).shape[0]),
         "x_shape": [n, matrix_store._nbytes(p)],
         "patterns_shape": [p, matrix_store._nbytes(n)],
         "x_sha256": matrix_store.sha256_file(out / "X.npy")}
    (out / "matrix_summary.json").write_text(json.dumps(s, indent=2, default=str) + "\n")
    return s


# ---------------------------------------------------------------------------
# reading: an outer fold's view
# ---------------------------------------------------------------------------
class SubsetMatrix:
    """Some columns of a model matrix, read as a model matrix of their own."""

    def __init__(self, mm: ModelMatrix, cols):
        self.mm = mm
        self.cols = np.asarray(cols, dtype=np.int64)
        self.dir, self.summary, self.genomes = mm.dir, mm.summary, mm.genomes
        self.n_patterns = int(self.cols.size)

    @property
    def n_genomes(self) -> int:
        return self.mm.n_genomes

    @property
    def labels(self) -> np.ndarray:
        return self.mm.labels

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256(self.mm.fingerprint.encode())
        h.update(self.cols.tobytes())
        return h.hexdigest()

    def rows(self, row_indices) -> np.ndarray:
        return np.ascontiguousarray(self.mm.rows(row_indices)[:, self.cols])

    def row_batches(self, row_indices, batch_size: int) -> Iterator[np.ndarray]:
        idx = np.asarray(row_indices, dtype=np.int64)
        for s in range(0, idx.size, batch_size):
            yield self.rows(idx[s:s + batch_size])

    def pattern(self, pattern_id: int) -> np.ndarray:
        return self.mm.pattern(int(self.cols[pattern_id]))

    def present_in(self, genome_mask, block: int = 1 << 15) -> np.ndarray:
        return self.mm.present_in(genome_mask, block)[self.cols]

    def columns(self, pattern_ids) -> np.ndarray:
        ids = np.asarray(pattern_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n_patterns):
            raise IndexError(f"pattern ids must lie in [0, {self.n_patterns})")
        return self.mm.columns(self.cols[ids])


def has_budget(matrix_dir) -> bool:
    """Whether a model matrix was written with a feature budget (arm B or C)."""
    return (Path(matrix_dir) / UNION / UNIT_COLUMNS).exists()


def unit_matrix(matrix_dir, unit: str) -> SubsetMatrix:
    """The features one outer fold chose, as a model matrix."""
    union = Path(matrix_dir) / UNION
    with np.load(union / UNIT_COLUMNS) as z:
        if unit not in z.files:
            raise KeyError(f"{unit}: no feature selection in {union}")
        cols = z[unit]
    return SubsetMatrix(ModelMatrix(union), cols)


def final_pattern_ids(matrix_dir, sub: SubsetMatrix, cols) -> np.ndarray:
    """Pattern ids in the final model's matrix of some columns of an outer fold's view
    (the same source feature), -1 where the final model's choice lacks the feature."""
    src = pd.read_csv(Path(matrix_dir) / UNION / "patterns.csv")["source_id"].to_numpy()
    final = pd.read_csv(Path(matrix_dir) / "patterns.csv")["source_id"].to_numpy()
    if final.size and (np.diff(final) <= 0).any():
        raise ValueError("the final matrix's features must be in source id order")
    want = src[sub.cols[np.asarray(cols, dtype=np.int64)]]
    idx = np.searchsorted(final, want)
    ok = idx < final.size
    ok[ok] = final[idx[ok]] == want[ok]
    return np.where(ok, idx, -1).astype(np.int64)
