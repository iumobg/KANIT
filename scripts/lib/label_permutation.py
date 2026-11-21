"""Label permutation test of a model.

The first repeat of the lineage-aware arm: in each of R permutations the labels
are permuted over all genomes of the model, and the outer-fold models are
refitted on their training genomes with the hyperparameters, number of trees and
seed that 04 chose for that fold on the real labels; class weights follow the
permuted labels, as in training. The statistic is the pooled out-of-fold
ROC-AUC. p = (1 + #{null AUC ≥ observed AUC}) / (R + 1), z = (observed − null
mean) / null SD, Benjamini–Hochberg across models; a model with q ≥ alpha is
flagged permutation_not_significant and stays in every analysis.
Hyperparameters and numbers of trees are not searched again (a known
limitation).

Permutation b is seeded by (seed, b), so permutations are computed in any order:
a task is one chunk of permutations of one fold. Each fold's XGBoost matrix is
built once; only its labels and weights change.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import false_discovery_control
from sklearn.metrics import roc_auc_score

from lib import folds, train

REPEAT = 1
FLAG = "permutation_not_significant"


def permuted_labels(y, seed: int, b: int) -> np.ndarray:
    """Labels of permutation b: one permutation over all genomes of the model."""
    y = np.asarray(y)
    return y[np.random.default_rng([seed, b]).permutation(y.size)]


def chunks(n_permutations: int, size: int) -> list[np.ndarray]:
    """Permutation ids 0 .. R-1 in consecutive chunks of ``size``."""
    return [np.arange(s, min(s + size, n_permutations)) for s in range(0, n_permutations, size)]


@dataclass
class Fold:
    """One outer fold of the first lineage-aware repeat, as 04 trained it."""
    fold: int
    train_rows: np.ndarray
    test_rows: np.ndarray
    p: np.ndarray               # 04's predictions of the test genomes (real labels)
    params: dict
    n_trees: int
    seed: int


def folds_of(mm, cv_dir) -> list[Fold]:
    """The outer folds of repeat 1 of the lineage-aware arm; every one must be finished."""
    cv_dir = Path(cv_dir)
    ft = pd.read_csv(cv_dir / "folds.csv", dtype={"genome_id": str})
    ft = ft[(ft["arm"] == folds.LINEAGE_AWARE) & (ft["repeat"] == REPEAT)]
    if ft.empty:
        raise ValueError("no folds for the first lineage-aware repeat (not evaluable?)")
    row_of = pd.Series(np.arange(mm.n_genomes), index=mm.genomes["Genome ID"].astype(str))
    fold_of = pd.Series(ft["fold"].to_numpy(dtype=int), index=ft["genome_id"]).reindex(
        row_of.index)
    if fold_of.isna().any():
        raise ValueError("folds.csv does not cover the model's genomes")
    fold_of = fold_of.to_numpy(dtype=int)
    out = []
    for k in sorted(set(fold_of.tolist())):
        unit = cv_dir / "units" / folds.unit_name(folds.LINEAGE_AWARE, REPEAT, k)
        if not (unit / "record.json").exists():
            raise FileNotFoundError(f"outer fold not finished: {unit}")
        rec = json.loads((unit / "record.json").read_text())
        oof = pd.read_csv(unit / "oof.csv", dtype={"genome_id": str})
        test = np.flatnonzero(fold_of == k)
        p = pd.Series(oof["p"].to_numpy(), index=oof["genome_id"]).reindex(
            mm.genomes["Genome ID"].astype(str).to_numpy()[test])
        if p.isna().any():
            raise ValueError(f"{unit}: oof.csv does not cover the fold's genomes")
        out.append(Fold(k, np.flatnonzero(fold_of != k), test, p.to_numpy(dtype=float),
                        dict(rec["params"]), int(rec["n_trees"]), int(rec["seed"])))
    return out


class Refitter:
    """One fold's training matrix, built once, refitted with other labels."""

    def __init__(self, mm, fold: Fold, *, threads: int, batch_rows: int = 128):
        self.mm, self.fold, self.threads = mm, fold, threads
        y = mm.labels.astype(int)[fold.train_rows]
        self.d = train.dmatrix(mm, fold.train_rows, y, train.class_weight(y),
                               batch_rows=batch_rows)

    def predict(self, labels) -> np.ndarray:
        """Predictions of the fold's test genomes by the model refitted on ``labels``
        (all genomes of the model)."""
        y = np.asarray(labels)[self.fold.train_rows]
        self.d.set_label(y)
        self.d.set_weight(train.class_weight(y))
        params = {**train.BASE_PARAMS, **self.fold.params, "seed": self.fold.seed,
                  "nthread": self.threads}
        booster = xgb.train(params, self.d, self.fold.n_trees)
        return train.predict_sparse(booster, self.mm, self.fold.test_rows)


def chunk_file(out_dir, fold: int, chunk: int) -> Path:
    return Path(out_dir) / "null" / f"fold{fold}_chunk{chunk:03d}.npz"


def chunk_key(fold: Fold, labels, perms, seed: int, matrix: str) -> str:
    """Fingerprint of everything a chunk's null predictions depend on: the matrix (its
    checksum), the fold's genomes, the model 04 chose for it, the labels and the
    permutations. A chunk file is used only with its own key, so that a file of another run
    (other folds, models or matrix) is never mixed in (pilot, 2026-10-06)."""
    h = hashlib.sha256(json.dumps({"params": fold.params, "n_trees": fold.n_trees,
                                   "seed": fold.seed, "permutation_seed": int(seed),
                                   "matrix": matrix}, sort_keys=True).encode())
    for a in (fold.train_rows, fold.test_rows, labels, perms):
        h.update(np.ascontiguousarray(np.asarray(a), dtype=np.int64).tobytes())
    return h.hexdigest()


def chunk_is_current(path, key: str) -> bool:
    """Whether ``path`` holds a chunk written for the inputs of ``key``."""
    if not Path(path).exists():
        return False
    with np.load(path) as z:
        return "key" in z.files and str(z["key"]) == key


def run_chunk(refitter: Refitter, perms: np.ndarray, *, seed: int, path: Path) -> None:
    """Null predictions of one fold for the permutations ``perms``, written atomically
    with the key of their inputs."""
    labels = refitter.mm.labels.astype(int)
    p = np.stack([refitter.predict(permuted_labels(labels, seed, int(b))) for b in perms])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    key = chunk_key(refitter.fold, labels, perms, seed, refitter.mm.fingerprint)
    np.savez(tmp, perms=perms, test_rows=refitter.fold.test_rows, p=p, key=np.array(key))
    os.replace(tmp, path)


def null_auc(mm, fold_list: list[Fold], out_dir, *, n_permutations: int, chunk: int,
             seed: int) -> np.ndarray:
    """Pooled out-of-fold AUC of every permutation; every chunk of every fold must exist."""
    pred = np.full((n_permutations, mm.n_genomes), np.nan)
    labels = mm.labels.astype(int)
    missing = []
    for f in fold_list:
        for c, perms in enumerate(chunks(n_permutations, chunk)):
            path = chunk_file(out_dir, f.fold, c)
            if not path.exists():
                missing.append(path.name)
                continue
            z = np.load(path)
            if "key" not in z.files or str(z["key"]) != chunk_key(f, labels, perms, seed,
                                                                  mm.fingerprint):
                raise ValueError(f"{path}: written for other folds, models, labels, "
                                 "permutations or matrix")
            pred[np.ix_(perms, f.test_rows)] = z["p"]
    if missing:
        raise FileNotFoundError(f"{len(missing)} null chunk(s) missing, e.g. {missing[:3]}")
    if np.isnan(pred).any():
        raise ValueError("the folds do not cover every genome")
    labels = mm.labels.astype(int)
    return np.array([roc_auc_score(permuted_labels(labels, seed, b), pred[b])
                     for b in range(n_permutations)])


def observed_auc(mm, fold_list: list[Fold]) -> float:
    rows = np.concatenate([f.test_rows for f in fold_list])
    return float(roc_auc_score(mm.labels.astype(int)[rows],
                               np.concatenate([f.p for f in fold_list])))


def summary(observed: float, null: np.ndarray, *, seed: int) -> dict:
    n_ge = int((null >= observed).sum())
    sd = float(null.std(ddof=1)) if null.size > 1 else float("nan")
    return {"arm": folds.LINEAGE_AWARE, "repeat": REPEAT, "n_permutations": int(null.size),
            "seed": seed, "auc_observed": observed, "null_mean": float(null.mean()),
            "null_sd": sd, "z": (observed - float(null.mean())) / sd if sd > 0 else float("nan"),
            "n_null_ge_observed": n_ge, "p": (1 + n_ge) / (null.size + 1)}


def across_models(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Benjamini–Hochberg over the evaluable models; q ≥ alpha flags the model."""
    out = rows.copy()
    ev = out["p"].notna()
    out["q"] = np.nan
    if ev.any():
        out.loc[ev, "q"] = false_discovery_control(out.loc[ev, "p"].to_numpy(), method="bh")
    out["flag"] = np.where(ev & (out["q"] >= alpha), FLAG, "")
    return out
