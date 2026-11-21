"""Complementary pairs stability selection (CPSS) of one model and
the candidate patterns.

Prefilter: the patterns with the largest χ² (Pearson, the 2 × 2 table of the
pattern and the label, no continuity correction) over all genomes of the model;
patterns outside it have π = 0. B pairs of disjoint half-samples, each half
stratified by class, give 2B fits. Every fit uses the final model's
hyperparameters and number of trees (04), class weights of its half, and
colsample_bytree rescaled so that a tree samples the same expected number of
patterns as in the final model (at most the whole prefilter). A fit selects the q
patterns with the highest total gain, or every pattern it uses if fewer. π is the
share of the 2B fits that select a pattern; a pattern is stable at π ≥
pi_threshold. The Meinshausen–Bühlmann bound q² / ((2 π_thr − 1) · p) on the
expected number of falsely selected patterns is reported.

Pair b is split with the seed (seed, b) and its fits use the XGBoost seeds 2b and
2b + 1, so pairs are computed in any order and in chunks written to disk.

Candidates: the final model's q_gain patterns with the highest total gain (or all
it uses, if fewer) and the stable patterns.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xgboost as xgb

from lib import train

_TOL = 1e-12


def chi2_prefilter(mm, k: int) -> pd.DataFrame:
    """The k patterns with the largest χ² over all genomes (ties: smaller pattern id)."""
    resistant = mm.labels.astype(bool)
    n, n_r = resistant.size, int(resistant.sum())
    n_s = n - n_r
    a = mm.present_in(resistant)                 # carriers among R
    c = mm.present_in(~resistant)                # carriers among S
    b, d = n_r - a, n_s - c
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = np.where(denom > 0, n * (a * d - b * c).astype(float) ** 2 / denom, 0.0)
    ids = np.arange(mm.n_patterns)
    top = np.lexsort((ids, -chi2))[:k]
    return pd.DataFrame({"rank": np.arange(1, top.size + 1), "pattern_id": top, "chi2": chi2[top],
                         "present_resistant": a[top], "present_susceptible": c[top]})


def half_samples(y, seed: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves of the genomes, each holding half of every class."""
    y = np.asarray(y)
    rng = np.random.default_rng([seed, b])
    first, second = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        h = idx.size // 2
        first.append(idx[:h])
        second.append(idx[h:2 * h])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))


def rescaled_colsample(colsample: float, n_patterns: int, n_prefilter: int) -> float:
    """colsample_bytree that samples colsample · n_patterns patterns of the prefilter."""
    return min(1.0, colsample * n_patterns / n_prefilter)


def top_gain(booster: xgb.Booster, q: int) -> np.ndarray:
    """Feature indices of the q highest total gains (ties: smaller index), or all used."""
    gain = cast(dict[str, float], booster.get_score(importance_type="total_gain"))
    ranked = sorted((-g, int(f[1:])) for f, g in gain.items())
    return np.array([f for _, f in ranked[:q]], dtype=np.int64)


def fit_selection(x, y, rows, params: dict, n_trees: int, seed: int, q: int,
                  threads: int) -> np.ndarray:
    """Prefilter positions selected by one fit on ``rows``."""
    yr = np.asarray(y)[rows]
    d = xgb.QuantileDMatrix(x[rows], label=yr, weight=train.class_weight(yr),
                            max_bin=train.BASE_PARAMS["max_bin"])
    booster = xgb.train({**train.BASE_PARAMS, **params, "seed": seed, "nthread": threads}, d,
                        n_trees)
    return top_gain(booster, q)


def chunks(n_pairs: int, size: int) -> list[np.ndarray]:
    return [np.arange(s, min(s + size, n_pairs)) for s in range(0, n_pairs, size)]


def chunk_file(out_dir, chunk: int) -> Path:
    return Path(out_dir) / "pairs" / f"chunk{chunk:03d}.npz"


def chunk_key(pairs, *, params: dict, n_trees: int, q: int, seed: int, pattern_ids, y,
              matrix: str) -> str:
    """Fingerprint of everything a chunk's selections depend on: the matrix (its checksum),
    the final model's hyperparameters and number of trees, the prefilter, the labels and
    the pairs. A chunk file is used only with its own key, so that a file written for
    another final model or matrix is never mixed in (pilot, 2026-10-06)."""
    h = hashlib.sha256(json.dumps({"params": params, "n_trees": int(n_trees), "q": int(q),
                                   "seed": int(seed), "matrix": matrix},
                                  sort_keys=True).encode())
    for a in (pattern_ids, y, pairs):
        h.update(np.ascontiguousarray(np.asarray(a), dtype=np.int64).tobytes())
    return h.hexdigest()


def chunk_is_current(path, key: str) -> bool:
    """Whether ``path`` holds a chunk written for the inputs of ``key``."""
    if not Path(path).exists():
        return False
    with np.load(path) as z:
        return "key" in z.files and str(z["key"]) == key


def run_chunk(x, y, pairs: np.ndarray, *, params: dict, n_trees: int, q: int, seed: int,
              threads: int, path: Path, key: str) -> None:
    """Selections of both fits of every pair in ``pairs`` (pairs × 2 × q, -1 padded),
    written with the key of their inputs."""
    sel = np.full((pairs.size, 2, q), -1, dtype=np.int64)
    for i, b in enumerate(pairs.tolist()):
        for h, rows in enumerate(half_samples(y, seed, b)):
            s = fit_selection(x, y, rows, params, n_trees, 2 * b + h, q, threads)
            sel[i, h, :s.size] = s
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, pairs=pairs, selected=sel, key=np.array(key))
    os.replace(tmp, path)


def selection_frequency(out_dir, n_prefilter: int, *, n_pairs: int, chunk: int,
                        key: Callable[[np.ndarray], str]) -> np.ndarray:
    """π of every prefilter position; every chunk must exist, written for the inputs that
    ``key`` (pairs -> key) describes."""
    counts = np.zeros(n_prefilter, dtype=np.int64)
    missing = []
    for c, pairs in enumerate(chunks(n_pairs, chunk)):
        path = chunk_file(out_dir, c)
        if not path.exists():
            missing.append(path.name)
            continue
        z = np.load(path)
        if "key" not in z.files or str(z["key"]) != key(pairs):
            raise ValueError(f"{path}: written for another final model, prefilter, labels, "
                             "pairs or matrix")
        s = z["selected"]
        counts += np.bincount(s[s >= 0], minlength=n_prefilter)
    if missing:
        raise FileNotFoundError(f"{len(missing)} CPSS chunk(s) missing, e.g. {missing[:3]}")
    return counts / (2 * n_pairs)


def mb_bound(q: int, pi_threshold: float, n_prefilter: int) -> float:
    """Meinshausen–Bühlmann bound on the expected number of falsely selected patterns."""
    return q ** 2 / ((2 * pi_threshold - 1) * n_prefilter)


def stable(pi, pi_threshold: float) -> np.ndarray:
    return np.asarray(pi) >= pi_threshold - _TOL


def layer(pattern_ids, cpss_table: pd.DataFrame, pi_threshold: float) -> pd.DataFrame:
    """The CPSS layer of the given patterns: χ² and π from the CPSS table (π = 0 outside
    the prefilter) and whether π reaches the threshold."""
    out = pd.DataFrame({"pattern_id": np.asarray(list(pattern_ids), dtype=np.int64)}).merge(
        cpss_table[["pattern_id", "chi2", "pi"]], on="pattern_id", how="left")
    out = out.assign(in_prefilter=out["pi"].notna(), pi=out["pi"].fillna(0.0))
    out["passes"] = stable(out["pi"], pi_threshold)
    return out


def candidates(final_booster: xgb.Booster, cpss_table: pd.DataFrame, q_gain: int) -> pd.DataFrame:
    """The candidates: the final model's top total-gain patterns and the stable patterns."""
    gain = cast(dict[str, float], final_booster.get_score(importance_type="total_gain"))
    top = top_gain(final_booster, q_gain)
    by_gain = pd.DataFrame({"pattern_id": top, "gain_rank": np.arange(1, top.size + 1),
                            "total_gain": [gain[f"f{p}"] for p in top]})
    st = cpss_table.loc[cpss_table["stable"], ["pattern_id"]]
    out = by_gain.merge(st.assign(cpss_stable=True), on="pattern_id", how="outer")
    out["cpss_stable"] = out["cpss_stable"].astype("boolean").fillna(False).astype(bool)
    out["source"] = np.where(out["gain_rank"].notna() & out["cpss_stable"], "both",
                             np.where(out["gain_rank"].notna(), "gain", "cpss"))
    pi = cpss_table.set_index("pattern_id")["pi"]
    out["cpss_pi"] = out["pattern_id"].map(pi).fillna(0.0)
    out["gain_rank"] = out["gain_rank"].astype("Int64")
    return out.sort_values("pattern_id", ignore_index=True)
