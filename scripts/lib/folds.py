"""Outer folds of the nested cross-validation.

Two arms, identical except for the grouping: the lineage-aware arm keeps every
lineage on one side of each split (a stratified assignment of whole lineages in a
random order, ``group_split``), the lineage-blind arm does not (StratifiedKFold). In
each arm, repeat r uses the first seed of 1000·r + j (j = 0, 1, …) for which every
test fold holds at least
``min_minority`` genomes of the model's minority class, trying at most
``max_attempts`` seeds. The rule reads only labels and lineages, never results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LINEAGE_AWARE, LINEAGE_BLIND = "lineage_aware", "lineage_blind"
ARMS = (LINEAGE_AWARE, LINEAGE_BLIND)
NOT_EVALUABLE = 3       # exit status of a step skipped because the model is not evaluable


def unit_name(arm: str, repeat: int, fold: int) -> str:
    """Directory name of one outer fold's model under <cv_dir>/units/."""
    return f"{arm}_r{repeat}_f{fold}"


@dataclass
class RepeatSplit:
    arm: str
    repeat: int
    seed: int | None          # None when no seed satisfied the rule
    attempts: int
    fold_of: np.ndarray | None  # test fold of every genome, 0 .. n_folds-1

    @property
    def evaluable(self) -> bool:
        return self.fold_of is not None


def split(y, groups, arm: str, seed: int, n_folds: int) -> np.ndarray:
    """Test fold of every genome for one seed."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    if arm == LINEAGE_AWARE:
        return group_split(y, groups, seed, n_folds)
    if arm != LINEAGE_BLIND:
        raise ValueError(f"unknown arm {arm!r}")
    gen = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=seed).split(np.zeros(len(y)), y)
    fold_of = np.full(len(y), -1, dtype=np.int64)
    for k, (_, test) in enumerate(gen):
        fold_of[test] = k
    if (fold_of < 0).any():
        raise RuntimeError("a genome was assigned to no test fold")
    return fold_of


def group_split(y, groups, seed: int, n_folds: int) -> np.ndarray:
    """Lineage-aware test folds. Each lineage goes whole to the fold
    that keeps the class proportions of the folds most even, the criterion of scikit-learn's
    StratifiedGroupKFold (ties: the fold with fewer genomes). The lineages are visited in a
    random order drawn with the seed, those larger than a fold's share first, so that they
    do not land in a fold that is already full. StratifiedGroupKFold visits them in a fixed
    order and its shuffle only breaks ties: in the pilot every seed gave nearly the same
    folds (A. baumannii, 2026-10-06)."""
    y, groups = np.asarray(y), np.asarray(groups)
    lineages, g = np.unique(groups, return_inverse=True)
    if len(lineages) < n_folds:
        raise ValueError(f"{len(lineages)} lineages, fewer than {n_folds} folds")
    classes, c = np.unique(y, return_inverse=True)
    counts = np.zeros((len(lineages), len(classes)))
    np.add.at(counts, (g, c), 1)
    totals = counts.sum(axis=0)
    order = np.random.default_rng(seed).permutation(len(lineages))
    large = counts[order].sum(axis=1) > len(y) / n_folds
    per_fold = np.zeros((n_folds, counts.shape[1]))
    fold_of_lineage = np.empty(len(lineages), dtype=np.int64)
    for i in np.concatenate([order[large], order[~large]]):
        best, best_eval, best_n = 0, np.inf, np.inf
        for k in range(n_folds):
            per_fold[k] += counts[i]
            evenness = float(np.mean(np.std(per_fold / totals, axis=0)))
            per_fold[k] -= counts[i]
            n_k = float(per_fold[k].sum())
            if evenness < best_eval or (np.isclose(evenness, best_eval) and n_k < best_n):
                best, best_eval, best_n = k, evenness, n_k
        per_fold[best] += counts[i]
        fold_of_lineage[i] = best
    return fold_of_lineage[g]


def minority_label(y) -> int:
    """The smaller class of the model (ties: resistant)."""
    y = np.asarray(y)
    return 1 if (y == 1).sum() <= (y == 0).sum() else 0


def minority_per_fold(y, fold_of, n_folds: int) -> np.ndarray:
    y = np.asarray(y)
    minority = minority_label(y)
    return np.array([int(((fold_of == k) & (y == minority)).sum()) for k in range(n_folds)])


def assign_repeat(y, groups, arm: str, repeat: int, *, n_folds: int, min_minority: int,
                  max_attempts: int) -> RepeatSplit:
    """The first seed of 1000·repeat + j that satisfies the class-balance rule."""
    for j in range(max_attempts):
        seed = 1000 * repeat + j
        try:
            fold_of = split(y, groups, arm, seed, n_folds)
        except ValueError:          # e.g. fewer lineages than folds
            return RepeatSplit(arm, repeat, None, j + 1, None)
        if minority_per_fold(y, fold_of, n_folds).min() >= min_minority:
            return RepeatSplit(arm, repeat, seed, j + 1, fold_of)
    return RepeatSplit(arm, repeat, None, max_attempts, None)


def assign_all(y, groups, *, n_repeats: int, n_folds: int, min_minority: int,
               max_attempts: int) -> list[RepeatSplit]:
    """Every (arm, repeat), repeats numbered 1 .. n_repeats."""
    return [assign_repeat(y, groups, arm, r, n_folds=n_folds, min_minority=min_minority,
                          max_attempts=max_attempts)
            for arm in ARMS for r in range(1, n_repeats + 1)]


def folds_table(genome_ids, splits: list[RepeatSplit]) -> pd.DataFrame:
    """folds.csv: repeat, arm, fold, genome_id — one row per genome per evaluable split."""
    ids = np.asarray(genome_ids, dtype=str)
    parts = [pd.DataFrame({"repeat": s.repeat, "arm": s.arm, "fold": s.fold_of, "genome_id": ids})
             for s in splits if s.evaluable]
    if not parts:
        return pd.DataFrame(columns=["repeat", "arm", "fold", "genome_id"])
    return pd.concat(parts, ignore_index=True)


def seeds_table(splits: list[RepeatSplit]) -> pd.DataFrame:
    return pd.DataFrame([{"arm": s.arm, "repeat": s.repeat, "seed": s.seed,
                          "attempts": s.attempts, "evaluable": s.evaluable} for s in splits])


def composition_table(y, groups, splits: list[RepeatSplit], n_folds: int) -> pd.DataFrame:
    """Size, R and S counts, resistance rate and number of lineages of every test fold."""
    y, groups = np.asarray(y), np.asarray(groups)
    rows = []
    for s in splits:
        if s.fold_of is None:
            continue
        for k in range(n_folds):
            m = s.fold_of == k
            n_r, n = int((y[m] == 1).sum()), int(m.sum())
            rows.append({"arm": s.arm, "repeat": s.repeat, "fold": k, "n": n,
                         "n_resistant": n_r, "n_susceptible": n - n_r,
                         "resistance_rate": n_r / n if n else float("nan"),
                         "n_lineages": int(len(set(groups[m].tolist())))})
    return pd.DataFrame(rows)


def largest_lineage_share(groups) -> float:
    """Share of the genomes in the largest lineage; above 1/n_folds no balanced
    lineage-aware split exists, and the model is flagged."""
    _, counts = np.unique(np.asarray(groups), return_counts=True)
    return int(np.max(counts)) / int(np.sum(counts))
