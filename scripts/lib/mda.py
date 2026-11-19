"""Permutation importance (MDA) of the candidate patterns (protocol §8.3).

The fold models of the lineage-aware arm (04, every repeat) predict their own
test folds; a pattern, or a group of patterns, is permuted only within each test
fold, and the models are not refitted. The statistic is the drop in pooled
out-of-fold ROC-AUC, averaged over the repeats. One permutation draw is applied
to every repeat and fold together, and the same draws serve every group.
p = (1 + #{permuted AUC ≥ observed AUC}) / (R + 1), Benjamini–Hochberg over the
groups of the model, and the layer passes at q < alpha. The main analysis
permutes each candidate pattern (identical unitigs are one pattern already); the
sensitivity analysis permutes clusters of candidates linked by |r| ≥ r_min
(connected components, r over all genomes of the model).

A fold model's predictions depend on a permuted group only through the values
the group takes. Each fold model therefore predicts its test genomes once for
every distinct value vector of the group in that fold (two for one pattern), and
a draw only looks the predictions up. The test genomes reach XGBoost as a sparse
matrix that stores the patterns the fold model splits on, zeros included; a
pattern it never splits on cannot change its predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from scipy.sparse.csgraph import connected_components
from scipy.stats import false_discovery_control, rankdata

from lib import budget, folds, train

_TOL = 1e-12        # float slack when comparing AUCs and thresholds


def auc_rows(scores, y) -> np.ndarray:
    """ROC-AUC of every row of ``scores`` against the 0/1 labels ``y`` (ties count 1/2)."""
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    ranks = rankdata(np.atleast_2d(scores), axis=1)
    return (ranks[:, y].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass
class FoldModel:
    """One lineage-aware outer fold: its model and its test genomes."""
    repeat: int
    fold: int
    rows: np.ndarray            # model-matrix rows of the test genomes
    p: np.ndarray               # 04's out-of-fold predictions of these genomes
    booster: xgb.Booster
    used: np.ndarray            # patterns the model splits on
    csr: sp.csr_matrix          # test genomes × all patterns; the used ones stored
    # the used patterns' ids in the model matrix whose patterns are permuted, when the fold
    # model saw other columns (the feature budget, lib/budget.py); -1: not among them
    ids: np.ndarray | None = None

    @property
    def values(self) -> np.ndarray:
        """Stored values (test genomes × used patterns): a view into the matrix."""
        return self.csr.data.reshape(len(self.rows), len(self.used))

    def predict(self) -> np.ndarray:
        return np.asarray(self.booster.inplace_predict(self.csr), dtype=float)


def fold_model(mm, unit: Path, rows: np.ndarray, p: np.ndarray, repeat: int,
               fold: int) -> FoldModel:
    booster = xgb.Booster(model_file=str(unit / "model.ubj"))
    booster.set_param({"nthread": 1})          # the fastest for wide models (train.predict_sparse)
    if booster.num_features() != mm.n_patterns:
        raise ValueError(f"{unit}: the model has {booster.num_features()} features, "
                         f"the model matrix {mm.n_patterns} patterns")
    used = train.used_patterns(booster)
    fm = FoldModel(repeat, fold, rows, p, booster, used, train.sparse_rows(mm, rows, used))
    if not np.allclose(fm.predict(), p, rtol=0, atol=1e-6):
        raise ValueError(f"{unit}: the stored model does not reproduce oof.csv")
    return fm


def load_fold_models(mm, cv_dir) -> list[FoldModel]:
    """Every outer fold of every lineage-aware repeat; all must be finished."""
    cv_dir = Path(cv_dir)
    seeds = pd.read_csv(cv_dir / "seeds.csv")
    aware = seeds[seeds["arm"] == folds.LINEAGE_AWARE]
    if aware.empty or not aware["evaluable"].all():
        raise ValueError("the lineage-aware arm is not evaluable")
    ft = pd.read_csv(cv_dir / "folds.csv", dtype={"genome_id": str})
    ft = ft[ft["arm"] == folds.LINEAGE_AWARE]
    row_of = pd.Series(np.arange(mm.n_genomes), index=mm.genomes["Genome ID"].astype(str))
    labels = mm.labels.astype(int)
    out = []
    for repeat in sorted(aware["repeat"].astype(int)):
        fr = ft[ft["repeat"] == repeat]
        for fold in sorted(int(f) for f in fr["fold"].unique()):
            unit = cv_dir / "units" / folds.unit_name(folds.LINEAGE_AWARE, repeat, fold)
            if not (unit / "record.json").exists():
                raise FileNotFoundError(f"outer fold not finished: {unit}")
            oof = pd.read_csv(unit / "oof.csv", dtype={"genome_id": str})
            if set(oof["genome_id"]) != set(fr.loc[fr["fold"] == fold, "genome_id"]):
                raise ValueError(f"{unit}: test genomes differ from folds.csv")
            rows = row_of.reindex(oof["genome_id"])
            if rows.isna().any():
                raise ValueError(f"{unit}: genomes missing from the model matrix")
            rows = rows.to_numpy(dtype=np.int64)
            if not (labels[rows] == oof["y"].to_numpy(dtype=int)).all():
                raise ValueError(f"{unit}: labels differ from the model matrix")
            if budget.has_budget(mm.dir):   # the fold model saw its own training set's choice
                um = budget.unit_matrix(mm.dir, folds.unit_name(folds.LINEAGE_AWARE, repeat,
                                                                fold))
                fm = fold_model(um, unit, rows, oof["p"].to_numpy(dtype=float), repeat, fold)
                fm.ids = budget.final_pattern_ids(mm.dir, um, fm.used)
            else:
                fm = fold_model(mm, unit, rows, oof["p"].to_numpy(dtype=float), repeat, fold)
            out.append(fm)
    return out


@dataclass
class Design:
    """Fold models, their draws and the observed pooled AUC of every repeat."""
    models: list[FoldModel]
    n_permutations: int
    seed: int
    labels: np.ndarray
    perms: list[np.ndarray] = field(init=False)
    by_repeat: dict[int, list[int]] = field(init=False)
    auc: dict[int, float] = field(init=False)

    def __post_init__(self):
        self.perms = []
        for fm in self.models:      # draw b of every fold together make draw b
            rng = np.random.default_rng([self.seed, fm.repeat, fm.fold])
            perm = np.argsort(rng.random((self.n_permutations, len(fm.rows))), axis=1)
            self.perms.append(perm.astype(np.min_scalar_type(max(len(fm.rows) - 1, 0))))
        self.by_repeat = {}
        for i, fm in enumerate(self.models):
            self.by_repeat.setdefault(fm.repeat, []).append(i)
        self.auc = {r: float(auc_rows(np.concatenate([self.models[i].p for i in idx]),
                                      self.y(r))[0]) for r, idx in self.by_repeat.items()}

    def y(self, repeat: int) -> np.ndarray:
        return np.concatenate([self.labels[self.models[i].rows] for i in self.by_repeat[repeat]])

    @property
    def observed(self) -> float:
        return sum(self.auc[r] for r in sorted(self.auc)) / len(self.auc)


def _effect(fm: FoldModel, group: np.ndarray):
    """(codes, preds) when the fold model splits on the group, else None: codes[i]
    is test genome i's value vector of the group among the fold's distinct vectors,
    preds[c] the predictions of every test genome with the group set to vector c."""
    pos = np.flatnonzero(np.isin(fm.used if fm.ids is None else fm.ids, group))
    if not pos.size:
        return None
    view = fm.values
    real = view[:, pos].copy()
    uniq, codes = np.unique(real, axis=0, return_inverse=True)
    codes = codes.ravel()
    preds = np.empty((len(uniq), len(fm.rows)))
    try:
        for c, v in enumerate(uniq):
            view[:, pos] = v
            preds[c] = fm.predict()
    finally:
        view[:, pos] = real
    if not np.allclose(preds[codes, np.arange(codes.size)], fm.p, rtol=0, atol=1e-6):
        raise AssertionError("the group's own values must reproduce the fold's predictions")
    return codes, preds


def permuted_auc(design: Design, group) -> tuple[np.ndarray, int]:
    """Mean pooled AUC over the repeats in every draw, and the number of fold models
    that split on the group (none: every draw equals the observed AUC)."""
    group = np.asarray(group, dtype=np.int64)
    total = np.zeros(design.n_permutations)
    n_used = 0
    for r in sorted(design.by_repeat):
        blocks, changed = [], False
        for i in design.by_repeat[r]:
            fm, perm = design.models[i], design.perms[i]
            eff = _effect(fm, group)
            if eff is None:
                blocks.append(np.broadcast_to(fm.p, (design.n_permutations, fm.p.size)))
                continue
            codes, preds = eff
            blocks.append(preds[codes[perm], np.arange(codes.size)])
            changed, n_used = True, n_used + 1
        total += auc_rows(np.hstack(blocks), design.y(r)) if changed else design.auc[r]
    return total / len(design.by_repeat), n_used


# the columns of the layer (one row per pattern) and of its sensitivity analysis
MAIN_COLUMNS = ("pattern_id", "n_fold_models_using", "auc_observed", "auc_permuted", "mda",
                "n_permuted_ge_observed", "p", "q", "passes")
CLUSTER_COLUMNS = ("pattern_id", "cluster", "cluster_size", "cluster_patterns", *MAIN_COLUMNS[1:])


def evaluate_groups(design: Design, groups, *, alpha: float = 0.05) -> pd.DataFrame:
    """One row per group: observed and mean permuted AUC, MDA, p, q and passes."""
    obs, n = design.observed, design.n_permutations
    rows = []
    for g in groups:
        perm, n_used = permuted_auc(design, g)
        n_ge = int((perm >= obs - _TOL).sum())
        rows.append({"n_fold_models_using": n_used, "auc_observed": obs,
                     "auc_permuted": float(perm.mean()), "mda": float((obs - perm).mean()),
                     "n_permuted_ge_observed": n_ge, "p": (1 + n_ge) / (n + 1)})
    out = pd.DataFrame(rows, columns=list(MAIN_COLUMNS[1:-2]))
    out["q"] = false_discovery_control(out["p"], method="bh") if len(out) else []
    out["passes"] = out["q"] < alpha
    return out


def correlation_clusters(mm, patterns, r_min: float) -> list[np.ndarray]:
    """Groups of patterns linked by |r| ≥ r_min over the model's genomes (connected
    components); a pattern without such a partner is a group of its own."""
    patterns = np.asarray(patterns, dtype=np.int64)
    if patterns.size < 2:
        return [patterns[i:i + 1] for i in range(patterns.size)]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(mm.columns(patterns).astype(np.float64), rowvar=False)
    linked = sp.csr_matrix(np.nan_to_num(np.abs(r)) >= r_min - _TOL)
    _, label = connected_components(linked, directed=False)
    return [patterns[label == c] for c in range(label.max() + 1)]


def empty_layer() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(main, clusters) of no patterns (an empty association set), without the fold
    models."""
    return pd.DataFrame(columns=list(MAIN_COLUMNS)), pd.DataFrame(columns=list(CLUSTER_COLUMNS))


def mda_layer(mm, design: Design, patterns, *, alpha: float = 0.05,
              r_min: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(main, clusters): every candidate permuted alone, and the sensitivity
    analysis, in which each member of a cluster takes the cluster's result."""
    patterns = np.asarray(sorted(set(int(p) for p in patterns)), dtype=np.int64)
    if not patterns.size:
        return empty_layer()
    main = evaluate_groups(design, [[p] for p in patterns], alpha=alpha)
    main.insert(0, "pattern_id", patterns)
    clusters = correlation_clusters(mm, patterns, r_min)
    res = evaluate_groups(design, clusters, alpha=alpha)
    parts = [pd.DataFrame({"pattern_id": g, "cluster": c, "cluster_size": g.size,
                           "cluster_patterns": ";".join(map(str, g))})
             .assign(**res.iloc[c].to_dict()) for c, g in enumerate(clusters)]
    clus = pd.concat(parts, ignore_index=True).sort_values("pattern_id", ignore_index=True)
    return main, clus
