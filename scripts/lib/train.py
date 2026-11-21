"""Hyperparameter search, tree count and training of one model.

For one training set (an outer training set, or every genome for the final
model):

1. **Search data**: up to ``max_genomes`` genomes, drawn within lineages in
   proportion to lineage size and keeping the R/S ratio.
2. **Search**: Optuna TPE over the search space below; each trial is trained on
   four fifths of the search data with early stopping on the remaining fifth (an
   inner split of the same type as the outer arm) and pruned by a median pruner.
3. **Number of trees**: the chosen hyperparameters are fitted on four fifths of
   the whole training set with early stopping on the remaining fifth.
4. **Fit**: the whole training set, with that number of trees.

Every random step uses the given seed. Resistant genomes carry the weight
n(S)/n(R) of their training set. Matrices are built from row batches of the
packed model matrix, never densified whole.
"""
from __future__ import annotations

import math

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

from lib import folds

BASE_PARAMS = {"objective": "binary:logistic", "tree_method": "hist", "max_bin": 2,
               "eval_metric": "auc"}


def colsample_range(n_features: int) -> tuple[float, float]:
    """Log-scale window around 1/sqrt(p): 0.5/sqrt(p) (>= 1e-5) to 20/sqrt(p) (0.01 .. 1)."""
    r = math.sqrt(n_features)
    return max(0.5 / r, 1e-5), min(max(20 / r, 0.01), 1.0)


def suggest(trial: optuna.Trial, n_features: int) -> dict:
    lo, hi = colsample_range(n_features)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", lo, hi, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def class_weight(y) -> np.ndarray:
    y = np.asarray(y)
    n_r, n_s = int((y == 1).sum()), int((y == 0).sum())
    if n_r == 0 or n_s == 0:
        raise ValueError("a training set needs both classes")
    w = np.ones(len(y))
    w[y == 1] = n_s / n_r
    return w


def search_subsample(y, groups, max_genomes: int, seed: int) -> np.ndarray:
    """Positions of at most ``max_genomes`` genomes, drawn within (lineage, class)
    cells in proportion to cell size (largest-remainder rounding)."""
    y, groups = np.asarray(y), np.asarray(groups)
    n = len(y)
    if n <= max_genomes:
        return np.arange(n)
    keys = pd.Series(list(zip(groups.tolist(), y.tolist(), strict=True)))
    cells = keys.groupby(keys).indices                   # cell -> positions
    names = sorted(cells, key=str)
    sizes = np.array([len(cells[c]) for c in names])
    exact = max_genomes * sizes / n
    quota = np.floor(exact).astype(int)
    order = np.argsort(-(exact - quota), kind="stable")
    quota[order[:max_genomes - quota.sum()]] += 1
    rng = np.random.default_rng(seed)
    picked = [rng.choice(cells[c], size=q, replace=False) for c, q in zip(names, quota, strict=True)
              if q > 0]
    return np.sort(np.concatenate(picked))


def inner_split(y, groups, arm: str, seed: int, n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(train, validation) positions: of the folds of an ``n_folds`` split of the arm's
    type whose validation and training sides both hold both classes, the one closest in
    size to one fifth of the genomes (ties: the first). A lineage larger than a fold's
    share fills a fold of its own, which is no fifth (the first fold would be the
    largest lineage's)."""
    y = np.asarray(y)
    fold_of = folds.split(y, groups, arm, seed, n_folds)
    usable = [k for k in range(n_folds) if len(set(y[fold_of == k].tolist())) == 2
              and len(set(y[fold_of != k].tolist())) == 2]
    if not usable:
        raise ValueError("no inner validation fold holds both classes")
    k = min(usable, key=lambda k: (abs(int((fold_of == k).sum()) - len(y) / n_folds), k))
    val = fold_of == k
    return np.flatnonzero(~val), np.flatnonzero(val)


class _Rows(xgb.DataIter):
    def __init__(self, mm, rows, y, weight, batch_rows):
        self.mm, self.rows, self.y, self.w = mm, rows, y, weight
        self.starts = list(range(0, len(rows), batch_rows))
        self.batch_rows, self.i = batch_rows, 0
        super().__init__()

    def next(self, input_data):
        if self.i == len(self.starts):
            return False
        s = self.starts[self.i]
        e = min(s + self.batch_rows, len(self.rows))
        input_data(data=self.mm.rows(self.rows[s:e]), label=self.y[s:e],
                   weight=None if self.w is None else self.w[s:e])
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def dmatrix(mm, rows, y, weight=None, ref=None, batch_rows: int = 128) -> xgb.QuantileDMatrix:
    """QuantileDMatrix of the given model-matrix rows; ``y``/``weight`` align with ``rows``."""
    rows = np.asarray(rows, dtype=np.int64)
    return xgb.QuantileDMatrix(_Rows(mm, rows, np.asarray(y), weight, batch_rows),
                               max_bin=BASE_PARAMS["max_bin"], ref=ref)


class _Pruning(xgb.callback.TrainingCallback):
    """Report the validation AUC of every round to Optuna and stop pruned trials."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        self.trial.report(float(evals_log["val"]["auc"][-1]), epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return False


def search(mm, rows, y, groups, arm: str, seed: int, *, n_trials: int, max_genomes: int,
           startup_trials: int, warmup_rounds: int, early_stopping: int, max_rounds: int,
           threads: int, batch_rows: int = 128) -> dict:
    """Best hyperparameters for the training set ``rows`` (model-matrix rows)."""
    rows = np.asarray(rows, dtype=np.int64)
    y_all, g_all = np.asarray(y), np.asarray(groups)
    sub = rows[search_subsample(y_all[rows], g_all[rows], max_genomes, seed)]
    tr, va = inner_split(y_all[sub], g_all[sub], arm, seed)
    tr_rows, va_rows = sub[tr], sub[va]
    dtr = dmatrix(mm, tr_rows, y_all[tr_rows], class_weight(y_all[tr_rows]), batch_rows=batch_rows)
    dva = dmatrix(mm, va_rows, y_all[va_rows], ref=dtr, batch_rows=batch_rows)

    def objective(trial):
        params = {**BASE_PARAMS, **suggest(trial, mm.n_patterns), "seed": seed, "nthread": threads}
        booster = xgb.train(params, dtr, num_boost_round=max_rounds, evals=[(dva, "val")],
                            early_stopping_rounds=early_stopping, callbacks=[_Pruning(trial)],
                            verbose_eval=False)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return float(booster.best_score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=startup_trials,
                                           n_warmup_steps=warmup_rounds))
    study.optimize(objective, n_trials=n_trials)
    states = pd.Series([t.state.name for t in study.trials])
    trials = pd.DataFrame([{"number": t.number, "state": t.state.name, "value": t.value,
                            "best_iteration": t.user_attrs.get("best_iteration"), **t.params}
                           for t in study.trials])
    return {"params": dict(study.best_params), "inner_auc": float(study.best_value),
            "n_search_genomes": int(len(sub)), "n_search_train": int(len(tr_rows)),
            "n_search_validation": int(len(va_rows)),
            "n_trials_complete": int((states == "COMPLETE").sum()),
            "n_trials_pruned": int((states == "PRUNED").sum()), "trials": trials}


def tree_count(params: dict, mm, rows, y, groups, arm: str, seed: int, *, early_stopping: int,
               max_rounds: int, threads: int, batch_rows: int = 128) -> int:
    """Number of trees: early stopping on one fifth of the training set."""
    rows = np.asarray(rows, dtype=np.int64)
    y_all = np.asarray(y)
    tr, va = inner_split(y_all[rows], np.asarray(groups)[rows], arm, seed)
    dtr = dmatrix(mm, rows[tr], y_all[rows[tr]], class_weight(y_all[rows[tr]]), batch_rows=batch_rows)
    dva = dmatrix(mm, rows[va], y_all[rows[va]], ref=dtr, batch_rows=batch_rows)
    full = {**BASE_PARAMS, **params, "seed": seed, "nthread": threads}
    booster = xgb.train(full, dtr, num_boost_round=max_rounds, evals=[(dva, "val")],
                        early_stopping_rounds=early_stopping, verbose_eval=False)
    return int(booster.best_iteration) + 1


def fit(params: dict, mm, rows, y, n_trees: int, seed: int, *, threads: int,
        batch_rows: int = 128) -> xgb.Booster:
    rows = np.asarray(rows, dtype=np.int64)
    y_all = np.asarray(y)
    d = dmatrix(mm, rows, y_all[rows], class_weight(y_all[rows]), batch_rows=batch_rows)
    return xgb.train({**BASE_PARAMS, **params, "seed": seed, "nthread": threads}, d, n_trees)


def predict(booster: xgb.Booster, mm, rows, batch_rows: int = 512) -> np.ndarray:
    """Predicted probability of resistance for the given model-matrix rows."""
    rows = np.asarray(rows, dtype=np.int64)
    out = [booster.inplace_predict(block) for block in mm.row_batches(rows, batch_rows)]
    return np.concatenate(out) if out else np.empty(0)


def used_patterns(booster: xgb.Booster) -> np.ndarray:
    """Pattern ids (feature indices) the booster splits on."""
    names = booster.get_score(importance_type="weight")
    if any(not (k.startswith("f") and k[1:].isdigit()) for k in names):
        raise ValueError("the booster's features must be unnamed (f0, f1, ...)")
    return np.array(sorted(int(k[1:]) for k in names), dtype=np.int64)


def sparse_rows(mm, rows, used) -> sp.csr_matrix:
    """Model-matrix rows as a sparse matrix that stores the ``used`` patterns, zeros
    included. XGBoost treats an absent entry as missing, but a pattern the model
    never splits on is never read, so the predictions equal those of the dense rows."""
    rows, used = np.asarray(rows, dtype=np.int64), np.asarray(used, dtype=np.int64)
    vals = mm.columns(used)[rows].astype(np.float32)
    n, k = vals.shape
    return sp.csr_matrix((vals.ravel(), np.tile(used.astype(np.int32), n),
                          np.arange(n + 1, dtype=np.int64) * k), shape=(n, mm.n_patterns))


def predict_sparse(booster: xgb.Booster, mm, rows) -> np.ndarray:
    """``predict`` through ``sparse_rows``, single-threaded: XGBoost sets up a buffer
    as wide as the model for every thread, so with millions of patterns one thread
    is the fastest. Changes the booster's thread count."""
    booster.set_param({"nthread": 1})
    return np.asarray(booster.inplace_predict(sparse_rows(mm, rows, used_patterns(booster))),
                      dtype=float)


def versions() -> dict:
    """Versions of the packages a trained model depends on."""
    import sklearn
    return {"xgboost": xgb.__version__, "scikit_learn": sklearn.__version__,
            "optuna": optuna.__version__, "numpy": np.__version__}


def train(mm, rows, y, groups, arm: str, seed: int, cfg: dict, threads: int) -> dict:
    """search -> tree_count -> fit for one training set; returns the model and its record."""
    s = search(mm, rows, y, groups, arm, seed, n_trials=cfg["n_trials"],
               max_genomes=cfg["search_max_genomes"], startup_trials=cfg["pruner_startup_trials"],
               warmup_rounds=cfg["pruner_warmup_rounds"], early_stopping=cfg["early_stopping_rounds"],
               max_rounds=cfg["max_rounds"], threads=threads, batch_rows=cfg["batch_rows"])
    n_trees = tree_count(s["params"], mm, rows, y, groups, arm, seed,
                         early_stopping=cfg["early_stopping_rounds"], max_rounds=cfg["max_rounds"],
                         threads=threads, batch_rows=cfg["batch_rows"])
    booster = fit(s["params"], mm, rows, y, n_trees, seed, threads=threads,
                  batch_rows=cfg["batch_rows"])
    trials = s.pop("trials")
    return {"booster": booster, "trials": trials,
            "record": {**s, "n_trees": n_trees, "seed": seed, "arm": arm,
                       "versions": versions(),
                       "n_train": int(len(rows))}}
