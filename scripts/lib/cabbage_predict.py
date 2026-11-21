"""External validation on CABBAGE: the final models on the
external isolates.

The member unitigs of every pattern a final model uses are searched in the external
assemblies (unitig-caller, simple mode: exact match on either strand). A pattern is present
in an assembly when at least half of its member unitigs are; the model then gives the
probability of resistance, read at threshold 0.5. Every metric has a percentile interval
from lineage-cluster bootstrap resamples: the external isolates' PopPUNK clusters, drawn
with replacement, as in the internal evaluation.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from lib.oof_metrics import _auc

PRESENT_SHARE = 0.5                     # a pattern needs at least half of its members
COMPLETENESS_MIN, CONTAMINATION_MAX = 95.0, 5.0     # CheckM2, as for the panel genomes
METRICS = ["roc_auc", "pr_auc", "pr_auc_baseline", "balanced_accuracy", "sensitivity",
           "specificity", "very_major_error_rate", "major_error_rate"]


def qc_table(report) -> pd.DataFrame:
    """biosample_id, completeness, contamination and passes, from CheckM2's quality_report.tsv
    (its Name is the file stem, the BioSample)."""
    d = pd.read_csv(report, sep="\t", dtype={"Name": str})
    out = pd.DataFrame({"biosample_id": d["Name"], "completeness": d["Completeness"].astype(float),
                        "contamination": d["Contamination"].astype(float)})
    out["passes"] = ((out["completeness"] >= COMPLETENESS_MIN)
                     & (out["contamination"] <= CONTAMINATION_MAX))
    return out


def lineage_seen(lineage: str, model_lineages: set[str]) -> bool:
    """Whether a PopPUNK cluster of an external isolate holds genomes of the model; a cluster
    that the assignment merged ('12_45') counts when any of its parts does."""
    return any(part in model_lineages for part in str(lineage).split("_"))


def used_patterns(booster) -> np.ndarray:
    """Pattern ids (feature indices) the model's trees split on."""
    keys = booster.get_score(importance_type="weight")
    return np.array(sorted(int(re.sub(r"\D", "", k)) for k in keys), dtype=np.int64)


def query_table(mm, store, patterns) -> pd.DataFrame:
    """unitig_index, pattern_id and sequence of the member unitigs of the given patterns."""
    m = mm.members()
    m = m[m["pattern_id"].isin({int(p) for p in patterns})].sort_values("unitig_index")
    return m.assign(sequence=store.sequences(m["unitig_index"].to_numpy())).reset_index(drop=True)


def read_rtab(path) -> pd.DataFrame:
    """unitig-caller's Rtab: one row per unitig sequence, one 0/1 column per genome (read
    as uint8 at once: a large Rtab read as text would not fit in memory)."""
    with open(path, encoding="utf-8") as f:
        genomes = f.readline().rstrip("\n").split("\t")[1:]
    return pd.read_csv(path, sep="\t", index_col=0, dtype={g: np.uint8 for g in genomes})


def pyseer_to_rtab(pyseer, unitigs: list[str], genomes: list[str], rtab) -> None:
    """unitig-caller in simple mode writes only its pyseer format, ``seq | genome:count ...``,
    and leaves out a unitig found in no genome. The Rtab lists every query unitig and every
    genome, so a unitig or genome without a hit is 0 rather than missing."""
    row = {u: i for i, u in enumerate(unitigs)}
    col = {g: j for j, g in enumerate(genomes)}
    hits = np.zeros((len(unitigs), len(genomes)), dtype=np.uint8)
    with open(pyseer, encoding="utf-8") as f:
        for line in f:
            seq, _, calls = line.rstrip("\n").partition(" | ")
            if seq not in row:
                raise ValueError(f"{pyseer}: {seq[:40]} is not a query unitig")
            for call in calls.split():
                genome, _, count = call.rpartition(":")
                if genome not in col:
                    raise ValueError(f"{pyseer}: {genome} is not a query genome")
                hits[row[seq], col[genome]] = int(count) > 0
    pd.DataFrame(hits, index=pd.Index(unitigs, name="Unitig_sequence"), columns=genomes).to_csv(
        rtab, sep="\t")


def pattern_presence(query: pd.DataFrame, rtab: pd.DataFrame) -> pd.DataFrame:
    """pattern × genome 0/1 from the unitig calls; a member the Rtab does not list is absent."""
    hits = rtab.reindex(query["sequence"].to_numpy()).fillna(0).to_numpy(dtype=np.float64)
    share = pd.DataFrame(hits, columns=rtab.columns).groupby(query["pattern_id"].to_numpy()).mean()
    return (share >= PRESENT_SHARE).astype(np.uint8)


def predict(booster, n_patterns: int, presence: pd.DataFrame,
            batch_bytes: float = 2e8) -> pd.Series:
    """Probability of resistance of every genome (column of ``presence``); a pattern the
    model uses but ``presence`` lacks is absent."""
    if booster.num_features() != n_patterns:
        raise ValueError(f"the model has {booster.num_features()} features, the matrix "
                         f"{n_patterns} patterns")
    genomes = presence.columns.tolist()
    pats = presence.index.to_numpy(dtype=np.int64)
    rows = max(1, int(batch_bytes // (4 * n_patterns)))
    out = []
    for s in range(0, len(genomes), rows):
        g = genomes[s:s + rows]
        x = np.zeros((len(g), n_patterns), dtype=np.float32)
        x[:, pats] = presence[g].to_numpy(dtype=np.float32).T
        out.append(booster.inplace_predict(x))
    p = np.concatenate(out) if out else np.empty(0)
    return pd.Series(p, index=genomes, dtype=float)


def metrics(y, p, w=None, threshold: float = 0.5) -> dict:
    """The external-validation metrics, each genome weighted by ``w``."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    pred = p >= threshold
    tp, fn = w[(y == 1) & pred].sum(), w[(y == 1) & ~pred].sum()
    tn, fp = w[(y == 0) & ~pred].sum(), w[(y == 0) & pred].sum()
    nan = float("nan")
    sens = tp / (tp + fn) if tp + fn else nan
    spec = tn / (tn + fp) if tn + fp else nan
    both = (w[y == 1].sum() > 0) and (w[y == 0].sum() > 0)
    return {"roc_auc": _auc(y, p, w),
            "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if both else nan,
            "pr_auc_baseline": float(w[y == 1].sum() / w.sum()) if w.sum() else nan,
            "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
            "very_major_error_rate": 1 - sens, "major_error_rate": 1 - spec}


def bootstrap(y, p, lineage, *, n_boot: int = 1000, seed: int = 0, level: float = 0.95,
              threshold: float = 0.5) -> dict:
    """{metric: {low, high, n_valid}} from lineage-cluster bootstrap resamples."""
    lab = pd.Series(lineage).astype(str).to_numpy()
    lineages, index = np.unique(lab, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(lineages), len(lineages)),
                             minlength=len(lineages))
        m = metrics(y, p, counts[index], threshold)
        for k in METRICS:
            draws[k][b] = m[k]
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out: dict = {}
    for k, x in draws.items():
        x = x[~np.isnan(x)]
        out[k] = ({"low": float(np.percentile(x, lo)), "high": float(np.percentile(x, hi)),
                   "n_valid": int(x.size)} if x.size else {"low": None, "high": None, "n_valid": 0})
    out.update({"n_boot": n_boot, "seed": seed, "level": level, "n_lineages": len(lineages)})
    return out


def assess(y, p, lineage, *, min_per_class: int, n_boot: int = 1000, seed: int = 0,
           threshold: float = 0.5) -> dict:
    """Point estimates and intervals of one group of external isolates, or only its counts
    when it has fewer than ``min_per_class`` resistant or susceptible isolates."""
    y = np.asarray(y, dtype=int)
    n_r, n_s = int(y.sum()), int((y == 0).sum())
    out: dict = {"n": len(y), "n_resistant": n_r, "n_susceptible": n_s,
                 "assessed": n_r >= min_per_class and n_s >= min_per_class}
    if out["assessed"]:
        out.update(metrics(y, p, threshold=threshold))
        out["intervals"] = bootstrap(y, p, lineage, n_boot=n_boot, seed=seed, threshold=threshold)
    return out
