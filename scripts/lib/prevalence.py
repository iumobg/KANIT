"""Prevalence layer.

For every candidate pattern: its prevalence among the resistant and among the
susceptible genomes of the model, Δ = prev(R) − prev(S) with its direction, and
a two-sided Fisher exact test, Benjamini–Hochberg-adjusted over the model's
candidate patterns. The layer passes when |Δ| ≥ ``min_delta`` and q < ``alpha``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact


def prevalence_layer(mm, patterns, *, min_delta: float = 0.10,
                     alpha: float = 0.05) -> pd.DataFrame:
    """One row per pattern: counts, prevalences, Δ, direction, p, q and passes."""
    patterns = np.asarray(patterns, dtype=np.int64)
    resistant = mm.labels.astype(bool)
    n_r, n_s = int(resistant.sum()), int((~resistant).sum())
    if not n_r or not n_s:
        raise ValueError("the model needs resistant and susceptible genomes")
    x = mm.columns(patterns)
    a = x[resistant].sum(axis=0, dtype=np.int64)
    c = x[~resistant].sum(axis=0, dtype=np.int64)
    p = np.array([fisher_exact([[ai, n_r - ai], [ci, n_s - ci]])[1]
                  for ai, ci in zip(a.tolist(), c.tolist(), strict=True)], dtype=float)
    q = false_discovery_control(p, method="bh") if p.size else p
    delta = a / n_r - c / n_s
    return pd.DataFrame({
        "pattern_id": patterns, "n_resistant": n_r, "n_susceptible": n_s,
        "present_resistant": a, "present_susceptible": c,
        "prev_resistant": a / n_r, "prev_susceptible": c / n_s, "delta": delta,
        "direction": np.where(delta > 0, "R", np.where(delta < 0, "S", "none")),
        "fisher_p": p, "q": q,
        "passes": (np.abs(delta) >= min_delta - 1e-12) & (q < alpha),   # Δ = 0.1 exactly passes
    })
