#!/usr/bin/env python3
"""Binary concordance / clinical-error metrics for external validation (M13).

Pure, dependency-light functions (scipy only for p-values, with a stdlib
fallback) used by ``lib.external`` to compare, on the same set of
genomes, any two of: our unitig model's calls, AMRFinderPlus/ResFinder genotypic
calls, and the EUCAST/CLSI phenotype (ground truth).

Convention: label 1 = **resistant** (the positive/clinically-important class),
0 = susceptible. ``y_true`` is the reference (usually the phenotype); ``y_pred``
is the predictor being graded (a tool or the model).

FDA/CLSI genotype-vs-phenotype error bands: Major Error (ME) =
predict R when truly S (false resistant), band ME ≤ 3%; Very Major Error (VME) =
predict S when truly R (missed resistance — the dangerous one), band ≤ 1.5–7.5%.
"""

from __future__ import annotations

import random
from statistics import fmean, median

VME_BAND = 0.075   # FDA upper bound for Very Major Error rate (1.5–7.5%)
ME_BAND = 0.03     # FDA upper bound for Major Error rate


def _pairs(y_true, y_pred):
    """Zip to aligned int pairs, dropping any position where either side is None
    (a genome the tool/phenotype has no call for)."""
    out = []
    for t, p in zip(y_true, y_pred, strict=True):
        if t is None or p is None:
            continue
        out.append((int(t), int(p)))
    return out


def confusion(y_true, y_pred):
    """2×2 counts with 1=resistant as positive. Returns dict TP/FP/TN/FN/n."""
    tp = fp = tn = fn = 0
    for t, p in _pairs(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "n": tp + fp + tn + fn}


def _safe_div(a, b):
    return (a / b) if b else None


def sensitivity(cm):
    """TP / (TP+FN) — recall of resistance."""
    return _safe_div(cm["TP"], cm["TP"] + cm["FN"])


def specificity(cm):
    """TN / (TN+FP)."""
    return _safe_div(cm["TN"], cm["TN"] + cm["FP"])


def balanced_accuracy(cm):
    se, sp = sensitivity(cm), specificity(cm)
    return (se + sp) / 2 if (se is not None and sp is not None) else None


def fda_errors(cm):
    """Major-error and very-major-error rates + FDA-band pass flags.

    ME = FP / (all truly susceptible); VME = FN / (all truly resistant)."""
    me = _safe_div(cm["FP"], cm["TN"] + cm["FP"])
    vme = _safe_div(cm["FN"], cm["TP"] + cm["FN"])
    return {
        "major_error_rate": me,
        "very_major_error_rate": vme,
        "me_within_fda_band": (me is not None and me <= ME_BAND),
        "vme_within_fda_band": (vme is not None and vme <= VME_BAND),
    }


def cohen_kappa(y_true, y_pred):
    """Cohen's κ for two binary raters (chance-corrected agreement)."""
    cm = confusion(y_true, y_pred)
    n = cm["n"]
    if not n:
        return None
    po = (cm["TP"] + cm["TN"]) / n
    p_pos = (cm["TP"] + cm["FN"]) / n * (cm["TP"] + cm["FP"]) / n
    p_neg = (cm["TN"] + cm["FP"]) / n * (cm["TN"] + cm["FN"]) / n
    pe = p_pos + p_neg
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def mcnemar(y_a, y_b):
    """McNemar's test of marginal homogeneity between two predictors on the same
    samples. Discordant cells b=(A=R,B=S), c=(A=S,B=R). Uses an exact two-sided
    binomial p when b+c is small (<25), else the continuity-corrected χ² approx.
    Returns {b, c, statistic, p_value} (statistic is None for the exact branch)."""
    b = c = 0
    for a, d in _pairs(y_a, y_b):
        if a == 1 and d == 0:
            b += 1
        elif a == 0 and d == 1:
            c += 1
    nd = b + c
    if nd == 0:
        return {"b": 0, "c": 0, "statistic": None, "p_value": 1.0}
    if nd < 25:
        p = _binom_two_sided(min(b, c), nd)
        return {"b": b, "c": c, "statistic": None, "p_value": p}
    stat = (abs(b - c) - 1) ** 2 / nd
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, 1))
    except Exception:
        p = _chi2_sf_1df(stat)
    return {"b": b, "c": c, "statistic": stat, "p_value": p}


def sign_test(differences):
    """Two-sided sign test on paired differences (ours minus theirs).

    One pair per organism–antibiotic model: the difference of a scalar score,
    usually balanced accuracy on the genomes both predictors can call. Ties carry
    no sign and are dropped, which is the standard convention — a pair where both
    score the same says nothing about which is better. ``n_tied`` is returned so
    the dropped pairs stay visible, because counting a tie against one side
    inflates the p-value.

    Returns {n_pairs, n_positive, n_negative, n_tied, p_value, p_tie_to_ours,
    p_tie_to_theirs}; ``p_value`` is None when every pair is tied. The last two
    give the p-value when the ties are counted for one side instead of dropped,
    so a reader can see how much the result turns on that choice. With no tie the
    three agree.
    """
    pos = neg = tied = 0
    for d in differences:
        if d is None:
            continue
        if d > 0:
            pos += 1
        elif d < 0:
            neg += 1
        else:
            tied += 1
    n = pos + neg
    whole = n + tied
    return {"n_pairs": whole, "n_positive": pos, "n_negative": neg, "n_tied": tied,
            "p_value": _binom_two_sided(min(pos, neg), n) if n else None,
            "p_tie_to_ours": _binom_two_sided(min(pos + tied, neg), whole) if whole else None,
            "p_tie_to_theirs": _binom_two_sided(min(pos, neg + tied), whole) if whole else None}


def paired_difference(differences, clusters=None, n_boot=1000, seed=0):
    """Median paired difference with a 95% percentile bootstrap interval.

    ``clusters`` names the unit drawn with replacement — here the organism. The
    models of one organism share genomes and lineages, so resampling the pairs
    themselves would count them as independent evidence and narrow the interval.
    Drawing clusters instead follows the internal evaluation, which already reports cluster-aware
    intervals and a paired interval for the arm difference. Pass None to draw the
    pairs, which is only right when they are independent.

    The sign test answers whether one side wins more often; this answers by how
    much, which is what a claim of comparable accuracy needs.

    Returns {n, n_clusters, median, mean, ci_low, ci_high}; the interval is None
    when there is nothing to resample."""
    d = [float(x) for x in differences if x is not None]
    if not d:
        return {"n": 0, "n_clusters": 0, "median": None, "mean": None,
                "ci_low": None, "ci_high": None}
    if clusters is None:
        groups = [[x] for x in d]
    else:
        by: dict = {}
        for x, c in zip(d, clusters, strict=True):
            by.setdefault(c, []).append(float(x))
        groups = list(by.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pooled: list[float] = []
        for _ in groups:
            pooled.extend(groups[rng.randrange(len(groups))])
        if pooled:
            draws.append(median(pooled))
    draws.sort()
    lo = hi = None
    if len(draws) >= 2:
        lo = draws[int(0.025 * (len(draws) - 1))]
        hi = draws[int(round(0.975 * (len(draws) - 1)))]
    return {"n": len(d), "n_clusters": len(groups), "median": median(d), "mean": fmean(d),
            "ci_low": lo, "ci_high": hi}


def signed_rank_p(differences):
    """Two-sided Wilcoxon signed-rank p on paired differences, zeros split.

    Reported only as a sensitivity analysis. It uses the size of each difference
    and so has more power than the sign test, but it treats the pairs as
    independent; where several pairs come from one organism they are not, and
    the p-value is then optimistic. ``paired_difference`` with clusters is the
    measure to read. Returns None without scipy or without a signed pair."""
    d = [float(x) for x in differences if x is not None]
    if not any(x != 0 for x in d):
        return None
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    return float(wilcoxon(d, zero_method="zsplit").pvalue)


def _binom_two_sided(k, n, prob=0.5):
    """Exact two-sided binomial p for k successes in n at p=0.5 (McNemar exact)."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, prob, alternative="two-sided").pvalue)
    except Exception:
        from math import comb
        # two-sided at p=0.5 is symmetric: 2 * P(X <= k), capped at 1.
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)


def _chi2_sf_1df(x):
    """Survival function of χ²₁ via erfc (stdlib fallback, no scipy)."""
    import math
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def score_pair(y_true, y_pred):
    """Full metric bundle grading ``y_pred`` against reference ``y_true``."""
    cm = confusion(y_true, y_pred)
    out = {
        "n": cm["n"], "confusion": cm,
        "sensitivity": sensitivity(cm), "specificity": specificity(cm),
        "balanced_accuracy": balanced_accuracy(cm),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
    }
    out.update(fda_errors(cm))
    return out
