"""KANIT lookup on the external isolates (CABBAGE): the knowledge base used as a catalogue.

Added during the study (2026-11-13), next to the final models' predictions; the
rule was fixed before any lookup result was computed.

A marker is a graded candidate pattern of a model, of grade ``weak`` or better (primary,
allele-aware mode), with the direction of its prevalence difference: ``R`` when it is more
frequent in the resistant genomes, ``S`` when in the susceptible ones. An isolate is called
resistant by a rule when it carries at least one ``R`` marker of the rule; a pattern is
present in an isolate when at least half of its member unitigs are (as for the models,
lib/cabbage_predict.PRESENT_SHARE).

Rules:
  upto_<grade>  the R markers graded <grade> or better (confirmed > strong_novel >
                candidate > weak)
  b1            the R markers concordant with a curated determinant (CARD state b),
                whatever their grade: the catalogue part of the knowledge base alone
  score         secondary: R markers present minus S markers present, all grades up to
                weak; a continuous score, read at > 0

The primary rule is chosen on the training genomes, before the external isolates are
looked at: the upto_ rule with the highest balanced accuracy over the pooled counts of
every model (a tie goes to the stricter rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRADES = ("confirmed", "strong_novel", "candidate", "weak")      # strictest first
UPTO = tuple(f"upto_{g}" for g in GRADES)
RULES = UPTO + ("b1",)
SCORE = "score"


def markers(grades: pd.DataFrame, prevalence: pd.DataFrame) -> pd.DataFrame:
    """pattern_id, grade, card_state and direction of the markers of one model."""
    g = grades.loc[grades["grade"].isin(GRADES),
                   ["pattern_id", "grade", "allele_aware_card_state"]]
    g = g.rename(columns={"allele_aware_card_state": "card_state"})
    d = g.merge(prevalence[["pattern_id", "direction"]], on="pattern_id", how="left")
    if d["direction"].isna().any():
        raise ValueError("a marker has no prevalence direction")
    return d.astype({"pattern_id": np.int64}).reset_index(drop=True)


def rule_patterns(mk: pd.DataFrame, rule: str, direction: str = "R") -> list[int]:
    """The pattern ids a rule calls on, of one direction."""
    m = mk[mk["direction"] == direction]
    if rule == "b1":
        m = m[m["card_state"] == "b"]
    elif rule in UPTO:
        m = m[m["grade"].isin(GRADES[:UPTO.index(rule) + 1])]
    else:
        raise ValueError(f"unknown rule {rule}")
    return sorted(int(p) for p in m["pattern_id"])


def predict(presence: pd.DataFrame, mk: pd.DataFrame, rule: str) -> pd.Series:
    """0/1 per genome (column of ``presence``, pattern × genome) for one rule; a marker
    ``presence`` lacks is absent."""
    sub = presence.reindex(index=rule_patterns(mk, rule), fill_value=0)
    return (sub.sum(axis=0) > 0).astype(int)


def score(presence: pd.DataFrame, mk: pd.DataFrame) -> pd.Series:
    """R markers present minus S markers present, all grades up to weak. The counts are
    signed before the subtraction: presence is uint8, whose sums are unsigned, and a genome
    with more S than R markers wrapped round to about 2^64 (found 14.11.2026)."""
    r = presence.reindex(index=rule_patterns(mk, "upto_weak", "R"), fill_value=0).sum(axis=0)
    s = presence.reindex(index=rule_patterns(mk, "upto_weak", "S"), fill_value=0).sum(axis=0)
    return (r.astype(np.int64) - s.astype(np.int64)).astype(float)


def counts(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    return {"tp": int(((y == 1) & (pred == 1)).sum()), "fn": int(((y == 1) & (pred == 0)).sum()),
            "tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum())}


def balanced_accuracy(c: dict) -> float:
    r, s = c["tp"] + c["fn"], c["tn"] + c["fp"]
    if not r or not s:
        return float("nan")
    return (c["tp"] / r + c["tn"] / s) / 2


def pooled(per_model: list[dict]) -> dict:
    """{rule: summed counts} over models ({rule: counts} each)."""
    out: dict[str, dict[str, int]] = {}
    for m in per_model:
        for rule, c in m.items():
            acc = out.setdefault(rule, {"tp": 0, "fn": 0, "tn": 0, "fp": 0})
            for k in acc:
                acc[k] += int(c[k])
    return out


def select(pooled_counts: dict) -> str:
    """The upto_ rule with the highest pooled balanced accuracy; a tie goes to the stricter."""
    best, best_ba = None, -1.0
    for rule in UPTO:
        ba = balanced_accuracy(pooled_counts[rule]) if rule in pooled_counts else float("nan")
        if not np.isnan(ba) and ba > best_ba + 1e-12:
            best, best_ba = rule, ba
    if best is None:
        raise ValueError("no rule has training counts of both classes")
    return best
