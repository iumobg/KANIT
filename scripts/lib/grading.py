"""Grading rule: the grade of every candidate pattern.

A pattern is graded from its CARD state (lib/card_layer.py) and the
statistical layers it passes: prevalence, MDA, CPSS and pyseer. ``RULES`` is the
rule's table in its order; the first matching row gives the grade. Member
unitigs take the grade of their pattern and keep their own CARD annotation.

Both CARD modes are graded: ``allele_aware`` (the primary mode) and ``homolog_only``
(its sensitivity analysis, in which variant and overexpression hits are CARD
hits without ``b``). The configured primary mode names the one reported as
``grade``; both are stored.

Each statistical layer writes one file per model, ``<layers_dir>/<layer>.csv``,
with a ``pattern_id`` and a boolean ``passes`` column. Every layer is computed
for every candidate, so a candidate missing from a layer stops the
grading.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from lib.card_layer import B1, HIT_NO_B, MODES, NO_HIT

CONFIRMED, STRONG_NOVEL, CANDIDATE, WEAK, NONE = (
    "confirmed", "strong_novel", "candidate", "weak", "none")
GRADES = (CONFIRMED, STRONG_NOVEL, CANDIDATE, WEAK, NONE)
LAYERS = ("prevalence", "mda", "cpss", "pyseer")
CARD_STATES = (B1, HIT_NO_B, NO_HIT)


def n_passed(passed: Mapping[str, bool]) -> int:
    return sum(bool(passed[layer]) for layer in LAYERS)


# (CARD states, condition on the passed layers, grade), in order.
# "CPSS and pyseer and at least one other" is CPSS and pyseer among >= 3 layers.
RULES: tuple[tuple[frozenset[str], Callable[[Mapping[str, bool]], bool], str], ...] = (
    (frozenset({B1}), lambda p: n_passed(p) >= 2, CONFIRMED),
    (frozenset({B1}), lambda p: n_passed(p) <= 1, CANDIDATE),
    (frozenset({HIT_NO_B}), lambda p: n_passed(p) >= 2, CANDIDATE),
    (frozenset({NO_HIT}), lambda p: bool(p["cpss"] and p["pyseer"]) and n_passed(p) >= 3,
     STRONG_NOVEL),
    (frozenset({NO_HIT}), lambda p: n_passed(p) >= 2, CANDIDATE),
    (frozenset({HIT_NO_B, NO_HIT}), lambda p: n_passed(p) == 1, WEAK),
    (frozenset({HIT_NO_B, NO_HIT}), lambda p: n_passed(p) == 0, NONE),
)


def grade(card_state: str, passed: Mapping[str, bool]) -> str:
    """Grade of one pattern: the first row of the table that matches."""
    if card_state not in CARD_STATES:
        raise ValueError(f"unknown CARD state {card_state!r}")
    for states, condition, g in RULES:
        if card_state in states and condition(passed):
            return g
    raise AssertionError(f"no grading row matches {card_state!r}, {dict(passed)}")


def read_layer(path, layer: str) -> pd.Series:
    """{pattern_id: passes} of one layer file."""
    d = pd.read_csv(path)
    missing = sorted({"pattern_id", "passes"} - set(d.columns))
    if missing:
        raise ValueError(f"{path}: columns missing: {missing}")
    if d["pattern_id"].duplicated().any():
        raise ValueError(f"{path}: a pattern appears more than once")
    passes = d["passes"]
    if passes.dtype != bool:
        if not passes.isin([0, 1]).all():
            raise ValueError(f"{path}: passes must be true or false for every pattern")
        passes = passes.astype(bool)
    return pd.Series(passes.to_numpy(), index=d["pattern_id"].astype(int).to_numpy(), name=layer)


def grade_patterns(card_patterns: pd.DataFrame, layers: Mapping[str, pd.Series],
                   primary_mode: str) -> pd.DataFrame:
    """One row per candidate pattern: the layers it passes and, in both modes, its
    CARD state, the reasons of a CARD hit without b, and its grade; ``grade`` is
    the grade under ``primary_mode``."""
    if primary_mode not in MODES:
        raise ValueError(f"primary mode must be one of {MODES}, not {primary_mode!r}")
    if set(layers) != set(LAYERS):
        raise ValueError(f"layers must be {LAYERS}, got {sorted(layers)}")
    out = card_patterns[["pattern_id", "n_members"]].copy()
    pids = out["pattern_id"].astype(int).to_numpy()
    for layer in LAYERS:
        s = layers[layer]
        absent = sorted(set(pids.tolist()) - set(s.index.tolist()))
        if absent:
            raise ValueError(f"layer {layer} has no result for {len(absent)} candidate "
                             f"pattern(s), e.g. {absent[:5]}")
        out[layer] = s.reindex(pids).to_numpy().astype(bool)
    passed = out[list(LAYERS)].to_dict("records")
    out["n_layers"] = [n_passed(p) for p in passed]
    out["layers_passed"] = [";".join(x for x in LAYERS if p[x]) for p in passed]
    for mode in MODES:
        states = card_patterns[f"{mode}_state"].tolist()
        out[f"{mode}_card_state"] = states
        out[f"{mode}_card_reasons"] = card_patterns[f"{mode}_reasons"].fillna("").tolist()
        out[f"{mode}_grade"] = [grade(s, p) for s, p in zip(states, passed, strict=True)]
    out["grade"] = out[f"{primary_mode}_grade"]
    return out


def grade_unitigs(card_unitigs: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Every member unitig with its own CARD annotation and its pattern's grades."""
    own = {f"{mode}_{x}": f"{mode}_card_{x}" for mode in MODES
           for x in ("state", "reasons", "aros", "n_b")}
    grades = graded[["pattern_id", *(f"{mode}_grade" for mode in MODES), "grade"]]
    u = card_unitigs.rename(columns=own).merge(grades, on="pattern_id", how="left",
                                               validate="many_to_one")
    if u["grade"].isna().any():
        raise ValueError("a unitig belongs to a pattern without a grade")
    return u


def summary(graded: pd.DataFrame, primary_mode: str) -> dict:
    return {
        "primary_mode": primary_mode, "n_patterns": len(graded),
        "patterns_by_grade": {mode: {g: int((graded[f"{mode}_grade"] == g).sum()) for g in GRADES}
                              for mode in MODES},
        "patterns_passing": {layer: int(graded[layer].sum()) for layer in LAYERS},
    }
