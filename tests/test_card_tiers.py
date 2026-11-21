#!/usr/bin/env python3
"""The grades against CARD's RGI tier (scripts/lib/card_tiers.py)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import card_tiers as ct  # noqa: E402

pytestmark = pytest.mark.unit

# Genome 562.10010 would read 562.1001 as a float; the tier lookup must not lose it.
HITS = pd.DataFrame({
    "genome_id": ["562.10010", "562.2", "562.2", "562.3", "562.3", "562.4", "562.4"],
    "aro": ["3000873", "3000873", "3000873", "3002652", "3002652", "3000873", "3003000"],
    "cut_off": ["Perfect", "Perfect", "Strict", "Strict", "Loose", "Strict", "Perfect"]})
UNITIGS = pd.DataFrame({
    "pattern_id": [1, 1, 2, 3, 4, 5, 6],
    "located_genomes": ["562.10010", "562.10010", "562.3", "562.10010;562.4", "562.2",
                        "562.9", "562.10010"],
    "allele_aware_card_state": ["b", "b", "b", "card_hit_without_b", "b", "b",
                                "no_card_hit"],
    "allele_aware_card_aros": ["3000873", "3000873.0", "3002652", "3000873", "3000873",
                               "3000873", ""]})
PATTERNS = pd.DataFrame({
    "pattern_id": [1, 2, 3, 4, 5, 6],
    "allele_aware_grade": ["confirmed", "candidate", "weak", "confirmed", "candidate",
                           "strong_novel"],
    "allele_aware_card_state": ["b", "b", "card_hit_without_b", "b", "b", "no_card_hit"]})


def test_a_genome_with_both_tiers_of_an_aro_is_ambiguous():
    t = ct.pair_tiers(HITS)
    assert t[("562.10010", "3000873")] == "Perfect"
    assert t[("562.2", "3000873")] == ct.AMBIGUOUS
    assert t[("562.3", "3002652")] == "Strict"          # the Loose hit is not a tier here


def test_the_tier_of_a_pattern_comes_from_its_located_genomes():
    p = ct.pattern_tiers(UNITIGS, ct.pair_tiers(HITS)).set_index("pattern_id")
    # two unitigs on the same (genome, ARO) count once; the float ARO reads as text
    assert p.loc[1, "tier"] == "Perfect" and p.loc[1, "n_perfect"] == 1
    assert p.loc[2, "tier"] == "Strict"
    assert p.loc[3, "tier"] == "mixed" and (p.loc[3, "n_perfect"], p.loc[3, "n_strict"]) == (1, 1)
    assert pd.isna(p.loc[4, "tier"]) and p.loc[4, "n_ambiguous"] == 1
    assert pd.isna(p.loc[5, "tier"]) and p.loc[5, "n_no_hit"] == 1
    assert 6 not in p.index                              # no CARD hit, no tier


def test_a_candidate_is_grouped_by_its_card_state():
    m = ct.model_table(PATTERNS, UNITIGS, ct.pair_tiers(HITS)).set_index("pattern_id")
    assert m.loc[2, "group"] == "candidate_b" and m.loc[3, "group"] == "weak"
    assert ct.group("candidate", "card_hit_without_b") == "candidate_no_b"
    assert 6 not in m.index


def test_the_mantel_haenszel_estimate_matches_its_textbook_form():
    # checked against statsmodels' StratifiedTable (oddsratio_pooled, its confint and
    # test_null_odds(correction=False)); the last two strata lack a column and a row
    r = ct.mantel_haenszel([(10, 5, 4, 11), (6, 6, 3, 9), (2, 8, 1, 7), (3, 0, 2, 0),
                            (0, 0, 4, 4)])
    assert r["n_strata"] == 3
    assert r["odds_ratio"] == pytest.approx(3.5970149253731343)
    assert (r["ci_low"], r["ci_high"]) == (pytest.approx(1.2512201740045663),
                                           pytest.approx(10.340719117361253))
    assert r["p"] == pytest.approx(0.016345701000900492)


def test_no_stratum_no_estimate():
    r = ct.mantel_haenszel([(3, 0, 2, 0)])
    assert r == {"n_strata": 0, "odds_ratio": None, "ci_low": None, "ci_high": None, "p": None}


def test_the_comparison_counts_only_the_two_grades_with_b_and_a_tier():
    t = pd.DataFrame({
        "organism": ["a"] * 6 + ["b"] * 3,
        "group": ["confirmed", "confirmed", "candidate_b", "candidate_b", "candidate_no_b",
                  "confirmed", "confirmed", "candidate_b", "weak"],
        "tier": ["Perfect", "Strict", "Perfect", "mixed", "Perfect", None, "Perfect", "Strict",
                 "Strict"]})
    c = ct.compare(t)
    assert (c["n_confirmed"], c["n_candidate_b"]) == (3, 3)
    assert c["n_strata"] == 2


def _reports():
    spec = importlib.util.spec_from_file_location("amrtest_reports_tiers",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTables:
    def get(self, tid, organism=None, antibiotic=None, **read_csv):
        if tid == "rgi_hits":
            return HITS[read_csv["usecols"]] if organism == "ecoli" else None
        if (organism, antibiotic) != ("ecoli", "ampicillin"):
            return None
        return {"grades_patterns": PATTERNS, "grades_unitigs": UNITIGS}.get(tid)


def test_the_numbers_file_keys_the_comparison():
    r = _reports()
    table = r.card_tier_table(FakeTables(), [("ecoli", "ampicillin"), ("kpneumoniae", "meropenem")])
    rows = []
    r._card_tier_numbers(table, lambda key, value, *a, **k: rows.append((key, value)))
    got = dict(rows)
    assert got["card_tier.n_patterns"] == 3 and got["card_tier.n_patterns_without_tier"] == 2
    assert got["card_tier.n_models"] == 1 and got["card_tier.n_aros"] == 2
    assert (got["card_tier.n_confirmed_perfect"], got["card_tier.n_candidate_b_strict"],
            got["card_tier.n_weak_mixed"]) == (1, 1, 1)
    assert got["card_tier.share_perfect_confirmed"] == 1.0
    assert got["card_tier.ecoli.n_candidate_b"] == 1
    assert got["card_tier.n_pairs_ambiguous"] == 1 and got["card_tier.n_pairs_no_hit"] == 1
    assert got["card_tier.compare.n_organisms"] == 1
    # one confirmed Perfect, one candidate Strict: S = 0, so no odds ratio and no interval
    assert got["card_tier.compare.odds_ratio"] is None
