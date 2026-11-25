#!/usr/bin/env python3
"""The keys of the independent external isolates (CABBAGE) in the numbers file
(scripts/reports.py, _cabbage_numbers)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

pytestmark = pytest.mark.unit

ORGS = ["ecoli", "kpneumoniae"]
# (pair, tool, assessable, n, n_resistant, balanced accuracy)
COMPARISON = {
    "ecoli": [("ecoli__amp", "model", 1, 100, 40, 0.80), ("ecoli__amp", "amrfinderplus", 1, 100, 40, 0.70),
              ("ecoli__amp", "resfinder", 0, 100, 40, None),
              ("ecoli__cip", "model", 1, 100, 50, 0.90), ("ecoli__cip", "amrfinderplus", 1, 100, 50, 0.85),
              ("ecoli__cip", "resfinder", 1, 100, 50, 0.95),
              # five resistant isolates: below min_per_class, not scored
              ("ecoli__gen", "model", 1, 100, 5, 0.99), ("ecoli__gen", "amrfinderplus", 1, 100, 5, 0.99)],
    "kpneumoniae": [("kpneumoniae__mer", "model", 1, 60, 30, 0.70),
                    ("kpneumoniae__mer", "amrfinderplus", 1, 60, 30, 0.75),
                    ("kpneumoniae__mer", "resfinder", 1, 60, 30, 0.72)],
}
# lookup balanced accuracy per pair and rule (None: not assessed)
LOOKUP = {"ecoli": {"amp": {"upto_confirmed": 0.70, "upto_strong_novel": 0.82, "b1": 0.70},
                    "cip": {"upto_confirmed": 0.88, "upto_strong_novel": 0.88, "b1": 0.88},
                    "gen": {"upto_confirmed": None, "upto_strong_novel": None, "b1": None}},
          "kpneumoniae": {"mer": {"upto_confirmed": 0.60, "upto_strong_novel": 0.74, "b1": 0.65}}}


def _assessed(ba, seen=None):
    if ba is None:
        return {"assessed": False}
    return {"assessed": True, "balanced_accuracy": ba, "sensitivity": ba, "specificity": ba,
            "very_major_error_rate": 1 - ba, "major_error_rate": 1 - ba}


class CabbageTables:
    config: dict = {"cv": {"n_bootstrap": 200}, "cabbage": {"min_per_class": 10}}

    def get(self, tid, organism=None, antibiotic=None, **_):
        if tid == "cabbage_comparison" and organism in COMPARISON:
            return pd.DataFrame(
                [{"model_id": m, "tool": t, "assessable": a, "n": n, "n_resistant": r,
                  "balanced_accuracy": ba, "sensitivity": ba, "specificity": ba,
                  "very_major_error_rate": None if ba is None else 1 - ba,
                  "major_error_rate": None if ba is None else 1 - ba}
                 for m, t, a, n, r, ba in COMPARISON[organism]])
        if tid == "cabbage_predictions" and organism in COMPARISON:
            return pd.DataFrame({"biosample_id": ["S1", "S2", "S2"] if organism == "ecoli"
                                 else ["S3"]})
        return None

    def json(self, location, file, organism=None, antibiotic=None):
        if file == "search_check.json":
            return {"n_isolates": 12, "n_calls": 120, "n_same": 119}
        if file == "cabbage_lookup_selection.json":
            return {"primary_rule": "upto_strong_novel",
                    "balanced_accuracy": {"upto_confirmed": 0.77, "upto_strong_novel": 0.82}}
        if file == "cabbage_selection.json":
            return {"isolates_left": {"ecoli": {"assembly": 10}, "kpneumoniae": {"assembly": 5},
                                      "enterobacter_cloacae": {"assembly": 2}}}
        if file == "cabbage_prepare.json" and organism in ORGS:
            return {"n_fetched": 8 if organism == "ecoli" else 4, "n_qc_passed": 7}
        if file == "cabbage_lookup_metrics.json" and organism in LOOKUP:
            return {"pairs": {ab: {"score": {"all": {"assessed": ab != "gen",
                                                     "roc_auc": {"amp": 0.9, "cip": 0.7, "mer": 0.8}.get(ab)}},
                                   **{rule: {"all": _assessed(ba),
                                          "lineage_seen": _assessed(ba),
                                          "lineage_unseen": _assessed(None if ba is None
                                                                      else ba - 0.1)}
                                   for rule, ba in rules.items()}}
                              for ab, rules in LOOKUP[organism].items()}}
        if file == "cabbage_metrics.json" and organism == "ecoli":
            return {"pairs": {
                "amp": {"all": {"assessed": True, "balanced_accuracy": 0.8, "roc_auc": 0.90},
                        "lineage_seen": {"assessed": True, "balanced_accuracy": 0.82},
                        "lineage_unseen": {"assessed": False},
                        "internal_lineage_aware": {"roc_auc": 0.85}},
                "cip": {"all": {"assessed": True, "balanced_accuracy": 0.9, "roc_auc": 0.95},
                        "lineage_seen": {"assessed": True, "balanced_accuracy": 0.92},
                        "lineage_unseen": {"assessed": True, "balanced_accuracy": 0.80},
                        "internal_lineage_aware": {"roc_auc": 0.97}}}}
        return None


@pytest.fixture(scope="module")
def keys():
    spec = importlib.util.spec_from_file_location("amrtest_reports_cabbage",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = {}

    def put(key, value, table, column, description, digits="auto"):
        out[key] = value
    m._cabbage_numbers(CabbageTables(), ORGS, put)
    return out


def test_a_pair_is_scored_with_enough_isolates_of_both_classes(keys):
    assert keys["cabbage.n_pairs_scored"] == 3                 # ecoli__gen has 5 resistant
    assert keys["cabbage.model.n_pairs"] == 3
    assert keys["cabbage.resfinder.n_pairs"] == 2              # it cannot assess ampicillin
    assert keys["cabbage.lookup_b1.n_pairs"] == 3


def test_the_lookup_is_its_primary_rule(keys):
    assert keys["cabbage.lookup.primary_rule"] == "upto_strong_novel"
    assert keys["cabbage.lookup.median_ba"] == keys["cabbage.lookup_upto_strong_novel.median_ba"]


def test_pairs_are_compared_where_both_score(keys):
    k = "cabbage.compare.lookup__resfinder"
    assert keys[f"{k}.n_pairs"] == 2
    assert keys[f"{k}.n_a_better"] == 1 and keys[f"{k}.n_b_better"] == 1
    assert keys["cabbage.compare.lookup__lookup_b1.n_tied"] == 1   # ciprofloxacin
    assert keys["cabbage.common.n_pairs"] == 2


def test_isolate_flow_is_summed(keys):
    assert keys["cabbage.n_isolates_selected"] == 17
    assert keys["cabbage.n_isolates_fetched"] == 12
    assert keys["cabbage.n_isolates_qc_pass"] == 14
    assert keys["cabbage.n_isolates_predicted"] == 3


def test_auc_and_lineages(keys):
    assert keys["cabbage.model.median_roc_auc"] == pytest.approx(0.925)
    assert keys["cabbage.model.n_external_above_internal"] == 1
    assert keys["cabbage.lineage.model.n_pairs"] == 1          # amp has no unseen isolates
    assert keys["cabbage.lineage.model.median_diff"] == pytest.approx(-0.12)
    assert keys["cabbage.lineage.lookup.n_pairs"] == 3


def test_the_score_enters_as_its_roc_auc(keys):
    assert keys["cabbage.lookup_score.n_pairs"] == 3          # gentamicin is not assessed
    assert keys["cabbage.lookup_score.median_roc_auc"] == pytest.approx(0.8)
    assert keys["cabbage.lookup_score.min_roc_auc"] == pytest.approx(0.7)
    assert "cabbage.lookup_score.median_ba" not in keys


def test_the_genome_search_check_enters(keys):
    assert keys["search.check.n_isolates"] == 12
    assert keys["search.check.n_calls"] == 120 and keys["search.check.n_same"] == 119
