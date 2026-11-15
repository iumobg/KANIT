#!/usr/bin/env python3
"""The aggregate keys of the numbers file (scripts/reports.py, numbers())."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.unit

DECISIONS = pd.DataFrame({
    "organism": ["ecoli", "ecoli", "kpneumoniae", "enterobacter_cloacae"],
    "antibiotic": ["ampicillin", "ciprofloxacin", "meropenem", "amikacin"],
    "drug_class": ["beta_lactams", "quinolones", "beta_lactams", "aminoglycosides"],
    "decision": ["included", "included", "included", "excluded"],
    "minority": [100, 200, 300, 50],
})
# Genomes that pass QC, per organism; the panel's own rule never sees these.
QC = {"ecoli": [True] * 10 + [False, False], "kpneumoniae": [True] * 5}
# Two models share genomes, so the union is smaller than the sum (3 + 3 + 2 = 8).
GENOMES = {("ecoli", "ampicillin"): ["g1", "g2", "g3"],
           ("ecoli", "ciprofloxacin"): ["g2", "g3", "g4"],
           ("kpneumoniae", "meropenem"): ["k1", "k2"]}
# The grade of the primary mode, per pattern: three of the six are weak or none.
PATTERNS = {("ecoli", "ampicillin"): ["weak", "confirmed"],
            ("ecoli", "ciprofloxacin"): ["none", "weak", "candidate"],
            ("kpneumoniae", "meropenem"): ["strong_novel"]}
# The evidence a pattern carries, in the order of PATTERNS. The homolog-only grades are
# chosen so the totals alone cannot say which pattern moved where: confirmed -1,
# candidate +1 fits both "confirmed -> candidate" and a pair of offsetting moves.
EVIDENCE = {
    ("ecoli", "ampicillin"): {
        "prevalence": [True, True], "mda": [False, True], "cpss": [False, True],
        "pyseer": [False, True],
        "allele_aware_card_state": ["no_card_hit", "b"],
        "allele_aware_card_reasons": ["", ""],
        "homolog_only_card_state": ["no_card_hit", "no_card_hit"],
        "homolog_only_grade": ["weak", "candidate"],
        "gain_rank": [1, 2], "cpss_stable": [False, True]},
    ("ecoli", "ciprofloxacin"): {
        "prevalence": [False, True, True], "mda": [False, False, True],
        "cpss": [False, False, False], "pyseer": [False, False, True],
        "allele_aware_card_state": ["no_card_hit", "no_card_hit", "card_hit_without_b"],
        "allele_aware_card_reasons": ["", "", "co_carried_known_gene"],
        "homolog_only_card_state": ["no_card_hit", "no_card_hit", "card_hit_without_b"],
        "homolog_only_grade": ["none", "none", "candidate"],
        "gain_rank": [1, 2, None], "cpss_stable": [False, False, False]},
    ("kpneumoniae", "meropenem"): {
        "prevalence": [True], "mda": [True], "cpss": [True], "pyseer": [True],
        "allele_aware_card_state": ["no_card_hit"], "allele_aware_card_reasons": [""],
        "homolog_only_card_state": ["no_card_hit"], "homolog_only_grade": ["strong_novel"],
        "gain_rank": [1], "cpss_stable": [True]},
}

# The sign of each pattern's prevalence difference, in the order of PATTERNS; the
# strong_novel pattern is the one susceptible-direction case that must be counted twice.
DIRECTION = {("ecoli", "ampicillin"): ["R", "R"],
             ("ecoli", "ciprofloxacin"): ["S", "R", "R"],
             ("kpneumoniae", "meropenem"): ["S"]}


# One win, one tie and one loss against the tool, spread over two organisms.
COMPARISON = {
    "ecoli": [("ecoli__ampicillin", "model", 1, 0.9),
              ("ecoli__ampicillin", "amrfinderplus", 1, 0.8),
              ("ecoli__ciprofloxacin", "model", 1, 0.8),
              ("ecoli__ciprofloxacin", "amrfinderplus", 1, 0.8)],
    "kpneumoniae": [("kpneumoniae__meropenem", "model", 1, 0.7),
                    ("kpneumoniae__meropenem", "amrfinderplus", 1, 0.9)],
}


class FakeTables:
    """Only the tables numbers() reads; everything else is absent."""

    config: dict = {"cv": {"n_bootstrap": 200}}

    def get(self, tid, organism=None, antibiotic=None, **_):
        if tid == "panel_decisions":
            return DECISIONS
        key = (organism, antibiotic)
        if tid == "model_genomes" and key in GENOMES:
            ids = GENOMES[key]
            return pd.DataFrame({"Genome ID": ids, "label": [1] * len(ids)})
        if tid == "genome_qc" and organism in QC:
            return pd.DataFrame({"pass_overall": QC[organism]})
        if tid == "external_comparison" and organism in COMPARISON:
            return pd.DataFrame(COMPARISON[organism],
                                columns=["model_id", "tool", "assessable",
                                         "balanced_accuracy"])
        if tid == "grades_patterns" and key in PATTERNS:
            g = PATTERNS[key]
            d = {"pattern_id": range(len(g)), "allele_aware_grade": g, "grade": g}
            e = EVIDENCE[key]
            d.update({k: v for k, v in e.items() if k not in ("gain_rank", "cpss_stable")})
            return pd.DataFrame(d)
        if tid == "prevalence_layer" and key in DIRECTION:
            return pd.DataFrame({"pattern_id": range(len(DIRECTION[key])),
                                 "direction": DIRECTION[key]})
        if tid == "candidates" and key in PATTERNS:
            e = EVIDENCE[key]
            return pd.DataFrame({"pattern_id": range(len(PATTERNS[key])),
                                 "gain_rank": e["gain_rank"], "cpss_stable": e["cpss_stable"]})
        return None

    def json(self, location, file, organism=None, antibiotic=None):
        return None


class PermutedTables(FakeTables):
    """The same run with its label permutation table (q per model)."""

    config: dict = {"cv": {"n_bootstrap": 200}, "label_permutation": {"alpha": 0.05}}

    def __init__(self, q: dict[str, float]):
        self.q = q

    def get(self, tid, organism=None, antibiotic=None, **_):
        if tid == "label_permutation_models":
            return pd.DataFrame({"organism": [m.split("__")[0] for m in self.q],
                                 "antibiotic": [m.split("__")[1] for m in self.q],
                                 "q": list(self.q.values())})
        return super().get(tid, organism, antibiotic)


def _numbers(tables=None):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amrtest_reports",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    df = m.numbers(tables or FakeTables(), ["ecoli", "kpneumoniae", "enterobacter_cloacae"])
    return df.set_index("key")["value"].to_dict()


def test_panel_aggregates_count_only_included_pairs():
    """An organism whose every pair is excluded has no model, so it is not counted."""
    n = _numbers()
    assert n["panel.n_pairs"] == 4
    assert n["panel.n_included"] == 3
    assert n["panel.n_organisms_with_models"] == 2
    assert n["panel.n_antibiotics_included"] == 3
    assert n["panel.n_drug_classes_included"] == 2


def test_genomes_in_models_is_a_union_and_patterns_a_sum():
    """Genomes are shared between models; candidate patterns are model-specific."""
    n = _numbers()
    assert n["all.n_genomes_in_models"] == 6
    assert sum(n[f"{o}__{a}.n_genomes"] for o, a in GENOMES) == 8
    assert n["all.n_candidate_patterns"] == 6


def test_weak_or_none_share_is_measured_on_the_reported_rule():
    """Three of the six candidate patterns clear no evidence threshold."""
    n = _numbers()
    assert n["all.weak_or_none_share"] == 0.5


def test_the_share_is_absent_without_the_reported_grade():
    """An output carrying only the per-mode columns gives no share, not a zero."""
    class PerModeOnly(FakeTables):
        def get(self, tid, organism=None, antibiotic=None, **_):
            out = super().get(tid, organism, antibiotic)
            if tid == "grades_patterns" and out is not None:
                return out.drop(columns=["grade"])
            return out

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amrtest_reports3",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    df = m.numbers(PerModeOnly(), ["ecoli", "kpneumoniae"])
    keys = set(df["key"])
    assert "all.weak_or_none_share" not in keys
    assert "all.n_candidate_patterns" in keys


def test_aggregates_are_absent_without_models():
    """Nothing is reported when no model has an output yet."""
    class Empty(FakeTables):
        def get(self, tid, organism=None, antibiotic=None, **_):
            return DECISIONS if tid == "panel_decisions" else None

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amrtest_reports2",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    keys = set(m.numbers(Empty(), ["ecoli"])["key"])
    assert "all.n_genomes_in_models" not in keys
    assert "all.n_candidate_patterns" not in keys
    assert "all.weak_or_none_share" not in keys
    assert "panel.n_organisms_with_models" in keys


def test_the_sign_test_counts_pairs_but_tests_only_the_signed_ones():
    """The assessable count keeps the tie; the sign test drops it."""
    n = _numbers()
    assert n["compare.amrfinderplus_n_assessable"] == 3
    assert n["compare.amrfinderplus_n_model_better"] == 1
    assert n["compare.amrfinderplus_n_tool_better"] == 1
    assert n["compare.amrfinderplus_n_tied"] == 1
    assert n["compare.amrfinderplus_p"] == 1.0
    assert n["compare.amrfinderplus_median_ba_model"] == 0.8
    assert n["compare.amrfinderplus_median_ba_tool"] == 0.8
    # Two organisms carry the three pairs, so the interval is drawn over two clusters.
    assert n["compare.amrfinderplus_n_organisms"] == 2
    assert n["compare.amrfinderplus_median_diff"] == 0.0


def test_a_very_small_p_is_not_rounded_to_zero():
    """Four significant digits, so a p far below the fourth decimal survives."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("amrtest_reports4",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._four_significant(8.1861e-18) == 8.186e-18
    assert m._four_significant(0.058058) == 0.05806
    assert m._four_significant(0) == 0
    assert m._four_significant(None) is None


def test_a_p_value_keeps_its_four_significant_digits_in_the_file(monkeypatch):
    """The numbers file must not round it again to four decimals: 0.058058 is written
    0.05806, not 0.0581."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from lib import concordance
    monkeypatch.setattr(concordance, "signed_rank_p", lambda d: 0.058058)
    assert _numbers()["compare.amrfinderplus_p_signed_rank"] == 0.05806


def test_grade_totals_sum_to_the_candidate_patterns():
    """The five grades partition the candidate set, so they add up to its size."""
    n = _numbers()
    grades = ("confirmed", "strong_novel", "candidate", "weak", "none")
    assert [n[f"all.n_{g}"] for g in grades] == [1, 1, 1, 2, 1]
    assert sum(n[f"all.n_{g}"] for g in grades) == n["all.n_candidate_patterns"]
    assert n["all.weak_or_none_share"] == 0.5


def test_per_organism_panel_counts_and_the_largest_minority():
    """An organism whose pairs all fall short keeps its counts and its minority."""
    n = _numbers()
    assert n["ecoli.n_pairs_assessed"] == 2
    assert n["ecoli.n_pairs_included"] == 2
    assert n["ecoli.max_minority"] == 200
    assert n["enterobacter_cloacae.n_pairs_assessed"] == 1
    assert n["enterobacter_cloacae.n_pairs_included"] == 0
    assert n["enterobacter_cloacae.max_minority"] == 50


class DesignTables(FakeTables):
    """The same run with each model's fold design (cv_design.json)."""

    SHARE = {("ecoli", "ampicillin"): 0.1, ("ecoli", "ciprofloxacin"): 0.4,
             ("enterobacter_cloacae", "amikacin"): 0.9}

    def json(self, location, file, organism=None, antibiotic=None):
        if location == "cv_dir" and (organism, antibiotic) in self.SHARE:
            return {"largest_lineage_share": self.SHARE[(organism, antibiotic)]}
        return None


def test_the_largest_lineage_share_is_over_the_included_models():
    """Median and largest over an organism's models; an excluded pair's design does not
    count, and an organism without designs has no key."""
    n = _numbers(DesignTables())
    assert n["ecoli.model_largest_lineage_share_median"] == 0.25
    assert n["ecoli.model_largest_lineage_share_max"] == 0.4
    assert "enterobacter_cloacae.model_largest_lineage_share_max" not in n
    assert "kpneumoniae.model_largest_lineage_share_median" not in n


def test_qc_genomes_are_summed_over_organisms():
    """Each organism has its own genomes, so the total is a sum, not a union."""
    n = _numbers()
    assert n["ecoli.n_genomes_qc_pass"] == 10
    assert n["all.n_genomes_qc_pass"] == 15


def test_layer_counts_and_shares_are_over_all_candidates():
    """Six candidates; the shares divide by six, not by the models. A share keeps six
    decimals so it can be rendered as a percentage with one."""
    n = _numbers()
    assert n["all.n_pass_prevalence"] == 5
    assert n["all.share_pass_prevalence"] == round(5 / 6, 6)
    assert n["all.n_pass_cpss"] == 2
    assert n["all.n_layers_0"] == 1
    assert n["all.n_layers_1"] == 2
    assert n["all.n_layers_4"] == 2


def test_every_mode_transition_gets_its_own_key():
    """The totals hide offsetting moves: confirmed -1 and candidate +1 fits more than
    one reading, so the pair of grades a pattern moves between must be keyed."""
    n = _numbers()
    assert n["all.n_grade_differs_between_modes"] == 2
    assert n["all.n_mode_change_confirmed_to_candidate"] == 1
    assert n["all.n_mode_change_weak_to_none"] == 1
    assert "all.n_mode_change_confirmed_to_weak" not in n


def test_the_h2_share_is_a_median_over_models_not_a_pooled_ratio():
    """Two models have a stable pattern: one reaches b = 1, the other does not. The
    pooled ratio is 1/2; the median over models is also 0.5 here, and the model with no
    stable pattern must not enter either."""
    n = _numbers()
    assert n["all.n_stable_patterns"] == 2
    assert n["all.n_stable_b"] == 1
    assert n["all.n_models_with_stable_patterns"] == 2
    assert n["all.n_models_no_stable_b"] == 1
    assert n["all.stable_b_median_share.allele_aware"] == 0.5


def test_the_top_gain_subset_excludes_patterns_without_a_gain_rank():
    """A pattern that reaches the candidate set only through stability selection has no
    gain rank, so the gain-ranking numbers must not count it."""
    n = _numbers()
    assert n["all.n_top_gain_patterns"] == 5
    assert n["all.top_gain_n_b"] == 1
    assert n["all.n_models_top_feature_b"] == 0
    assert "all.top_gain_n_reason_co_carried_known_gene" not in n


Q = {"ecoli__ampicillin": 0.01, "ecoli__ciprofloxacin": 0.02, "kpneumoniae__meropenem": 0.3}


def test_the_flagged_model_leaves_the_grade_distribution():
    """Protocol §14 item 9: meropenem is flagged; the five E. coli candidates remain, three
    of them weak or none. Its own layers are keyed; its grades already are."""
    n = _numbers(PermutedTables(Q))
    assert n["label_permutation.n_flagged"] == 1 and n["label_permutation.n_significant"] == 2
    assert n["lp_sensitivity.n_models"] == 2 and n["lp_sensitivity.n_candidate_patterns"] == 5
    assert n["lp_sensitivity.n_strong_novel"] == 0 and n["lp_sensitivity.n_weak"] == 2
    assert n["lp_sensitivity.weak_or_none_share"] == 0.6
    assert n["lp_sensitivity.share_pass_mda"] == 0.4
    assert n["kpneumoniae__meropenem.n_pass_mda"] == 1
    assert n["kpneumoniae__meropenem.n_pass_cpss"] == 1
    assert "ecoli__ampicillin.n_pass_mda" not in n
    assert n["all.n_strong_novel"] == 1                  # the primary numbers keep it


def test_no_sensitivity_keys_without_every_q_or_a_flag():
    """A model without its q leaves the correction open; with no flag there is nothing
    to recompute."""
    partial = {k: v for k, v in Q.items() if k != "ecoli__ampicillin"}
    assert not any(k.startswith("lp_sensitivity.") for k in _numbers(PermutedTables(partial)))
    n = _numbers(PermutedTables({k: 0.01 for k in Q}))
    assert n["label_permutation.n_flagged"] == 0
    assert not any(k.startswith("lp_sensitivity.") or (".n_pass_" in k and not k.startswith("all."))
                   for k in n)


H7 = {"n_expected": 5, "n_recovered": 4, "share_confirmed": 0.75, "supported_primary": True,
      "secondary_table": [[3, 1], [2, 6]], "secondary_p": 0.0123456, "supported_secondary": True}
SUMMARY = {"H1": {"n_models": 3, "n_with_stable": 2, "supported": False},
           "H2": {"n_defined": 2, "median": 0.75, "median_homolog_only": 0.5, "supported": True},
           "H3": {"n_within": 1, "n_cross": 2, "median_within": 1.0, "median_cross": 0.0,
                  "p": None, "supported": False},
           "H7": {"0.1": H7, "0.05": {**H7, "n_expected": 6}, "0.2": {"n_expected": 0,
                                                                      "supported_primary": False}},
           "H6": {"n_acquired": 0, "n_mutational": 2, "n_mixed": 1,
                  "p_acquired_gt_mutational": 8.1861e-18}}


class HypothesisTables(FakeTables):
    """The same run after the hypothesis tests."""

    def __init__(self, sensitivity: dict):
        self.summary = {**SUMMARY, "reference_missing": ["kpneumoniae"],
                        "sensitivity_label_permutation": sensitivity}

    def json(self, location, file, organism=None, antibiotic=None):
        return self.summary if file == "hypotheses/hypotheses.json" else None


def test_the_hypotheses_are_keyed_by_what_they_measure():
    """The protocol's H3 and H7 are the article's H4 and H3: no key carries the number.
    A verdict is 1 or 0; a statistic the test could not compute has no key."""
    n = _numbers(HypothesisTables({"status": "no model flagged: the primary analysis",
                                   "excluded_models": []}))
    hyp = {k: v for k, v in n.items() if k.startswith("hyp.")}
    assert not any(f"h{i}" in k.lower() for k in hyp for i in range(1, 8))
    assert n["hyp.stable.supported"] == 0 and n["hyp.curated.supported"] == 1
    assert n["hyp.curated.median_share"] == 0.75 and n["hyp.class_overlap.n_within"] == 1
    assert "hyp.class_overlap.p" not in n
    assert n["hyp.acquired_vs_mutational.p"] == 8.186e-18      # four significant digits
    assert n["hyp.expected.share_confirmed.thr10"] == 0.75
    assert n["hyp.expected.n_expected.thr5"] == 6 and n["hyp.expected.n_expected.thr20"] == 0
    assert "hyp.expected.share_confirmed.thr20" not in n
    assert n["hyp.expected.secondary_expected_confirmed.thr10"] == 3
    assert n["hyp.expected.secondary_other_not_confirmed.thr10"] == 6
    assert n["hyp.expected.secondary_p.thr10"] == 0.01235
    assert n["hyp.acquired_vs_mutational.n_mutational"] == 2
    assert n["hyp.n_organisms_without_reference"] == 1
    assert not any(k.startswith("lp_sensitivity.hyp.") for k in n)


def test_the_hypotheses_without_the_flagged_models_get_their_own_keys():
    n = _numbers(HypothesisTables({"status": "computed",
                                   "excluded_models": ["kpneumoniae__meropenem"],
                                   **SUMMARY, "H2": {**SUMMARY["H2"], "median": 0.5}}))
    assert n["lp_sensitivity.hyp.n_models_excluded"] == 1
    assert n["lp_sensitivity.hyp.curated.median_share"] == 0.5
    assert n["hyp.curated.median_share"] == 0.75
    assert n["lp_sensitivity.hyp.expected.share_confirmed.thr10"] == 0.75


# Unitigs of the two paths: (state, AROs). Only b = 1 counts; a CARD hit without b does not.
UNITIGS = {("ecoli", "ampicillin"): ([("b", "3000001"), ("b", "3000002;3000003"),
                                     ("card_hit_without_b", "3000009")],
                                    [("b", "3000003"), ("b", "3000004"), ("no_card_hit", None)]),
           ("ecoli", "ciprofloxacin"): ([("b", "3000001")], [("card_hit_without_b", "3000001")])}


class AssocTables(FakeTables):
    """The two E. coli models also have the association path; meropenem has none."""

    def get(self, tid, organism=None, antibiotic=None, **_):
        key = (organism, antibiotic)
        if key in UNITIGS and tid in ("card_unitigs", "association_card_unitigs"):
            rows = UNITIGS[key][tid.startswith("association")]
            return pd.DataFrame(rows, columns=["allele_aware_state", "allele_aware_aros"])
        if key in UNITIGS and tid == "association_grades_patterns":
            return pd.DataFrame({"pattern_id": [0], "grade": ["confirmed"]})
        return super().get(tid, organism, antibiotic)


def test_the_determinant_overlap_counts_b_aros_per_model():
    """3.7: ampicillin finds 3000001-3 among its candidates and 3000003-4 by association;
    ciprofloxacin finds 3000001 only among the candidates (its association hit has no b)."""
    n = _numbers(AssocTables())
    assert n["assoc_overlap.n_candidate_only"] == 3
    assert n["assoc_overlap.n_both"] == 1
    assert n["assoc_overlap.n_association_only"] == 1
    assert n["assoc_overlap.n_models_association_only"] == 1


def test_no_overlap_keys_without_the_association_path():
    assert not any(k.startswith("assoc_overlap.") for k in _numbers())


def test_direction_counts_every_candidate_and_the_strong_novel_ones():
    """Direction is reported next to the grade, not inside it: all candidates, and the
    strong_novel ones separately."""
    n = _numbers()
    assert n["all.n_direction_r"] == 4 and n["all.n_direction_s"] == 2
    assert n["all.n_strong_novel_direction_r"] == 0 and n["all.n_strong_novel_direction_s"] == 1
