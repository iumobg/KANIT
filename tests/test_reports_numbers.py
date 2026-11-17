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


class LambdaTables(FakeTables):
    """The same run with each model's pyseer summary; one included model has none."""

    LAMBDA = {("ecoli", "ampicillin"): 0.4, ("ecoli", "ciprofloxacin"): 1.2,
              ("enterobacter_cloacae", "amikacin"): 0.6}

    def json(self, location, file, organism=None, antibiotic=None):
        if location == "pyseer_dir" and (organism, antibiotic) in self.LAMBDA:
            return {"lambda_background": self.LAMBDA[(organism, antibiotic)]}
        return None


def test_the_background_lambda_is_summarised_over_the_included_models():
    """Median, range and the count below 1 are over the included models with a summary; an
    excluded pair's summary does not count."""
    n = _numbers(LambdaTables())          # E. cloacae-amikacin is excluded, K. pneumoniae has none
    assert n["models.n_lambda_background"] == 2
    assert n["models.lambda_background_median"] == 0.8
    assert n["models.lambda_background_min"] == 0.4
    assert n["models.lambda_background_max"] == 1.2
    assert n["models.n_lambda_background_below_1"] == 1
    assert n["ecoli__ampicillin.lambda_background"] == 0.4
    assert "enterobacter_cloacae__amikacin.lambda_background" not in n


def test_no_background_lambda_keys_without_a_summary():
    assert not any("lambda_background" in k for k in _numbers())


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


# ---- Makale 3 (arms B and C) -----------------------------------------------------------
# Three included models; enterobacter_cloacae is excluded and must not count.
M3_AUC = {"A": {"ecoli__ampicillin": 0.90, "ecoli__ciprofloxacin": 0.80,
                "kpneumoniae__meropenem": 0.70},
          "B": {"ecoli__ampicillin": 0.92, "ecoli__ciprofloxacin": 0.80,
                "kpneumoniae__meropenem": 0.72},
          "C": {"ecoli__ampicillin": 0.88, "ecoli__ciprofloxacin": 0.79,
                "kpneumoniae__meropenem": 0.75, "enterobacter_cloacae__amikacin": 0.5}}
# Arm C's final model of ecoli/ampicillin: features 0, 1 and 2 share one presence vector,
# feature 3 has its own. Candidates 0 (weak), 1 (confirmed, b = 1) and 3 (candidate).
M3_BITS = [[0b10100000], [0b10100000], [0b10100000], [0b01000000]]
M3_GRADES = {"pattern_id": [0, 1, 3], "grade": ["weak", "confirmed", "candidate"],
             "allele_aware_card_state": ["no_card_hit", "b", "no_card_hit"],
             "prevalence": [True, True, True], "mda": [False, True, False],
             "cpss": [False, False, True], "pyseer": [False, True, True]}


class M3Tables:
    """One arm's outputs: a ROC-AUC per model; candidates, χ² prefilter and model matrix
    for ecoli/ampicillin only."""

    def __init__(self, arm, root=None, prefilter=(0, 1, 3)):
        self.arm, self.prefilter = arm, list(prefilter)
        # the arm's files below root (the matrix, the run and the measurement tables); no
        # root points nowhere, so the project's own files never leak into a test
        self.config = {"cv": {"n_bootstrap": 200}, "candidates": {"top_gain": 50},
                       "paths_prefix": str(root or "/nonexistent-m3-test"),
                       "paths_organism": {"matrix_dir": "{organism}/{antibiotic}"}}

    def get(self, tid, organism=None, antibiotic=None, **_):
        if tid == "panel_decisions":
            return DECISIONS
        if (organism, antibiotic) != ("ecoli", "ampicillin"):
            return None
        if tid == "grades_patterns":
            return pd.DataFrame(M3_GRADES)
        if tid == "cpss_prefilter":
            return pd.DataFrame({"pattern_id": self.prefilter,
                                 "chi2": [30.0, 20.0, 10.0][:len(self.prefilter)]})
        if tid == "card_patterns":
            return pd.DataFrame({"pattern_id": [1], "allele_aware_aros": ["3000001"]})
        if tid == "candidates":
            return pd.DataFrame({"pattern_id": [0, 1, 3], "gain_rank": [1, 2, 3]})
        return None

    def json(self, location, file, organism=None, antibiotic=None):
        v = M3_AUC[self.arm].get(f"{organism}__{antibiotic}")
        return None if v is None else {"arms": {"lineage_aware": {
            "roc_auc": v, "balanced_accuracy": v - 0.1, "pr_auc": None}}}


def _m3_numbers(arms, base=None):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amrtest_reports_m3",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = {}
    m._m3_numbers(base or M3Tables("A"), arms,
                  ["ecoli", "kpneumoniae", "enterobacter_cloacae"],
                  lambda key, value, *a, **k: rows.__setitem__(key, value))
    return rows


def test_m3_arms_are_compared_over_the_included_models_they_share():
    n = _m3_numbers({"B": M3Tables("B"), "C": M3Tables("C")})
    assert n["m3.C.n_models"] == 3            # the excluded pair stays out
    assert n["m3.compare.C__B.n_pairs"] == 3
    assert (n["m3.compare.C__B.n_a_better"], n["m3.compare.C__B.n_b_better"]) == (1, 2)
    assert n["m3.compare.B__A.n_tied"] == 1
    assert n["m3.compare.C__B.median_diff"] == pytest.approx(-0.01)


def test_m3_redundancy_counts_distinct_presence_vectors(tmp_path):
    import numpy as np
    d = tmp_path / "ecoli" / "ampicillin"
    d.mkdir(parents=True)
    np.array(M3_BITS, dtype=np.uint8).tofile(d / "patterns.bin")
    (d / "matrix_summary.json").write_text('{"patterns_shape": [4, 1]}')
    n = _m3_numbers({"C": M3Tables("C", root=tmp_path)})
    assert n["m3.C.distinct_vectors_median"] == 2
    assert n["m3.C.features_per_vector_median"] == 2
    assert n["m3.C.largest_vector_share_max"] == pytest.approx(0.75)
    # candidates 0 and 1 are one signal, graded by its best copy
    assert n["m3.C.n_candidate_signals"] == 2
    assert n["m3.C.signal.share_confirmed"] == 0.5
    assert n["m3.C.signal.share_weak"] == 0
    assert n["m3.C.copies_2_10.n_candidates"] == 2
    assert n["m3.C.copies_one.share_pass_cpss"] == 1
    assert n["m3.C.n_models_fewer_than_top_gain"] == 1


def test_m3_no_signal_keys_without_the_arm_matrix():
    n = _m3_numbers({"B": M3Tables("B")})
    assert n["m3.B.n_candidates"] == 3
    assert not any(k.startswith(("m3.B.distinct", "m3.B.signal", "m3.B.copies")) for k in n)


def test_m3_arm_a_is_graded_only_inside_its_prefilter():
    n = _m3_numbers({"B": M3Tables("B")}, base=M3Tables("A", prefilter=(1, 3)))
    assert n["m3.A_prefilter.n_candidates"] == 2
    assert n["m3.A_prefilter.share_card_b"] == 0.5
    assert n["m3.B.n_candidates"] == 3
    assert n["m3.B.candidates_per_b_aro_median"] == 1


def test_m3_other_metrics_get_their_own_keys_and_a_missing_one_none():
    n = _m3_numbers({"B": M3Tables("B"), "C": M3Tables("C")})
    assert n["m3.compare.C__B.ba.median_diff"] == pytest.approx(-0.01)
    assert n["m3.C.ba_median"] == pytest.approx(0.69)
    assert not any(".pr_auc" in k for k in n)


def test_m3_cost_is_read_from_the_arm_last_completed_run(tmp_path):
    import json
    runs = tmp_path / "runs" / "nextflow"
    for name, arm, ok, core in (("20261101T000000", "C", True, 5.0),
                                ("20261102T000000", "C", True, 9.0),
                                ("20261103T000000", "C", False, 99.0),
                                ("20261104T000000", "B", True, 2.0)):
        d = runs / name
        d.mkdir(parents=True)
        (d / "run_manifest.json").write_text(json.dumps({
            "run": {"entry": "M3"}, "params": {"config_overlay": f"config/m3_{arm}.yml"},
            "completed": {"success": ok}}))
        (d / "resources_summary.json").write_text(json.dumps({
            "total": {"tasks": 3, "core_hours": core, "cpu_hours": core / 2},
            "by_process": {"M3:M3_ARM:M3_KMER_STORE": {"cpu_hours": 1.0, "max_rss_gib": 37.0,
                                                        "max_elapsed_h": 2.5}}}))
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    for org, frac in (("ecoli", 0.98), ("enterococcus_faecium", 0.78)):
        (tables / f"m3_0a_summary_{org}.json").write_text(json.dumps({
            "organism": org, "homogeneous_frac": frac, "n_unitigs_tested": 5000,
            "n_genomes_sampled": 600, "max_pairwise_hamming": {"max": 6}}))
    d = runs / "20261020T000000"                    # the main run: arm A
    d.mkdir(parents=True)
    (d / "run_manifest.json").write_text(json.dumps({
        "run": {"entry": "main"}, "params": {"config_overlay": None},
        "completed": {"success": True}}))
    (d / "resources_summary.json").write_text(json.dumps({
        "total": {"tasks": 9, "core_hours": 500.0, "cpu_hours": 300.0},
        "by_process": {"MODELS:CV_UNITS": {"core_hours": 100.0, "cpu_hours": 80.0,
                                           "max_rss_gib": 90.0, "max_elapsed_h": 3.0},
                       "EVIDENCE:MDA": {"core_hours": 4.0, "cpu_hours": 1.0,
                                        "max_rss_gib": 2.0, "max_elapsed_h": 0.5},
                       "EVIDENCE:MDA_ASSOCIATION": {"core_hours": 4.0, "cpu_hours": 1.0,
                                                    "max_rss_gib": 2.0, "max_elapsed_h": 0.5},
                       "EVIDENCE:LP_CHUNKS": {"core_hours": 300.0, "cpu_hours": None,
                                              "max_rss_gib": None, "max_elapsed_h": None}}}))
    base = M3Tables("A", root=tmp_path)
    n = _m3_numbers({"B": M3Tables("B"), "C": M3Tables("C")}, base=base)
    # arm A: only the steps the arms also run; the association's MDA and the permutation
    # test stay out
    assert n["m3.A.cost.shared_steps.core_hours"] == 104.0
    assert n["m3.A.cost.evidence.cpu_hours"] == 1.0
    assert n["m3.A.cost.core_hours"] == 500.0
    assert n["m3.C.cost.core_hours"] == 9.0         # the last completed run, not the failed one
    assert n["m3.B.cost.core_hours"] == 2.0
    assert n["m3.C.cost.store.max_rss_gib"] == 37.0
    assert "m3.C.cost.cv.cpu_hours" not in n
    assert n["m3_0a.homogeneous_share_min"] == 0.78
    assert n["m3_0a.n_unitigs_tested"] == 5000


def test_the_exact_column_keeps_the_value_before_any_rounding(monkeypatch):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amrtest_reports_exact",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p = m._four_significant(0.0015503169494868645)
    assert p == 0.00155 and p.raw == 0.0015503169494868645
    df = m.numbers(FakeTables(), ["ecoli", "kpneumoniae", "enterobacter_cloacae"])
    row = df.set_index("key").loc["compare.amrfinderplus_median_diff"]
    assert float(row["exact"]) == pytest.approx(0.0)       # a float carries its exact value
    assert df.set_index("key").loc["panel.n_included", "exact"] == ""   # an integer does not


def _reports():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amrtest_reports_ctx",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_best_hit_gene_is_matched_to_card_and_amrfinderplus_names_in_any_form():
    m = _reports()
    idx = m.amr_name_index(["TEM-1", "blaOXA-23", "aac(6')-Ib", "sul1"])
    assert m.is_amr_name("blaTEM-1", idx) and m.is_amr_name("blaTEM", idx)
    assert m.is_amr_name("OXA-23", idx) and m.is_amr_name("SUL1", idx)
    assert m.is_amr_name("aac(6')-Ib", idx)
    assert not m.is_amr_name("pdxR", idx) and not m.is_amr_name(None, idx)
    assert not m.is_amr_name("", idx)


def test_context_by_grade_counts_only_candidates_without_a_card_hit():
    m = _reports()
    graded = pd.DataFrame({"model_id": ["m"] * 4, "pattern_id": [1, 2, 3, 4],
                           "grade": ["strong_novel", "strong_novel", "weak", "confirmed"],
                           "card_state": ["no_card_hit", "card_hit_without_b", "no_card_hit", "b"]})
    ctx = pd.DataFrame({"model_id": ["m"] * 4, "pattern_id": [1, 2, 3, 4],
                        "unitig_id": list("abcd"), "n_hits": [5, 5, 3, 9],
                        "plasmid_share": [0.8, 0.9, 0.0, 1.0],
                        "gene": ["blaTEM-1", None, None, "blaTEM-1"],
                        "product": ["beta-lactamase", "IS26 family transposase",
                                    "IS26 family transposase", ""]})
    rows = {}
    m._kb_context_grade_numbers(graded, ctx, m.amr_name_index(["TEM-1"]),
                                lambda k, v, *a, **kw: rows.__setitem__(k, v))
    assert rows["kb.context.no_card.strong_novel.n_patterns"] == 1      # the CARD hit stays out
    assert rows["kb.context.no_card.strong_novel.share_amr_name"] == 1.0
    assert rows["kb.context.no_card.weak.share_mobile"] == 1.0
    assert rows["kb.context.no_card.weak.share_mostly_plasmid"] == 0.0
    assert rows["kb.context.confirmed_all.share_amr_name"] == 1.0
    # pattern 4 has a gene but an empty product, pattern 3 a product but no gene: both are named
    assert rows["kb.context.confirmed_all.share_named"] == 1.0
    assert rows["kb.context.no_card.weak.share_named"] == 1.0


def test_a_hit_without_gene_or_product_counts_as_unnamed():
    m = _reports()
    graded = pd.DataFrame({"model_id": ["m"] * 2, "pattern_id": [1, 2], "grade": ["strong_novel"] * 2,
                           "card_state": ["no_card_hit"] * 2})
    ctx = pd.DataFrame({"model_id": ["m"] * 2, "pattern_id": [1, 2], "unitig_id": list("ab"),
                        "n_hits": [4, 4], "plasmid_share": [1.0, 0.0],
                        "gene": ["blaTEM-1", None], "product": [None, "  "]})
    rows = {}
    m._kb_context_grade_numbers(graded, ctx, m.amr_name_index(["TEM-1"]),
                                lambda k, v, *a, **kw: rows.__setitem__(k, v))
    assert rows["kb.context.no_card.strong_novel.share_named"] == 0.5
    assert rows["kb.context.no_card.strong_novel.share_amr_name"] == 0.5   # unnamed hits stay in the denominator
