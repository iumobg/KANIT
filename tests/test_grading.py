#!/usr/bin/env python3
"""Grading rule (lib.grading, 14b_grading.py): the rule's truth table and the
grades of both CARD modes."""
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import grading as gr  # noqa: E402
from lib.card_layer import (  # noqa: E402
    ALLELE_NOT_IN,
    B1,
    CO_CARRIED,
    HIT_NO_B,
    NO_HIT,
    VARIANT_NOT_COUNTED,
)

pytestmark = pytest.mark.unit

# The grading rule, all 48 combinations, written out by hand.
# prevalence mda cpss pyseer | b = 1 | CARD hit without b | no CARD hit
TRUTH = """
0000 candidate none      none
1000 candidate weak      weak
0100 candidate weak      weak
0010 candidate weak      weak
0001 candidate weak      weak
1100 confirmed candidate candidate
1010 confirmed candidate candidate
1001 confirmed candidate candidate
0110 confirmed candidate candidate
0101 confirmed candidate candidate
0011 confirmed candidate candidate
1110 confirmed candidate candidate
1101 confirmed candidate candidate
1011 confirmed candidate strong_novel
0111 confirmed candidate strong_novel
1111 confirmed candidate strong_novel
"""


def _truth():
    for line in TRUTH.strip().splitlines():
        bits, *grades = line.split()
        passed = dict(zip(gr.LAYERS, (b == "1" for b in bits), strict=True))
        yield from ((state, passed, g) for state, g in zip((B1, HIT_NO_B, NO_HIT), grades,
                                                           strict=True))


def test_truth_table_covers_all_48_combinations():
    cases = list(_truth())
    assert len(cases) == 48
    combos = {(s, tuple(p.values())) for s, p, _ in cases}
    assert combos == {(s, bits) for s in gr.CARD_STATES
                      for bits in itertools.product([False, True], repeat=4)}


@pytest.mark.parametrize("state,passed,expected", list(_truth()))
def test_grade_follows_the_rule_table(state, passed, expected):
    assert gr.grade(state, passed) == expected


def test_unknown_card_state_is_refused():
    with pytest.raises(ValueError):
        gr.grade("b=1", dict.fromkeys(gr.LAYERS, True))


def _card_patterns():
    # p1 variant b (allele-aware only), p2 homolog b, p3 co-carried gene, p4 no hit,
    # p5 no hit, p6 near-universal gene
    return pd.DataFrame({
        "pattern_id": [1, 2, 3, 4, 5, 6], "n_members": [1, 2, 1, 3, 1, 1],
        "allele_aware_state": [B1, B1, HIT_NO_B, NO_HIT, NO_HIT, HIT_NO_B],
        "allele_aware_reasons": ["", "", CO_CARRIED, "", "", "near_universal_gene"],
        "homolog_only_state": [HIT_NO_B, B1, HIT_NO_B, NO_HIT, NO_HIT, HIT_NO_B],
        "homolog_only_reasons": [VARIANT_NOT_COUNTED, "", CO_CARRIED, "", "",
                                 "near_universal_gene"],
    })


def _layers(rows):
    """rows: {pattern_id: 'pmcy' bits}."""
    return {layer: pd.Series({p: bits[i] == "1" for p, bits in rows.items()}, name=layer)
            for i, layer in enumerate(gr.LAYERS)}


LAYER_BITS = {1: "1100", 2: "0010", 3: "0011", 4: "1011", 5: "0011", 6: "1000"}


def test_both_modes_are_graded_and_the_rule_picks_grade():
    g = gr.grade_patterns(_card_patterns(), _layers(LAYER_BITS), "allele_aware")
    by = g.set_index("pattern_id")
    assert by["allele_aware_grade"].to_dict() == {
        1: "confirmed", 2: "candidate", 3: "candidate", 4: "strong_novel", 5: "candidate",
        6: "weak"}
    assert by["homolog_only_grade"].to_dict() == {
        1: "candidate", 2: "candidate", 3: "candidate", 4: "strong_novel", 5: "candidate",
        6: "weak"}
    assert (g["grade"] == g["allele_aware_grade"]).all()
    assert by.loc[3, "allele_aware_card_reasons"] == CO_CARRIED          # the flag of row 3
    assert by.loc[1, "homolog_only_card_reasons"] == VARIANT_NOT_COUNTED
    assert by.loc[4, "layers_passed"] == "prevalence;cpss;pyseer" and by.loc[4, "n_layers"] == 3
    h = gr.grade_patterns(_card_patterns(), _layers(LAYER_BITS), "homolog_only")
    assert (h["grade"] == h["homolog_only_grade"]).all()
    s = gr.summary(g, "allele_aware")
    assert s["patterns_by_grade"]["allele_aware"]["confirmed"] == 1
    assert s["patterns_by_grade"]["homolog_only"]["confirmed"] == 0
    assert s["patterns_passing"] == {"prevalence": 3, "mda": 1, "cpss": 4, "pyseer": 3}


def test_strong_novel_never_has_a_card_hit():
    states = [B1, HIT_NO_B, NO_HIT]
    rows = [(s, bits) for s in states for bits in itertools.product("01", repeat=4)]
    cp = pd.DataFrame({"pattern_id": range(len(rows)), "n_members": 1,
                       "allele_aware_state": [s for s, _ in rows], "allele_aware_reasons": "",
                       "homolog_only_state": [s for s, _ in rows], "homolog_only_reasons": ""})
    g = gr.grade_patterns(cp, _layers({i: "".join(b) for i, (_, b) in enumerate(rows)}),
                          "allele_aware")
    novel = g[g["grade"] == gr.STRONG_NOVEL]
    assert len(novel) == 3 and (novel["allele_aware_card_state"] == NO_HIT).all()


def test_a_candidate_missing_from_a_layer_stops_the_grading():
    bits = dict(LAYER_BITS)
    del bits[5]
    with pytest.raises(ValueError, match="no result"):
        gr.grade_patterns(_card_patterns(), _layers(bits), "allele_aware")
    with pytest.raises(ValueError, match="primary mode"):
        gr.grade_patterns(_card_patterns(), _layers(LAYER_BITS), "blast")


def test_read_layer(tmp_path):
    f = tmp_path / "cpss.csv"
    pd.DataFrame({"pattern_id": [3, 1], "pi": [0.7, 0.1], "passes": [True, False]}).to_csv(
        f, index=False)
    assert gr.read_layer(f, "cpss").to_dict() == {3: True, 1: False}
    pd.DataFrame({"pattern_id": [3, 1], "passes": [1, 0]}).to_csv(f, index=False)
    assert gr.read_layer(f, "cpss").to_dict() == {3: True, 1: False}
    for bad in ({"pattern_id": [3, 1], "passes": [1, None]},
                {"pattern_id": [3, 3], "passes": [1, 0]}, {"pattern_id": [3]}):
        pd.DataFrame(bad).to_csv(f, index=False)
        with pytest.raises(ValueError):
            gr.read_layer(f, "cpss")


def test_unitigs_take_their_pattern_grade_and_keep_their_own_card_state():
    g = gr.grade_patterns(_card_patterns(), _layers(LAYER_BITS), "allele_aware")
    cu = pd.DataFrame({"pattern_id": [1, 2, 2], "unitig_index": [10, 20, 21],
                       "sequence": ["A", "C", "G"],
                       "allele_aware_state": [B1, B1, HIT_NO_B],
                       "allele_aware_reasons": ["", "", ALLELE_NOT_IN],
                       "allele_aware_aros": ["1", "2", "3"], "allele_aware_n_b": [3, 3, 0],
                       "homolog_only_state": [HIT_NO_B, B1, HIT_NO_B],
                       "homolog_only_reasons": [VARIANT_NOT_COUNTED, "", VARIANT_NOT_COUNTED],
                       "homolog_only_aros": ["1", "2", "3"], "homolog_only_n_b": [0, 3, 0]})
    u = gr.grade_unitigs(cu, g).set_index("unitig_index")
    assert u["grade"].to_dict() == {10: "confirmed", 20: "candidate", 21: "candidate"}
    assert u.loc[21, "allele_aware_card_state"] == HIT_NO_B
    assert u.loc[21, "allele_aware_card_reasons"] == ALLELE_NOT_IN
    with pytest.raises(ValueError):
        gr.grade_unitigs(cu.assign(pattern_id=[1, 2, 99]), g)


def test_grading_script_end_to_end(tmp_path, monkeypatch, load_script):
    m = load_script("14b_grading.py")
    paths = {"candidates_file": tmp_path / "candidates.csv", "card_layer_dir": tmp_path / "card",
             "layers_dir": tmp_path / "layers", "grades_dir": tmp_path / "grades"}
    config = {"paths_organism": {k: str(v) for k, v in paths.items()},
              "grading": {"primary_mode": "allele_aware"}}
    monkeypatch.setattr(m, "load_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["14b", "--organism", "ecoli", "--antibiotic", "x"])
    cp = _card_patterns().assign(allele_aware_aros="", homolog_only_aros="")
    paths["card_layer_dir"].mkdir()
    cp.to_csv(paths["card_layer_dir"] / "card_patterns.csv", index=False)
    pd.DataFrame({"pattern_id": [1, 4], "unitig_index": [10, 40], "sequence": ["A", "C"],
                  **{f"{mode}_{x}": v for mode in ("allele_aware", "homolog_only")
                     for x, v in (("state", [B1, NO_HIT]), ("reasons", ["", ""]),
                                  ("aros", ["1", ""]), ("n_b", [3, 0]))}}).to_csv(
        paths["card_layer_dir"] / "card_unitigs.csv", index=False)
    paths["layers_dir"].mkdir()
    for layer, s in _layers(LAYER_BITS).items():
        pd.DataFrame({"pattern_id": s.index, "passes": s.to_numpy()}).to_csv(
            paths["layers_dir"] / f"{layer}.csv", index=False)
    pd.DataFrame({"pattern_id": [6, 5, 4, 3, 2, 1]}).to_csv(paths["candidates_file"], index=False)
    m.main()
    out = pd.read_csv(paths["grades_dir"] / "grades_patterns.csv", keep_default_na=False)
    assert out.set_index("pattern_id").loc[3, "allele_aware_card_reasons"] == CO_CARRIED
    summary = json.loads((paths["grades_dir"] / "grades_summary.json").read_text())
    assert summary["primary_mode"] == "allele_aware" and summary["n_unitigs"] == 2
    pd.DataFrame({"pattern_id": [1, 2]}).to_csv(paths["candidates_file"], index=False)
    with pytest.raises(SystemExit, match="rerun 09"):
        m.main()


@pytest.mark.parametrize("primary", ["allele_aware", "homolog_only"])
def test_truth_table_through_both_modes(primary):
    """All 48 combinations through grade_patterns: each mode grades from its own CARD
    state, and the primary mode picks the reported grade."""
    cases = list(_truth())
    expected = {(s, tuple(p.values())): g for s, p, g in cases}
    states = list(gr.CARD_STATES)
    rows, bits = [], {}
    for i, (state, passed, _) in enumerate(cases):
        other = states[(states.index(state) + 1) % 3]          # the other mode differs
        rows.append({"pattern_id": i, "n_members": 1, "allele_aware_state": state,
                     "allele_aware_reasons": "", "homolog_only_state": other,
                     "homolog_only_reasons": ""})
        bits[i] = "".join("1" if passed[layer] else "0" for layer in gr.LAYERS)
    g = gr.grade_patterns(pd.DataFrame(rows), _layers(bits), primary).set_index("pattern_id")
    for i, (state, passed, want) in enumerate(cases):
        other = states[(states.index(state) + 1) % 3]
        assert g.loc[i, "allele_aware_grade"] == want
        assert g.loc[i, "homolog_only_grade"] == expected[(other, tuple(passed.values()))]
        assert g.loc[i, "grade"] == g.loc[i, f"{primary}_grade"]
