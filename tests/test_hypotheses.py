#!/usr/bin/env python3
"""Hypothesis tests (lib.hypotheses) on a small knowledge base written by hand."""
import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import hypotheses as hy  # noqa: E402
from lib import knowledge_base as kb  # noqa: E402

pytestmark = pytest.mark.unit

GYRA = "Escherichia coli gyrA conferring resistance to fluoroquinolones"
# model -> [(pattern, stable, card state, grade, AROs[, route])]; the ampicillin model's
# pattern 9, nominated by association, enters no hypothesis test:
# its qnr hit would link ampicillin to ciprofloxacin in H3
PATTERNS = {
    "ecoli__ciprofloxacin": [(1, True, "b", "confirmed", ["1"]), (2, True, "no_card_hit",
                                                                  "strong_novel", []),
                             (3, False, "card_hit_without_b", "candidate", ["2"])],
    "ecoli__levofloxacin": [(1, True, "b", "confirmed", ["1"])],
    "ecoli__ampicillin": [(1, False, "b", "candidate", ["3"]),
                          (9, False, "card_hit_without_b", "candidate", ["2"], "association")],
}
AROS = [("ARO:1", GYRA, "variant", "fluoroquinolone resistant gyrA"),
        ("ARO:2", "QnrS1", "homolog", "quinolone resistance protein (qnr)"),
        ("ARO:3", "TEM-1", "homolog", "TEM beta-lactamase")]


def _seq(name: str) -> str:
    return "".join("ACGT"[int(c, 16) % 4] for c in hashlib.sha256(name.encode()).hexdigest()[:40])


def _kb(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    kb.create(conn)
    ins = kb.insert
    ins(conn, "organism", pd.DataFrame([{"organism_id": "ecoli", "name": "E. coli",
                                         "ncbi_taxids": "562"}]))
    ins(conn, "antibiotic", pd.DataFrame({
        "antibiotic_id": ["ciprofloxacin", "levofloxacin", "ampicillin"],
        "drug_class": ["quinolones", "quinolones", "penicillins"],
        "card_drug_classes": ["fluoroquinolone antibiotic"] * 2 + ["penicillin beta-lactam"]}))
    ins(conn, "aro", pd.DataFrame(AROS, columns=["aro_accession", "name", "model_type",
                                                 "gene_family"]))
    ids = [f"g{i}" for i in range(10)]
    ins(conn, "genome", pd.DataFrame({"genome_id": ids, "organism_id": "ecoli", "qc_pass": 1,
                                     "ncbi_taxid": 562}))
    for mid, pats in PATTERNS.items():
        ab = mid.split("__")[1]
        ins(conn, "model", pd.DataFrame([{
            "model_id": mid, "organism_id": "ecoli", "antibiotic_id": ab, "n_genomes": 10,
            "n_resistant": 5, "n_susceptible": 5, "n_lineages": 5, "largest_lineage_share": 0.2,
            "n_unitigs": 10, "n_patterns": 5, "evaluable": 1}]))
        ins(conn, "model_genome", pd.DataFrame({"model_id": mid, "genome_id": ids,
                                                "row_index": range(10),
                                                "resistant": [1] * 5 + [0] * 5,
                                                "lineage_cluster": "1"}))
        for pid, stable, state, grade, aros, *route in pats:
            uid, seq = kb.unitig_id(_seq(f"{mid}:{pid}"))
            ins(conn, "pattern", pd.DataFrame([{"model_id": mid, "pattern_id": pid,
                                                "n_members": 1, "n_present": 5}]))
            if conn.execute("SELECT 1 FROM unitig WHERE unitig_id = ?", (uid,)).fetchone() is None:
                ins(conn, "unitig", pd.DataFrame([{"unitig_id": uid, "sequence": seq,
                                                   "length": len(seq)}]))
            ins(conn, "pattern_member", pd.DataFrame([{"model_id": mid, "pattern_id": pid,
                                                       "unitig_id": uid}]))
            assoc = route == ["association"]
            ins(conn, "candidate", pd.DataFrame([{
                "model_id": mid, "pattern_id": pid, "route": "association" if assoc else "model",
                "source": "pyseer" if assoc else "both"}]))
            ins(conn, "cpss_result", pd.DataFrame([{"model_id": mid, "pattern_id": pid,
                                                    "in_prefilter": 1, "n_selected": 1,
                                                    "pi": 0.9 if stable else 0.1,
                                                    "passes": int(stable)}]))
            for mode in ("allele_aware", "homolog_only"):
                ins(conn, "grade", pd.DataFrame([{
                    "model_id": mid, "pattern_id": pid, "mode": mode,
                    "primary_mode": int(mode == "allele_aware"), "card_state": state,
                    "card_reasons": "", "n_layers": 2, "layers_passed": "cpss;pyseer",
                    "grade": grade}]))
                ins(conn, "card_annotation", pd.DataFrame([{
                    "model_id": mid, "unitig_id": uid, "mode": mode, "state": state,
                    "reasons": "", "n_b": 3 if state == "b" else 0, "located_genomes": "g0"}]))
            for a in aros:
                ins(conn, "card_hit", pd.DataFrame([{"model_id": mid, "unitig_id": uid,
                                                     "aro_accession": f"ARO:{a}"}]))
    ins(conn, "parameter", pd.DataFrame({"name": ["cpss.q", "cpss.pi_threshold"],
                                         "value": ["50", "0.6"], "description": "d"}))
    conn.commit()
    return conn


@pytest.fixture
def conn(tmp_path):
    c = _kb(tmp_path / "kb.sqlite")
    yield c
    c.close()


def test_h1_h2(conn):
    t1, s1 = hy.h1(conn)
    assert dict(zip(t1["model_id"], t1["n_stable"], strict=True)) == {
        "ecoli__ampicillin": 0, "ecoli__ciprofloxacin": 2, "ecoli__levofloxacin": 1}
    assert not s1["supported"] and s1["n_with_stable"] == 2
    assert t1["mb_bound"].iloc[0] == pytest.approx(2500 / (0.2 * 2))   # ampicillin: 2 in the prefilter
    t2, s2 = hy.h2(conn)
    shares = dict(zip(t2["model_id"], t2["share_b_allele_aware"], strict=True))
    assert shares["ecoli__ciprofloxacin"] == 0.5 and shares["ecoli__levofloxacin"] == 1.0
    assert np.isnan(shares["ecoli__ampicillin"])
    assert s2["median"] == pytest.approx(0.75) and s2["supported"]


def test_h3(conn):
    t, s = hy.h3(conn, n_mc=2000, seed=1)
    by = {(r.model_a, r.model_b): r for r in t.itertuples()}
    cl = by[("ecoli__ciprofloxacin", "ecoli__levofloxacin")]
    assert cl.same_class and cl.overlap == 1.0 and cl.n_universe == 3
    assert by[("ecoli__ampicillin", "ecoli__ciprofloxacin")].overlap == 0.0
    assert cl.fisher_p == pytest.approx(cl.mc_p, abs=0.05)
    assert s["n_within"] == 1 and s["n_cross"] == 2 and not s["supported"]
    assert hy.benjamini_yekutieli([0.01, 0.02, np.nan]).tolist()[:2] == pytest.approx(
        [0.03, 0.03])


def test_symbol_mapping():
    idx = hy.card_family_index(pd.DataFrame({
        "ARO Name": ["TEM-1", "QnrS1", "vanH_in_vanA_cl", "CTX-M-15", "CTX-M-27"],
        "CARD Short Name": ["TEM-1", "QnrS1", "vanH_in_vanA_cl", "CTX-M-15", "CTX-M-27"],
        "AMR Gene Family": ["TEM beta-lactamase", "quinolone resistance protein (qnr)",
                            "vanH", "CTX-M beta-lactamase", "CTX-M beta-lactamase"]}))
    assert hy.map_symbol("blaTEM-1", idx) == {"TEM beta-lactamase"}
    assert hy.map_symbol("qnrS1", idx) == {"quinolone resistance protein (qnr)"}
    assert hy.map_symbol("vanH-A", idx) == {"vanH"}
    assert hy.map_symbol("blaCTX-M", idx) == {"CTX-M beta-lactamase"}   # one family behind it
    assert hy.map_symbol("blaOXA-48", idx) == set()


def _reference():
    rows = []
    for g in ("g0", "g1", "g2", "g3"):                    # resistant genomes, point mutation
        rows.append((g, "gyrA_S83L", "AMR", "POINT", "core", "QUINOLONE", "QUINOLONE"))
    for g in ("g0", "g1", "g5"):                          # qnrS1 in two resistant, one not
        rows.append((g, "qnrS1", "AMR", "AMR", "core", "QUINOLONE", "QUINOLONE"))
    for g in ("g0", "g1", "g2", "g3", "g4"):              # TEM-1 in every resistant genome
        rows.append((g, "blaTEM-1", "AMR", "AMR", "core", "BETA-LACTAM", "BETA-LACTAM"))
    rows.append(("g6", "blaTEM-1", "AMR", "AMR", "plus", "BETA-LACTAM", "BETA-LACTAM"))
    return {"ecoli": pd.DataFrame(rows, columns=["genome_id", "element_symbol", "type", "subtype",
                                                 "scope", "class", "subclass"])}


KEYWORDS = {"ciprofloxacin": {"QUINOLONE", "FLUOROQUINOLONE"},
            "levofloxacin": {"QUINOLONE", "FLUOROQUINOLONE"},
            "ampicillin": {"AMPICILLIN", "BETA-LACTAM"}}
IDX = {hy._norm("TEM-1"): {"TEM beta-lactamase"},
       hy._norm("QnrS1"): {"quinolone resistance protein (qnr)"}}


def test_h7(conn):
    t, s, unmapped = hy.h7(conn, _reference(), IDX, KEYWORDS)
    cip = t[t["model_id"] == "ecoli__ciprofloxacin"].set_index("determinant")
    gyr, qnr = cip.loc["GENE:gyrA"], cip.loc["FAM:quinolone resistance protein (qnr)"]
    assert gyr["prev_resistant"] == 0.8 and gyr["expected_0.1"] and gyr["best_grade"] == "confirmed"
    assert qnr["prev_resistant"] == 0.4 and qnr["prev_susceptible"] == 0.2
    assert qnr["recovered"] and qnr["best_grade"] == "candidate"
    amp = t[t["model_id"] == "ecoli__ampicillin"].set_index("determinant")
    assert amp.loc["FAM:TEM beta-lactamase", "prev_susceptible"] == 0.0     # 'plus' scope dropped
    assert set(hy.model_patterns(conn, "ecoli__ampicillin")["pattern_id"]) == {1}
    s10 = s["0.1"]
    # expected and recovered: gyrA (cip, levo), qnr (cip), TEM (amp) -> 2 of 4 confirmed
    assert s10["n_recovered"] == 4 and s10["share_confirmed"] == 0.5 and s10["supported_primary"]
    assert unmapped == {}


def test_h6(conn):
    t2, _ = hy.h2(conn)
    t, s = hy.h6(conn, _reference(), KEYWORDS, t2)
    cls = dict(zip(t["model_id"], t["class"], strict=True))
    assert cls == {"ecoli__ampicillin": "acquired", "ecoli__ciprofloxacin": "mutational",
                   "ecoli__levofloxacin": "mutational"}
    a = t.set_index("model_id")
    assert a.loc["ecoli__ciprofloxacin", "M"] == 0.8 and a.loc["ecoli__ciprofloxacin", "A"] == 0.4
    assert s["n_mutational"] == 2 and s["n_acquired"] == 0      # ampicillin has no H2 share


def _permuted(conn, flagged=()):
    kb.insert(conn, "label_permutation", pd.DataFrame([{
        "model_id": mid, "auc_observed": 0.9, "null_mean": 0.5, "n_permutations": 100,
        "p": 0.3 if mid in flagged else 0.0099, "q": 0.3 if mid in flagged else 0.01,
        "flag": hy.PERMUTATION_FLAG if mid in flagged else ""} for mid in PATTERNS]))


def test_the_sensitivity_analysis_drops_the_flagged_models(conn):
    """Sensitivity analysis: without levofloxacin, H3 loses its only within-class pair and
    H7 two of its confirmed determinants; the per-model results stay as they were."""
    _permuted(conn, flagged={"ecoli__levofloxacin"})
    assert hy.flagged_models(conn) == ["ecoli__levofloxacin"]
    s = hy.sensitivity(conn, _reference(), IDX, KEYWORDS, n_mc=200, seed=1)
    assert s["status"] == "computed" and s["excluded_models"] == ["ecoli__levofloxacin"]
    assert s["H1"]["n_models"] == 2 and s["H2"]["median"] == 0.5 and s["H2"]["supported"]
    assert s["H3"]["n_within"] == 0 and s["H3"]["n_cross"] == 1 and not s["H3"]["supported"]
    assert s["H7"]["0.1"]["n_recovered"] == 3 and not s["H7"]["0.1"]["supported_primary"]
    assert s["H6"]["n_mutational"] == 1
    t1, _ = hy.h1(conn)                                  # the primary analysis is unchanged
    assert len(t1) == 3


def test_no_sensitivity_analysis_without_every_permutation_or_a_flag(conn):
    """The q values are corrected across all models, so a missing test stops it; with no
    flagged model it is the primary analysis."""
    assert hy.flagged_models(conn) is None
    s = hy.sensitivity(conn, _reference(), IDX, KEYWORDS, n_mc=200, seed=1)
    assert s["status"].startswith("not computed") and "H1" not in s
    _permuted(conn)
    s = hy.sensitivity(conn, _reference(), IDX, KEYWORDS, n_mc=200, seed=1)
    assert s["status"].startswith("no model flagged") and s["excluded_models"] == []


def test_the_numbers_file_reads_fields_the_summaries_have(conn):
    """reports.py keys hypotheses.json by field name; a renamed field would drop a key
    without an error."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("amrtest_reports_h",
                                                  PROJECT_ROOT / "scripts" / "reports.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    s = hy.run(conn, _reference(), IDX, KEYWORDS, n_mc=200, seed=1)["summary"]
    for test, field, *_ in r.HYPOTHESIS_KEYS:
        assert field in s[test], (test, field)
    assert set(r.EXPECTED_THRESHOLDS) == set(s["H7"])
    for field, *_ in r.EXPECTED_KEYS:
        assert field in s["H7"]["0.1"], field
    assert len(s["H7"]["0.1"]["secondary_table"]) == 2


def test_script_end_to_end(tmp_path, monkeypatch):
    import importlib.util
    import json
    _kb(tmp_path / "kanit.sqlite").close()
    with sqlite3.connect(tmp_path / "kanit.sqlite") as c:
        kb.insert(c, "release", pd.DataFrame([{
            "kb_version": "t", "schema_version": kb.SCHEMA_VERSION, "created_at": "now",
            "config_sha256": "y",
            "tools": "{}"}]))
    (tmp_path / "card").mkdir()
    pd.DataFrame({"ARO Name": ["QnrS1", "TEM-1"], "CARD Short Name": ["QnrS1", "TEM-1"],
                  "AMR Gene Family": ["quinolone resistance protein (qnr)", "TEM beta-lactamase"]}
                 ).to_csv(tmp_path / "card" / "aro_index.tsv", sep="\t", index=False)
    ext = tmp_path / "ecoli" / "external"
    ext.mkdir(parents=True)
    _reference()["ecoli"].to_csv(ext / "amrfinder_calls.csv", index=False)
    pd.DataFrame({"genome_id": [f"g{i}" for i in range(10)]}).to_csv(
        ext / "amrfinder_genomes.csv", index=False)
    config = {"paths_organism": {"kb_dir": str(tmp_path), "cross_model_dir": str(tmp_path / "x"),
                                 "external_dir": str(tmp_path / "{organism}" / "external")},
              "card": {"card_json": str(tmp_path / "card" / "card.json")}}
    spec = importlib.util.spec_from_file_location("amrtest_hyp", PROJECT_ROOT / "scripts" /
                                                  "hypotheses.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "load_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["x", "--n-mc", "200"])
    m.main()
    s = json.loads((tmp_path / "x" / "hypotheses" / "hypotheses.json").read_text())
    assert s["H2"]["supported"] and s["reference_missing"] == [] and s["H6"]["n_mutational"] == 2
    assert s["sensitivity_label_permutation"]["status"].startswith("not computed")
    assert (tmp_path / "x" / "hypotheses" / "h3_pairs.csv").exists()
    from lib import contract  # the tables follow the output contract
    for tid in ("h1_models", "h2_models", "h3_pairs", "h6_models", "h7_determinants",
                "h7_unmapped_symbols"):
        path = contract.table_path(tid, config)
        assert contract.validate_csv(path, contract.load()["tables"][tid]) == [], tid
    pd.DataFrame({"genome_id": ["g0"]}).to_csv(ext / "amrfinder_genomes.csv", index=False)
    with pytest.raises(SystemExit, match="did not analyse"):
        m.main()
