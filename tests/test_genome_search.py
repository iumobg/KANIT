#!/usr/bin/env python3
"""The KANIT search of one assembly (lib/genome_search.py, lib/genome_report.py,
lib/genome_annotation.py, scripts/kanit_search.py) on a small knowledge base."""
import gzip
import json
import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import genome_annotation as ga  # noqa: E402
from lib import genome_report as gr  # noqa: E402
from lib import genome_search as gs  # noqa: E402
from lib import knowledge_base as kb  # noqa: E402
from lib.card_layer import revcomp  # noqa: E402

rng = random.Random(7)


def _seq(n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


U = {f"u{i}": _seq(40 + i) for i in range(1, 8)}       # unitigs, 41-47 bases


def _kb(path: Path) -> Path:
    """Two models of one organism. TMP: pattern 1 (u1, u2; confirmed, R), pattern 2 (u3;
    strong_novel, R), pattern 3 (u4, u5, u6; candidate, S), pattern 9 (u7; none, R).
    CIP: no marker."""
    conn = sqlite3.connect(path)
    kb.create(conn)
    ins = lambda t, rows: conn.executemany(  # noqa: E731
        f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
    ins("release", [("1.0.0", "1.0", "2026-11-24", "x", "abc", 0, "4.0.2", "{}", None, None)])
    ins("organism", [("ecoli", "Escherichia coli", "562", "negative", "Pseudomonadota")])
    ins("antibiotic", [("trimethoprim", "folate_pathway_inhibitors", "diaminopyrimidine antibiotic", None),
                       ("ciprofloxacin", "quinolones", "fluoroquinolone antibiotic", None)])
    ins("model", [(f"ecoli__{a}", "ecoli", a, 100, 50, 50, 10, 0.1, 1000, 500, 1, None, None)
                  for a in ("trimethoprim", "ciprofloxacin")])
    members = {1: ["u1", "u2"], 2: ["u3"], 3: ["u4", "u5", "u6"], 9: ["u7"]}
    grade = {1: "confirmed", 2: "strong_novel", 3: "candidate", 9: "none"}
    direction = {1: "R", 2: "R", 3: "S", 9: "R"}
    m = "ecoli__trimethoprim"
    ins("pattern", [(m, p, len(v), 10, None) for p, v in members.items()])
    ins("candidate", [(m, p, "model", "gain", i + 1, 1.0) for i, p in enumerate(members)])
    for mode, prim in (("allele_aware", 1), ("homolog_only", 0)):
        ins("grade", [(m, p, mode, prim, "b" if p == 1 else "no_card_hit", "", 2, "mda", grade[p])
                      for p in members])
    ins("prevalence_result", [(m, p, 5, 1, 0.5, 0.1, 0.4, direction[p], 0.01, 0.02, 1) for p in members])
    ins("unitig", [(u, s, len(s)) for u, s in U.items()])
    ins("pattern_member", [(m, p, u) for p, v in members.items() for u in v])
    ins("aro", [("ARO:3002858", "dfrA17", "homolog", "trimethoprim resistant dihydrofolate reductase dfr",
                 "diaminopyrimidine antibiotic", "antibiotic target replacement")])
    ins("card_hit", [(m, "u1", "ARO:3002858")])
    ins("unitig_context", [("u3", "ecoli", "ncbi", "2026-11-09", None, 5, "CP1", "plasmid pX", 99.0, 1.0,
                            1e-20, "", "hypothetical protein", 0.8)])
    conn.commit()
    conn.close()
    return path


def _genome(*parts) -> dict[str, str]:
    """contig A: u1 (reverse strand) and u3; contig B: u4 only; filler between."""
    a = _seq(2000) + revcomp(U["u1"]) + _seq(500) + U["u3"] + _seq(3000) + U["u7"] + _seq(1000)
    b = _seq(800) + U["u4"] + _seq(800)
    return {"A": a, "B": b}


def test_find_unitigs_both_strands_and_each_occurrence():
    g = {"c": _seq(100) + U["u1"] + _seq(50) + revcomp(U["u1"]) + _seq(30)}
    h = gs.find_unitigs(g, {"u1": U["u1"]})
    assert list(h["strand"]) == ["+", "-"] and list(h["start"]) == [100, 100 + len(U["u1"]) + 50]
    assert (h["end"] - h["start"] == len(U["u1"])).all()
    pal = "ACGT" * 8                                    # its own reverse complement
    assert len(gs.find_unitigs({"c": "TT" + pal + "TT"}, {"p": pal})) == 1
    with pytest.raises(ValueError, match="shorter than the seed"):
        gs.find_unitigs(g, {"short": "ACGT"})


def test_presence_needs_half_of_the_members():
    members = pd.DataFrame({"model_id": "m", "pattern_id": [1, 1, 2, 2, 2],
                            "unitig_id": ["a", "b", "c", "d", "e"]})
    hits = pd.DataFrame({"unitig_id": ["a", "c"]})
    p = gs.pattern_presence(members, hits).set_index("pattern_id")
    assert bool(p.loc[1, "present"]) and not bool(p.loc[2, "present"])          # 1/2 and 1/3


def test_read_assembly(tmp_path):
    f = tmp_path / "g.fna.gz"
    f.write_bytes(gzip.compress(b">c1 desc\nacgt\nACGT\n>c2\nGG\n"))
    assert gs.read_assembly(f) == {"c1": "ACGTACGT", "c2": "GG"}
    with pytest.raises(ValueError, match="duplicate"):
        gs.parse_fasta(">a\nAC\n>a\nGT\n")
    with pytest.raises(ValueError, match="not a FASTA"):
        gs.parse_fasta("ACGT\n")


def test_search_calls_and_hits(tmp_path):
    db = _kb(tmp_path / "kanit.sqlite")
    with sqlite3.connect(db) as conn:
        res = gs.search(conn, _genome(), "ecoli")
        with pytest.raises(ValueError, match="no model"):
            gs.search(conn, _genome(), "kpneumoniae")
    mk = res.markers.set_index("pattern_id")
    assert set(mk.index) == {1, 2, 3}                     # grade none is no marker
    assert bool(mk.loc[1, "present"]) and bool(mk.loc[2, "present"])     # u1 of u1+u2; u3
    assert not bool(mk.loc[3, "present"])                                # u4 of three members
    assert mk.loc[1, "card_genes"] == "dfrA17" and mk.loc[2, "context_plasmid_share"] == 0.8
    c = res.calls.set_index("antibiotic_id")
    assert c.loc["trimethoprim", "call"] == "R" and c.loc["trimethoprim", "n_markers_rule"] == 2
    assert c.loc["trimethoprim", "evidence"].startswith("confirmed #1 (dfrA17)")
    assert c.loc["ciprofloxacin", "call"] == "S" and c.loc["ciprofloxacin", "n_markers_rule"] == 0
    assert c.loc["trimethoprim", "score"] == 2
    # rule confirmed only: the strong_novel marker no longer counts
    with sqlite3.connect(db) as conn:
        r2 = gs.search(conn, _genome(), "ecoli", "upto_confirmed")
    assert r2.calls.set_index("antibiotic_id").loc["trimethoprim", "n_markers_rule"] == 1
    # hits: present markers only, with their position and strand
    h = res.hits.set_index("unitig_id")
    assert set(h.index) == {"u1", "u3"} and h.loc["u1", "strand"] == "-" and h.loc["u1", "contig"] == "A"
    w = gr.windows(res.hits, res.contigs)
    assert len(w) == 1 and w.loc[0, "grade"] == "confirmed" and w.loc[0, "n_markers"] == 2


def test_annotation_readers(tmp_path):
    t = tmp_path / "x.tsv"
    t.write_text("# Bakta\n#Sequence Id\tType\tStart\tStop\tStrand\tLocus Tag\tGene\tProduct\tDbXrefs\n"
                 "A\tcds\t1\t300\t+\tL1\tblaTEM-1\tclass A beta-lactamase TEM-1\t\n"
                 "A\tcds\t400\t900\t-\tL2\t\tIS26 family transposase\t\n"
                 "A\tcds\t1000\t1200\t+\tL3\t\thypothetical protein\t\n"
                 "A\tcds\t1300\t1500\t+\tL4\tdnaA\tchromosomal replication initiator protein DnaA\t\n")
    g = ga.read_bakta_tsv(t)
    assert list(g["category"]) == ["amr", "mge", "hypothetical", "other"]
    assert g.loc[0, "start"] == 0 and g.loc[0, "end"] == 300
    m = tmp_path / "contig_report.txt"
    m.write_text("sample_id\tmolecule_type\tprimary_cluster_id\tcontig_id\nx\tplasmid\tAA1\tA\n"
                 "x\tchromosome\t-\tB\n")
    assert ga.read_mob_contig_report(m) == {"A": "plasmid (AA1, MOB-suite)", "B": "chromosome (MOB-suite)"}
    s = tmp_path / "p.tsv"
    s.write_text("seq_name\tlength\tplasmid_score\nA\t100\t0.97\n")
    assert ga.read_genomad_plasmids(s, ["A", "B"]) == {
        "A": "plasmid (geNomad score 0.97)", "B": "not classified as plasmid (geNomad)"}


def test_command_line_report(tmp_path):
    import importlib.util
    db = _kb(tmp_path / "kanit.sqlite")
    fa = tmp_path / "iso.fna"
    fa.write_text("".join(f">{k}\n{v}\n" for k, v in _genome().items()))
    bakta = tmp_path / "b.tsv"
    bakta.write_text("#Sequence Id\tType\tStart\tStop\tStrand\tLocus Tag\tGene\tProduct\tDbXrefs\n"
                     "A\tcds\t1900\t2100\t+\tL1\tdfrA17\ttrimethoprim-resistant dihydrofolate reductase\t\n")
    spec = importlib.util.spec_from_file_location("ks", PROJECT_ROOT / "scripts" / "kanit_search.py")
    ks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ks)
    out = tmp_path / "out"
    assert ks.main(["--genome", str(fa), "--organism", "ecoli", "--db", str(db), "--out", str(out),
                    "--annotate", "full", "--bakta-tsv", str(bakta), "--plasmids", "none"]) == 0
    for f in ("calls.csv", "markers.csv", "hits.csv", "report.html", "run.json",
              "figures/signature.svg", "figures/map.svg", "figures/neighbourhood.svg"):
        assert (out / f).exists(), f
    html = (out / "report.html").read_text()
    assert "Research use only" in html and "dfrA17" in html and "<svg" in html
    run = json.loads((out / "run.json").read_text())
    assert run["calls"] == {"ciprofloxacin": "S", "trimethoprim": "R"} and run["n_markers_found"] == 2
