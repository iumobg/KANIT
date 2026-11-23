#!/usr/bin/env python3
"""CARD layer (lib.card_layer): locating unitigs, overlap, majority and pattern state."""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import card_layer as cl  # noqa: E402
from lib import registry  # noqa: E402

pytestmark = pytest.mark.unit


def test_locate_finds_both_strands_and_repeats():
    contigs = {"c1": "AAACCCGGGTTT" + "ACGTT" + "GG", "c2": "TTAACGTAA"}
    hits = cl.locate("ACGTT", contigs)
    # forward on c1 at 12; reverse complement (AACGT) on c2 at 2
    assert ("c1", 12, 17) in hits and ("c2", 2, 7) in hits
    assert cl.locate("acgtt", contigs) == hits                 # case-insensitive
    assert cl.locate("GGGGGGGG", contigs) == []


def test_read_fasta_uses_the_first_word(tmp_path):
    f = tmp_path / "g.fna"
    f.write_text(">accn|g1.con.0001   contig_1   [E. coli]\nacgt\nAC\n>accn|g1.con.0002 x\nGG\n")
    assert cl.read_fasta(f) == {"accn|g1.con.0001": "ACGTAC", "accn|g1.con.0002": "GG"}


def test_pick_genomes_orders_by_quality_then_id():
    qc = pd.DataFrame({"genome_id": ["g1", "g2", "g3", "g4", "g5"],
                       "completeness": [99.0, 100.0, 100.0, 98.0, 100.0],
                       "contamination": [0.5, 1.0, 0.2, 0.1, 0.2]})
    assert cl.pick_genomes(["g1", "g2", "g3", "g4", "g5"], qc) == ["g3", "g5", "g2"]
    assert cl.pick_genomes(["g4"], qc) == ["g4"]


def test_overlap_needs_half_of_the_unitig():
    hits = pd.DataFrame({"contig": ["c1", "c1", "c2"], "start": [100, 140, 100],
                         "end": [200, 300, 200], "aro": ["A", "B", "C"]})
    # unitig [90, 150): 50 bp; overlaps A by 50 (all), B by 10 (< 25)
    got = cl.overlapping(hits, "c1", 90, 150)
    assert got["aro"].tolist() == ["A"]
    # exactly half counts
    assert cl.overlapping(hits, "c1", 75, 125)["aro"].tolist() == ["A"]


def test_majority_rule_and_conservative_hit_state():
    b, nob = (True, True, set(), {"A"}), (False, True, {cl.CO_CARRIED}, {"A"})
    none = (False, False, set(), set())
    assert cl.combine_genomes([b]).state == cl.B1                  # 1 of 1
    assert cl.combine_genomes([b, b]).state == cl.B1               # 2 of 2
    one_of_two = cl.combine_genomes([b, none])                     # 1 of 2: not b
    assert one_of_two.state == cl.HIT_NO_B and one_of_two.reasons == {cl.B_MINORITY}
    assert cl.combine_genomes([b, nob, none]).reasons == {cl.CO_CARRIED, cl.B_MINORITY}
    assert cl.combine_genomes([b, b, none]).state == cl.B1         # 2 of 3
    u = cl.combine_genomes([nob, none, none])                      # any hit is a hit
    assert u.state == cl.HIT_NO_B and u.reasons == {cl.CO_CARRIED}
    assert cl.combine_genomes([none, none]).state == cl.NO_HIT


def test_pattern_takes_the_best_member_state():
    assert cl.pattern_state([cl.NO_HIT, cl.B1, cl.HIT_NO_B]) == cl.B1
    assert cl.pattern_state([cl.NO_HIT, cl.HIT_NO_B]) == cl.HIT_NO_B
    assert cl.pattern_state([cl.NO_HIT]) == cl.NO_HIT


def test_card_drug_classes_follow_the_registry():
    assert registry.card_drug_classes("ceftriaxone") == {"cephalosporin", "cephamycin"}
    assert registry.card_drug_classes("tigecycline") == {"glycylcycline"}
    assert registry.card_drug_classes("not_a_drug") == set()


def test_card_drug_classes_of_a_single_folate_drug_are_its_own():
    """sul genes do not confer trimethoprim resistance, nor dfr genes sulfonamide resistance;
    the combination keeps both terms."""
    assert registry.card_drug_classes("trimethoprim") == {"diaminopyrimidine antibiotic"}
    assert registry.card_drug_classes("sulfamethoxazole") == {"sulfonamide antibiotic"}
    assert registry.card_drug_classes("trimethoprim_sulfamethoxazole") == {
        "diaminopyrimidine antibiotic", "sulfonamide antibiotic"}


# ---------------------------------------------------------------------------
# real RGI 6.0.8 / CARD 4.0.1 rows (E. coli 562.100405): gyrA and parC variant
# models (+ and - strand), CTX-M-27, soxS overexpression without a mutation, AcrF
# ---------------------------------------------------------------------------
SAMPLE = PROJECT_ROOT / "tests" / "data" / "rgi_ecoli_sample.txt"
QUINOLONE = {"fluoroquinolone antibiotic"}
CEPHALOSPORIN = {"cephalosporin", "cephamycin"}


@pytest.fixture(scope="module")
def genome():
    hits = cl.read_rgi(SAMPLE, "g1")
    size = {}
    for h in hits.itertuples():
        size[h.contig] = max(size.get(h.contig, 0), h.end + 100)
    contigs = {c: ["N"] * n for c, n in size.items()}
    for h in hits.itertuples():
        orf = h.dna if h.strand == "+" else cl.revcomp(h.dna)
        contigs[h.contig][h.start:h.end] = list(orf)
    return hits, {c: "".join(s) for c, s in contigs.items()}


def _row(hits, name):
    return hits[hits["aro_name"].str.contains(name, regex=False)].iloc[0]


def test_read_rgi_normalises_the_real_output(genome):
    hits, _ = genome
    assert set(hits["model_type"]) == {"homolog", "variant", "overexpression"}
    gyra = _row(hits, "gyrA")
    assert (gyra["start"], gyra["end"], gyra["strand"]) == (118027, 120655, "+")
    assert cl.drug_classes(gyra["drug_class"]) >= QUINOLONE


def test_mutations_map_to_the_mutant_codon_on_both_strands(genome):
    from Bio.Seq import Seq
    hits, contigs = genome
    gyra = _row(hits, "gyrA")
    iv = {snp: (s, e) for snp, s, e in cl.mutation_intervals(gyra)}
    assert set(iv) == {"S83L", "D87N"}
    s, e = iv["S83L"]
    assert str(Seq(contigs[gyra["contig"]][s:e]).translate()) == "L"
    parc = _row(hits, "parC")                                  # minus strand
    (snp, s, e), = cl.mutation_intervals(parc)
    assert snp == "S80I"
    assert str(Seq(cl.revcomp(contigs[parc["contig"]][s:e])).translate()) == "I"
    assert cl.mutation_intervals(_row(hits, "soxS")) == []     # no reported mutation


def test_evaluate_location_follows_the_rule(genome):
    hits, contigs = genome
    gyra = _row(hits, "gyrA")
    (s83, e83), = [(s, e) for snp, s, e in cl.mutation_intervals(gyra) if snp == "S83L"]
    # a 41-bp unitig centred on the S83L codon, in a quinolone model
    r = cl.evaluate_location(hits, gyra["contig"], s83 - 19, e83 + 19, QUINOLONE, set())
    assert r["allele_aware"][0] is True
    assert r["homolog_only"][0] is False and cl.VARIANT_NOT_COUNTED in r["homolog_only"][2]
    # elsewhere in gyrA: a variant hit whose mutation the unitig does not carry
    far = gyra["start"] + 1500
    r = cl.evaluate_location(hits, gyra["contig"], far, far + 41, QUINOLONE, set())
    assert r["allele_aware"][:2] == (False, True) and cl.ALLELE_NOT_IN in r["allele_aware"][2]
    # CTX-M-27: b in a cephalosporin model, co-carried in a quinolone model,
    # and no b through a homolog hit when the gene is near-universal
    ctx = _row(hits, "CTX-M-27")
    mid = ctx["start"] + 300
    assert cl.evaluate_location(hits, ctx["contig"], mid, mid + 41, CEPHALOSPORIN, set())[
        "allele_aware"][0] is True
    r = cl.evaluate_location(hits, ctx["contig"], mid, mid + 41, QUINOLONE, set())
    assert r["allele_aware"][:2] == (False, True) and r["allele_aware"][2] == {cl.CO_CARRIED}
    r = cl.evaluate_location(hits, ctx["contig"], mid, mid + 41, CEPHALOSPORIN, {ctx["aro"]})
    assert r["allele_aware"][2] == {cl.NEAR_UNIVERSAL}
    # outside every hit: no CARD hit
    r = cl.evaluate_location(hits, ctx["contig"], 0, 41, CEPHALOSPORIN, set())
    assert r["allele_aware"] == (False, False, set(), set())


def test_unitig_state_locates_and_combines(genome):
    hits, contigs = genome
    gyra = _row(hits, "gyrA")
    (s83, e83), = [(s, e) for snp, s, e in cl.mutation_intervals(gyra) if snp == "S83L"]
    seq = contigs[gyra["contig"]][s83 - 19:e83 + 19]
    states, located = cl.unitig_state(cl.revcomp(seq), ["g1"], lambda g: contigs, lambda g: hits,
                                      QUINOLONE, set())       # found on the other strand
    assert located == ["g1"]
    assert states["allele_aware"].state == cl.B1
    assert states["homolog_only"].state == cl.HIT_NO_B
    # found whole in no carrier: no located genome, no CARD hit (2026-10-12)
    states, located = cl.unitig_state("ACGT" * 10, ["g1"], lambda g: contigs, lambda g: hits,
                                      QUINOLONE, set())
    assert located == [] and states["allele_aware"].state == cl.NO_HIT
    assert states["allele_aware"].n_located == 0


def test_unitig_state_skips_a_carrier_whose_assembly_breaks_inside_the_unitig(genome):
    """A carrier holds the unitig's k-mers but not the unitig whole when its contig ends inside
    it (K. pneumoniae tigecycline, 2026-10-12); the next carrier by quality is used."""
    hits, contigs = genome
    gyra = _row(hits, "gyrA")
    (s83, e83), = [(s, e) for snp, s, e in cl.mutation_intervals(gyra) if snp == "S83L"]
    seq = contigs[gyra["contig"]][s83 - 19:e83 + 19]
    broken = {name: c.replace(seq, seq[:30] + "NNNNN" + seq[30:]) for name, c in contigs.items()}
    genomes = {"g0": broken, "g1": contigs, "g2": contigs, "g3": contigs}
    states, located = cl.unitig_state(seq, ["g0", "g1", "g2", "g3"], genomes.__getitem__,
                                      lambda g: hits, QUINOLONE, set(), max_located=2)
    assert located == ["g1", "g2"]
    assert states["allele_aware"].n_located == 2 and states["allele_aware"].state == cl.B1


def test_near_universal_aros_counts_genomes():
    hits = pd.DataFrame({"genome_id": ["g1", "g2", "g3", "g1"], "aro": ["A", "A", "A", "B"]})
    assert cl.near_universal_aros(hits, ["g1", "g2", "g3"], 0.95) == {"A"}
    assert cl.near_universal_aros(hits, ["g1", "g2", "g3", "g4"], 0.95) == set()


def test_annotate_end_to_end(genome, tmp_path):
    from lib import matrix_store
    hits1, contigs = genome
    gyra, ctx = _row(hits1, "gyrA"), _row(hits1, "CTX-M-27")
    (s83, e83), = [(s, e) for snp, s, e in cl.mutation_intervals(gyra) if snp == "S83L"]
    free = "ACGTTGCAAGGCTTACCGGATCCATGGTACCAGT" * 2           # outside every hit
    c = gyra["contig"]
    contigs = dict(contigs)
    contigs[c] = free + contigs[c][len(free):]
    unitigs = {
        "S83": contigs[c][s83 - 19:e83 + 19],
        "FAR": contigs[c][gyra["start"] + 1500:gyra["start"] + 1541],
        "CTX": contigs[ctx["contig"]][ctx["start"] + 300:ctx["start"] + 341],
        "FREE": free[:41],
    }
    genomes = [f"g{i:02d}" for i in range(1, 13)]
    carriers = {"S83": [1, 2, 3, 9], "FAR": [1, 2, 4, 10], "CTX": [1, 3, 4, 11], "FREE": [2, 3, 4, 12]}
    rtab = tmp_path / "u.rtab"
    with open(rtab, "w") as f:
        f.write("Unitig_sequence\t" + "\t".join(genomes) + "\n")
        for name, seq in unitigs.items():
            bits = ["1" if int(g[1:]) in carriers[name] else "0" for g in genomes]
            f.write(seq + "\t" + "\t".join(bits) + "\n")
    matrix_store.build_store(rtab, tmp_path / "store", min_support=2)
    gdf = pd.DataFrame({"Genome ID": genomes, "label": [i % 2 for i in range(12)], "lineage": 0})
    matrix_store.build_model_matrix(tmp_path / "store", gdf, tmp_path / "model", min_support=2)
    mm, store = matrix_store.ModelMatrix(tmp_path / "model"), matrix_store.Store(tmp_path / "store")
    qc = pd.DataFrame({"genome_id": genomes, "completeness": [100 - i for i in range(12)],
                       "contamination": 0.5})
    hits = pd.concat([hits1.assign(genome_id=g) for g in genomes[:8]], ignore_index=True)
    pid = {}
    m = mm.members()
    seq_of = dict(zip(m["unitig_index"], store.sequences(m["unitig_index"]), strict=True))
    for u, p in zip(m["unitig_index"], m["pattern_id"], strict=True):
        pid[next(k for k, s in unitigs.items() if s == seq_of[u])] = p

    def run(targets, column="allele_aware_state"):
        _, pats, _ = cl.annotate(sorted(pid.values()), mm, store, qc, hits, lambda g: contigs,
                                 targets)
        st = dict(zip(pats["pattern_id"], pats[column], strict=True))
        return {k: st[p] for k, p in pid.items()}

    assert run(QUINOLONE) == {"S83": cl.B1, "FAR": cl.HIT_NO_B, "CTX": cl.HIT_NO_B, "FREE": cl.NO_HIT}
    assert run(QUINOLONE, "allele_aware_reasons") == {
        "S83": "", "FAR": cl.ALLELE_NOT_IN, "CTX": cl.CO_CARRIED, "FREE": ""}
    assert run(QUINOLONE, "homolog_only_state")["S83"] == cl.HIT_NO_B
    assert run(QUINOLONE, "homolog_only_reasons")["S83"] == cl.VARIANT_NOT_COUNTED
    assert run(CEPHALOSPORIN) == {"S83": cl.HIT_NO_B, "FAR": cl.HIT_NO_B, "CTX": cl.B1,
                                  "FREE": cl.NO_HIT}
