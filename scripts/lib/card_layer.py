"""CARD layer of the grading rule: the state of every candidate unitig.

A candidate unitig is located by exact match (both strands) in up to three
genomes that carry it: the highest CheckM2 completeness first, then the lowest
contamination, then the genome identifier. In each located genome it is assigned
to every RGI hit (Perfect or Strict) that overlaps at least half of its length.
A hit gives ``b`` = 1 when its CARD drug class matches the modelled antibiotic
and it is

* a homolog-model hit to a gene that is not near-universal in the model's
  genomes (present in fewer than ``near_universal`` of them), or
* a variant- or overexpression-model hit whose reported mutation lies within the
  unitig (the unitig is a substring of a genome carrying the mutation, so it
  carries the resistant allele when it covers the position).

``b`` holds for the unitig when it holds in more than half of its located
genomes. A unitig with a hit but without ``b`` keeps the reasons of its hits:
``co_carried_known_gene`` (drug class does not match), ``near_universal_gene``,
``allele_not_in_unitig``, and ``b_in_minority_of_located_genomes`` when ``b`` holds
in some located genomes but not in more than half. Any hit in
any located genome counts as a CARD hit, so
a unitig is never called novel next to a known gene. A pattern takes the best
state of its member unitigs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

B1, HIT_NO_B, NO_HIT = "b", "card_hit_without_b", "no_card_hit"
CO_CARRIED, NEAR_UNIVERSAL, ALLELE_NOT_IN = ("co_carried_known_gene", "near_universal_gene",
                                             "allele_not_in_unitig")
B_MINORITY = "b_in_minority_of_located_genomes"
VARIANT_NOT_COUNTED = "variant_not_counted"     # homolog_only sensitivity analysis
MODES = ("allele_aware", "homolog_only")

# Evidence of one located occurrence or genome: (b, any_hit, reasons, aros).
Evidence = tuple[bool, bool, set[str], set[str]]

# RGI main --output_file <x>: <x>.txt, tab-separated (RGI 6). Coordinates are
# 1-based and inclusive; Contig is the FASTA identifier.
_RGI_COLUMNS = {"Contig": "contig", "Start": "start", "Stop": "end",
                "Orientation": "strand", "Cut_Off": "cut_off", "ARO": "aro",
                "Best_Hit_ARO": "aro_name", "Model_type": "model_type",
                "SNPs_in_Best_Hit_ARO": "snps", "Drug Class": "drug_class",
                "AMR Gene Family": "gene_family", "Resistance Mechanism": "mechanism",
                "Predicted_DNA": "dna", "Predicted_Protein": "protein",
                "CARD_Protein_Sequence": "ref_protein", "Model_ID": "model_id"}
MODEL_TYPES = {"protein homolog model": "homolog", "protein variant model": "variant",
               "protein overexpression model": "overexpression",
               "rRNA gene variant model": "rrna_variant"}
_SNP = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def read_fasta(path) -> dict[str, str]:
    """{sequence id (first word of the header): uppercase sequence}."""
    seqs: dict[str, list[str]] = {}
    name = None
    with open(path, encoding="ascii") as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip().upper())
    return {k: "".join(v) for k, v in seqs.items()}


def locate(seq: str, contigs: dict[str, str]) -> list[tuple[str, int, int]]:
    """Every exact occurrence of ``seq`` on either strand: (contig, start, end),
    0-based, end exclusive, in contig coordinates."""
    seq = seq.upper()
    out = []
    for probe in {seq, revcomp(seq)}:
        for contig, s in contigs.items():
            i = s.find(probe)
            while i >= 0:
                out.append((contig, i, i + len(probe)))
                i = s.find(probe, i + 1)
    return sorted(set(out))


def pick_genomes(carriers, qc_table: pd.DataFrame, k: int = 3) -> list[str]:
    """Up to ``k`` carrier genomes: highest completeness, lowest contamination, then id."""
    q = qc_table[qc_table["genome_id"].astype(str).isin({str(g) for g in carriers})].copy()
    q["genome_id"] = q["genome_id"].astype(str)
    q = q.sort_values(["completeness", "contamination", "genome_id"],
                      ascending=[False, True, True])
    return q["genome_id"].head(k).tolist()


def overlapping(hits: pd.DataFrame, contig: str, start: int, end: int,
                min_share: float = 0.5) -> pd.DataFrame:
    """Hits on ``contig`` overlapping at least ``min_share`` of [start, end)."""
    h = hits[hits["contig"] == contig]
    ov = (h["end"].clip(upper=end) - h["start"].clip(lower=start)).clip(lower=0)
    return h[ov >= min_share * (end - start)]


@dataclass
class UnitigState:
    state: str = NO_HIT
    reasons: set[str] = field(default_factory=set)
    aros: set[str] = field(default_factory=set)
    n_located: int = 0
    n_b: int = 0


def combine_genomes(per_genome: list[Evidence]) -> UnitigState:
    """Majority over located genomes. Each item: (b, any_hit, reasons, aros)."""
    u = UnitigState(n_located=len(per_genome))
    u.n_b = sum(1 for b, *_ in per_genome if b)
    for _, any_hit, reasons, aros in per_genome:
        u.aros |= aros
        if any_hit:
            u.reasons |= reasons
    if u.n_located and u.n_b > u.n_located / 2:
        u.state, u.reasons = B1, set()
    elif any(hit for _, hit, _, _ in per_genome):
        u.state = HIT_NO_B
        if u.n_b:                   # b in some located genomes, not in more than half
            u.reasons.add(B_MINORITY)
    return u


def pattern_state(member_states) -> str:
    """b if any member has b; a CARD hit without b if any member has a hit; else none."""
    states = set(member_states)
    if B1 in states:
        return B1
    if HIT_NO_B in states:
        return HIT_NO_B
    return NO_HIT


# ---------------------------------------------------------------------------
# RGI hits
# ---------------------------------------------------------------------------
def read_rgi(path, genome_id: str) -> pd.DataFrame:
    """Perfect and Strict hits of one genome, 0-based half-open coordinates."""
    d = pd.read_csv(path, sep="\t", dtype=str)
    missing = sorted(set(_RGI_COLUMNS) - set(d.columns))
    if missing:
        raise ValueError(f"{path}: RGI columns missing: {missing}")
    d = d[list(_RGI_COLUMNS)].rename(columns=_RGI_COLUMNS)
    unknown = sorted(set(d["model_type"]) - set(MODEL_TYPES))
    if unknown:
        raise ValueError(f"{path}: unknown RGI model types {unknown}")
    d = d[d["cut_off"].isin(["Perfect", "Strict"])].copy()
    if not d["strand"].isin(["+", "-"]).all():
        raise ValueError(f"{path}: Orientation must be + or -")
    d["model_type"] = d["model_type"].map(MODEL_TYPES)
    d["start"] = d["start"].astype(int) - 1
    d["end"] = d["end"].astype(int)
    d["snps"] = d["snps"].fillna("")
    d.insert(0, "genome_id", str(genome_id))
    return d.reset_index(drop=True)


def drug_classes(value) -> set[str]:
    return {x.strip().lower() for x in str(value).split(";") if x.strip() and x != "nan"}


def near_universal_aros(hits: pd.DataFrame, genomes, threshold: float) -> set[str]:
    """AROs that RGI detects in at least ``threshold`` of the given genomes."""
    genomes = {str(g) for g in genomes}
    h = hits[hits["genome_id"].astype(str).isin(genomes)]
    share = h.groupby("aro")["genome_id"].nunique() / max(len(genomes), 1)
    return set(share[share >= threshold].index)


def rrna_references(card_json) -> dict[str, str]:
    """{model id: reference DNA} of CARD's rRNA gene variant models."""
    with open(card_json, encoding="utf-8") as f:
        doc = json.load(f)
    out = {}
    for key, model in doc.items():
        if not isinstance(model, dict) or model.get("model_type") != "rRNA gene variant model":
            continue
        for seq in (model.get("model_sequences", {}).get("sequence", {}) or {}).values():
            dna = seq.get("dna_sequence", {}).get("sequence")
            if dna:
                out[str(model.get("model_id", key))] = dna.upper()
                break
    return out


@lru_cache(maxsize=2)
def _aligner(kind: str):
    from Bio.Align import PairwiseAligner, substitution_matrices
    a = PairwiseAligner(mode="global")
    if kind == "protein":
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score, a.extend_gap_score = -10, -0.5
    else:
        a.match_score, a.mismatch_score = 2, -3
        a.open_gap_score, a.extend_gap_score = -5, -2
    return a


def map_position(query: str, ref: str, ref_pos: int, kind: str) -> int | None:
    """0-based index of the query residue aligned to reference residue ``ref_pos`` (1-based)."""
    aln = _aligner(kind).align(query, ref)[0]
    target = ref_pos - 1
    for (qs, _), (rs, re_) in zip(*aln.aligned, strict=True):
        if rs <= target < re_:
            return int(qs + (target - rs))
    return None


def mutation_intervals(hit, ref_dna: dict | None = None) -> list[tuple[str, int, int]]:
    """Contig intervals of the reported mutations of a variant, overexpression or
    rRNA hit whose position maps to the genome's ORF and shows the mutant residue."""
    out = []
    for snp in [s.strip() for s in str(hit["snps"]).split(",") if s.strip()]:
        m = _SNP.match(snp)
        if not m:
            continue
        pos, mut = int(m.group(2)), m.group(3)
        if hit["model_type"] == "rrna_variant":
            ref = (ref_dna or {}).get(str(hit["model_id"]))
            query = str(hit["dna"]).upper()
            if not ref:
                continue
            q = map_position(query, ref, pos, "dna")
            if q is None or q >= len(query) or query[q] != mut:
                continue
            off, width = q, 1
        else:
            query = str(hit["protein"]).rstrip("*")
            q = map_position(query, str(hit["ref_protein"]).rstrip("*"), pos, "protein")
            if q is None or q >= len(query) or query[q] != mut:
                continue
            off, width = 3 * q, 3
        s = hit["start"] + off if hit["strand"] == "+" else hit["end"] - off - width
        out.append((snp, int(s), int(s + width)))
    return out


def evaluate_location(hits: pd.DataFrame, contig: str, start: int, end: int, targets: set[str],
                      near_universal: set[str], ref_dna: dict | None = None,
                      min_overlap: float = 0.5) -> dict[str, Evidence]:
    """{mode: (b, any_hit, reasons, aros)} for one located occurrence of a unitig."""
    ov = overlapping(hits, contig, start, end, min_overlap)
    if ov.empty:
        return {mode: (False, False, set(), set()) for mode in MODES}
    b = dict.fromkeys(MODES, False)
    reasons: dict[str, set[str]] = {mode: set() for mode in MODES}
    aros: set[str] = set()
    for _, hit in ov.iterrows():
        aros.add(str(hit["aro"]))
        if not drug_classes(hit["drug_class"]) & targets:
            for mode in MODES:
                reasons[mode].add(CO_CARRIED)
            continue
        if hit["model_type"] == "homolog":
            for mode in MODES:
                if str(hit["aro"]) in near_universal:
                    reasons[mode].add(NEAR_UNIVERSAL)
                else:
                    b[mode] = True
            continue
        reasons["homolog_only"].add(VARIANT_NOT_COUNTED)
        if any(start <= s and e <= end for _, s, e in mutation_intervals(hit, ref_dna)):
            b["allele_aware"] = True
        else:
            reasons["allele_aware"].add(ALLELE_NOT_IN)
    return {mode: (b[mode], True, reasons[mode], set(aros)) for mode in MODES}


def _merge(items: list[Evidence]) -> Evidence:
    """Several occurrences in one genome: b or a hit in any of them, all reasons and AROs."""
    reasons: set[str] = set()
    aros: set[str] = set()
    for _, _, r, a in items:
        reasons |= r
        aros |= a
    return any(x[0] for x in items), any(x[1] for x in items), reasons, aros


def unitig_state(seq: str, carriers: list[str], contigs_of, hits_of, targets: set[str],
                 near_universal: set[str], ref_dna: dict | None = None,
                 min_overlap: float = 0.5, max_located: int = 3) -> tuple[dict, list[str]]:
    """({mode: UnitigState}, located genomes) of one unitig.

    The unitig is located in the first ``max_located`` of ``carriers`` (best first) that
    contain all of it. A carrier can hold the unitig's k-mers without holding it whole,
    when its draft assembly breaks inside the unitig; such a carrier is skipped. A unitig
    found whole in no carrier has no located genome and no CARD hit.
    ``contigs_of(genome)`` returns the genome's contigs, ``hits_of(genome)`` its RGI hits."""
    per: dict[str, list[Evidence]] = {mode: [] for mode in MODES}
    located: list[str] = []
    for g in carriers:
        if len(located) == max_located:
            break
        locs = locate(seq, contigs_of(g))
        if not locs:
            continue
        located.append(g)
        hits = hits_of(g)
        found = [evaluate_location(hits, contig, s, e, targets, near_universal, ref_dna,
                                   min_overlap) for contig, s, e in locs]
        for mode in MODES:
            per[mode].append(_merge([f[mode] for f in found]))
    return {mode: combine_genomes(per[mode]) for mode in MODES}, located


# the columns of card_unitigs.csv and card_patterns.csv (also when there is no pattern)
UNITIG_COLUMNS = ("pattern_id", "unitig_index", "sequence", "length", "located_genomes",
                  *(f"{m}_{x}" for m in MODES for x in ("state", "reasons", "aros", "n_b")))
PATTERN_COLUMNS = ("pattern_id", "n_members",
                   *(f"{m}_{x}" for m in MODES for x in ("state", "reasons", "aros")))


def annotate(patterns, mm, store, qc_table: pd.DataFrame, hits: pd.DataFrame, contigs_of,
             targets: set[str], *, near_universal_share: float = 0.95, max_located: int = 3,
             min_overlap: float = 0.5, ref_dna: dict | None = None):
    """CARD state of every member unitig of the candidate patterns of one model.

    Returns (unitigs, patterns, summary): one row per unitig and per pattern, each
    with the state, reasons and AROs of both modes. A pattern's reasons are those of
    its members and are kept only when it has a CARD hit without b."""
    members = mm.members()
    genomes = mm.genomes["Genome ID"].astype(str).to_numpy()
    near = near_universal_aros(hits, genomes, near_universal_share)
    by_genome = {g: d for g, d in hits.groupby(hits["genome_id"].astype(str))}
    empty = hits.iloc[0:0]
    want = members[members["pattern_id"].isin(list(patterns))]
    seqs = dict(zip(want["unitig_index"], store.sequences(want["unitig_index"]), strict=True))
    urows, prows = [], []
    n_not_located = n_fewer = 0
    for pid in patterns:
        carriers = genomes[mm.pattern(int(pid)).astype(bool)]
        ranked = pick_genomes(carriers, qc_table, len(carriers))
        member_states: dict[str, list[UnitigState]] = {mode: [] for mode in MODES}
        for u in want.loc[want["pattern_id"] == pid, "unitig_index"]:
            states, located = unitig_state(seqs[u], ranked, contigs_of,
                                           lambda g: by_genome.get(g, empty), targets, near,
                                           ref_dna, min_overlap, max_located)
            n_not_located += not located
            n_fewer += 0 < len(located) < min(max_located, len(ranked))
            row = {"pattern_id": int(pid), "unitig_index": int(u), "sequence": seqs[u],
                   "length": len(seqs[u]), "located_genomes": ";".join(located)}
            for mode, s in states.items():
                member_states[mode].append(s)
                row.update({f"{mode}_state": s.state, f"{mode}_reasons": ";".join(sorted(s.reasons)),
                            f"{mode}_aros": ";".join(sorted(s.aros)), f"{mode}_n_b": s.n_b})
            urows.append(row)
        prow: dict[str, int | str] = {"pattern_id": int(pid),
                                      "n_members": len(member_states["allele_aware"])}
        for mode, ss in member_states.items():
            state = pattern_state(x.state for x in ss)
            prow[f"{mode}_state"] = state
            prow[f"{mode}_reasons"] = (";".join(sorted({r for x in ss for r in x.reasons}))
                                       if state == HIT_NO_B else "")
            prow[f"{mode}_aros"] = ";".join(sorted({a for x in ss for a in x.aros}))
        prows.append(prow)
    unitigs = pd.DataFrame(urows, columns=list(UNITIG_COLUMNS))
    pats = pd.DataFrame(prows, columns=list(PATTERN_COLUMNS))
    summary = {"n_patterns": len(pats), "n_unitigs": len(unitigs), "targets": sorted(targets),
               "n_unitigs_not_located": n_not_located, "n_unitigs_located_in_fewer": n_fewer,
               "near_universal_aros": sorted(near),
               "patterns_by_state": {mode: pats[f"{mode}_state"].value_counts().to_dict()
                                     for mode in MODES} if len(pats) else {}}
    return unitigs, pats, summary
