"""The KANIT search of one assembly: the knowledge base's biomarkers in the genome, and a call
for every antibiotic its organism has a model for.

The member unitigs of every graded candidate pattern of the organism's models are searched in
the assembly on either strand by exact match, as unitig-caller's simple mode searched them in
the external isolates (19_cabbage.py lookup-call). A pattern is present when at least half of
its members are found (lib.cabbage_lookup.PRESENT_SHARE). A marker is a present pattern of
grade ``weak`` or better, with the direction of its prevalence difference; a model's antibiotic
is called resistant when the assembly carries a marker of the rule in the resistant direction
(lib.cabbage_lookup). The knowledge base is the only input besides the assembly.

A call is a research result, not a susceptibility test result.
"""
from __future__ import annotations

import gzip
import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lib import cabbage_lookup as cl
from lib.card_layer import revcomp

SEED = 31                          # the shortest unitig: k of the unitig graph
DEFAULT_RULE = "upto_strong_novel"  # chosen on the training genomes, before CABBAGE (19_cabbage.py lookup-select)
RULES = cl.RULES
GRADE_ORDER = ("confirmed", "strong_novel", "candidate", "weak", "none")


# ---- the assembly ---------------------------------------------------------------------------------
def parse_fasta(text: str) -> dict[str, str]:
    """{contig id (first word of the header): uppercase sequence}."""
    seqs: dict[str, list[str]] = {}
    name = None
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0] if len(line) > 1 else f"contig_{len(seqs) + 1}"
            if name in seqs:
                raise ValueError(f"duplicate contig id: {name}")
            seqs[name] = []
        elif name is None:
            raise ValueError("not a FASTA file: sequence before the first header")
        else:
            seqs[name].append(line.upper())
    out = {k: "".join(v) for k, v in seqs.items()}
    if not any(out.values()):
        raise ValueError("no sequence in the assembly")
    return out


def read_assembly(source) -> dict[str, str]:
    """An assembly from a path or from bytes (an uploaded file), plain FASTA or gzip."""
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return parse_fasta(raw.decode("ascii", errors="replace"))


# ---- the knowledge base ------------------------------------------------------------------------
def organisms(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT organism_id FROM model ORDER BY organism_id")]


def models(conn: sqlite3.Connection, organism: str) -> pd.DataFrame:
    """model_id, antibiotic_id, drug_class of the organism's models."""
    return pd.read_sql("""SELECT m.model_id, m.antibiotic_id, a.drug_class FROM model m
                            JOIN antibiotic a ON a.antibiotic_id = m.antibiotic_id
                           WHERE m.organism_id = ? ORDER BY m.model_id""", conn, params=(organism,))


def release(conn: sqlite3.Connection) -> dict:
    cur = conn.execute("SELECT * FROM release")
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row, strict=True)) if row else {}


def load_markers(conn: sqlite3.Connection, organism: str) -> pd.DataFrame:
    """Every graded candidate pattern of the organism's models (model route), weak or better:
    model_id, antibiotic_id, drug_class, pattern_id, grade, card_state, direction, gain_rank,
    n_members, card_genes, and the NCBI context of its longest member (context_gene,
    context_product, context_plasmid_share; shown beside the grade, never as evidence)."""
    q = """SELECT c.model_id, m.antibiotic_id, a.drug_class, c.pattern_id, g.grade, g.card_state,
                  pr.direction, c.gain_rank, p.n_members,
                  (SELECT group_concat(DISTINCT ar.name)
                     FROM pattern_member pm
                     JOIN card_hit h ON h.model_id = pm.model_id AND h.unitig_id = pm.unitig_id
                     JOIN aro ar ON ar.aro_accession = h.aro_accession
                    WHERE pm.model_id = c.model_id AND pm.pattern_id = c.pattern_id) AS card_genes
             FROM candidate c
             JOIN model m ON m.model_id = c.model_id
             JOIN antibiotic a ON a.antibiotic_id = m.antibiotic_id
             JOIN pattern p ON p.model_id = c.model_id AND p.pattern_id = c.pattern_id
             JOIN grade g ON g.model_id = c.model_id AND g.pattern_id = c.pattern_id
                         AND g.primary_mode = 1
             JOIN prevalence_result pr ON pr.model_id = c.model_id AND pr.pattern_id = c.pattern_id
            WHERE m.organism_id = ? AND c.route = 'model'"""
    mk = pd.read_sql(q, conn, params=(organism,))
    mk = mk[mk["grade"].isin(cl.GRADES)].reset_index(drop=True)
    if mk["direction"].isna().any():
        raise ValueError("a marker has no prevalence direction")
    ctx = pd.read_sql(
        """SELECT pm.model_id, pm.pattern_id, u.length, c.gene AS context_gene,
                  c.product AS context_product, c.plasmid_share AS context_plasmid_share
             FROM pattern_member pm
             JOIN model m ON m.model_id = pm.model_id
             JOIN unitig u ON u.unitig_id = pm.unitig_id
             LEFT JOIN unitig_context c ON c.unitig_id = pm.unitig_id AND c.organism_id = m.organism_id
            WHERE m.organism_id = ?""", conn, params=(organism,))
    ctx = (ctx.sort_values(["model_id", "pattern_id", "length"], ascending=[True, True, False])
              .drop_duplicates(["model_id", "pattern_id"]).drop(columns="length"))
    mk = mk.merge(ctx, on=["model_id", "pattern_id"], how="left")
    mk["card_genes"] = mk["card_genes"].fillna("").str.replace(",", "; ")
    return mk


def load_members(conn: sqlite3.Connection, organism: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """The member unitigs of the organism's candidate patterns and {unitig_id: sequence}."""
    members = pd.read_sql(
        """SELECT pm.model_id, pm.pattern_id, pm.unitig_id
             FROM pattern_member pm
             JOIN model m ON m.model_id = pm.model_id
             JOIN candidate c ON c.model_id = pm.model_id AND c.pattern_id = pm.pattern_id
            WHERE m.organism_id = ? AND c.route = 'model'""", conn, params=(organism,))
    seqs = dict(conn.execute(
        """SELECT DISTINCT u.unitig_id, u.sequence FROM unitig u
             JOIN pattern_member pm ON pm.unitig_id = u.unitig_id
             JOIN model m ON m.model_id = pm.model_id
            WHERE m.organism_id = ?""", (organism,)).fetchall())
    return members, seqs


# ---- the search --------------------------------------------------------------------------------------
HIT_COLUMNS = ["unitig_id", "contig", "start", "end", "strand"]


def find_unitigs(contigs: dict[str, str], unitigs: dict[str, str], seed: int = SEED) -> pd.DataFrame:
    """Every exact occurrence of the unitigs in the contigs, on either strand: unitig_id, contig,
    start (0-based), end (exclusive), strand ("+" when the stored sequence reads on the contig)."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    for uid, s in unitigs.items():
        s = s.upper()
        if len(s) < seed:
            raise ValueError(f"unitig {uid} is shorter than the seed ({len(s)} < {seed})")
        for strand, q in (("+", s), ("-", revcomp(s))):
            index.setdefault(q[:seed], []).append((uid, strand, q))
    rows = []
    for name, seq in contigs.items():
        get = index.get
        for i in range(len(seq) - seed + 1):
            cands = get(seq[i:i + seed])
            if cands:
                for uid, strand, q in cands:
                    if seq.startswith(q, i):
                        rows.append((uid, name, i, i + len(q), strand))
    hits = pd.DataFrame(rows, columns=HIT_COLUMNS)
    # a palindromic unitig matches both strands at one place
    return hits.drop_duplicates(["unitig_id", "contig", "start", "end"]).reset_index(drop=True)


def pattern_presence(members: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """model_id, pattern_id, n_members, n_found, present (n_found / n_members ≥ PRESENT_SHARE)."""
    found = set(hits["unitig_id"])
    m = members.assign(found=members["unitig_id"].isin(found))
    p = (m.groupby(["model_id", "pattern_id"])["found"].agg(n_members="size", n_found="sum")
          .reset_index())
    p["present"] = p["n_found"] / p["n_members"] >= cl.PRESENT_SHARE
    return p


def calls(markers: pd.DataFrame, presence: pd.DataFrame, model_table: pd.DataFrame,
          rule: str = DEFAULT_RULE) -> pd.DataFrame:
    """One row per model of ``model_table``: the call of the rule, with the present markers behind
    it, and the secondary score (R markers present minus S markers present, all grades up to
    weak). A model without markers is called S."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule}; one of {RULES}")
    present = set(map(tuple, presence.loc[presence["present"], ["model_id", "pattern_id"]]
                      .itertuples(index=False, name=None)))
    rows = []
    by_model = dict(tuple(markers.groupby("model_id")))
    for mid, ab, cls in model_table[["model_id", "antibiotic_id", "drug_class"]].itertuples(index=False):
        mk = by_model.get(mid, markers.iloc[0:0])
        on = {p for p in cl.rule_patterns(mk, rule) if (mid, p) in present}
        r_all = {p for p in cl.rule_patterns(mk, "upto_weak", "R") if (mid, p) in present}
        s_all = {p for p in cl.rule_patterns(mk, "upto_weak", "S") if (mid, p) in present}
        drv = mk[mk["pattern_id"].isin(on)].copy()
        drv["_r"] = drv["grade"].map(GRADE_ORDER.index)
        drv = drv.sort_values(["_r", "gain_rank"])
        rows.append({"model_id": mid, "antibiotic_id": ab, "drug_class": cls, "rule": rule,
                     "call": "R" if on else "S", "n_markers_rule": len(on),
                     "n_markers_r": len(r_all), "n_markers_s": len(s_all),
                     "score": len(r_all) - len(s_all),
                     "evidence": "; ".join(_marker_label(r) for r in drv.itertuples())})
    return pd.DataFrame(rows)


def _marker_label(r) -> str:
    gene = r.card_genes or (r.context_gene if isinstance(r.context_gene, str) and r.context_gene else "")
    return f"{r.grade} #{r.pattern_id}" + (f" ({gene})" if gene else "")


@dataclass
class Result:
    organism: str
    rule: str
    contigs: pd.DataFrame        # contig, length
    hits: pd.DataFrame           # one row per unitig occurrence and pattern it belongs to
    markers: pd.DataFrame        # every marker with its presence
    calls: pd.DataFrame          # one row per model
    release: dict


def search(conn: sqlite3.Connection, contigs: dict[str, str], organism: str,
           rule: str = DEFAULT_RULE) -> Result:
    if organism not in organisms(conn):
        raise ValueError(f"no model for organism {organism!r}; one of {organisms(conn)}")
    markers = load_markers(conn, organism)
    members, seqs = load_members(conn, organism)
    occ = find_unitigs(contigs, seqs)
    presence = pattern_presence(members, occ)
    markers = markers.merge(presence.rename(columns={"n_members": "_n"}), on=["model_id", "pattern_id"],
                            how="left")
    if (markers["_n"].notna() & (markers["_n"] != markers["n_members"])).any():
        raise ValueError("the knowledge base's member counts disagree with its pattern table")
    markers = markers.drop(columns="_n")
    markers["n_found"] = markers["n_found"].fillna(0).astype(int)
    markers["present"] = markers["present"].fillna(False).astype(bool)
    hits = (occ.merge(members, on="unitig_id")
               .merge(markers[["model_id", "antibiotic_id", "drug_class", "pattern_id", "grade",
                               "direction", "present", "card_genes", "context_gene",
                               "context_product", "context_plasmid_share"]],
                      on=["model_id", "pattern_id"]))
    hits = hits[hits["present"]].drop(columns="present").sort_values(["contig", "start"])
    lengths = pd.DataFrame({"contig": list(contigs), "length": [len(s) for s in contigs.values()]})
    return Result(organism, rule, lengths, hits.reset_index(drop=True), markers,
                  calls(markers, presence, models(conn, organism), rule), release(conn))
