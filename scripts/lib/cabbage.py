"""External validation on CABBAGE: the eligible isolates.

CABBAGE's phenotypes (EBI AMR Portal) of the study organisms pass the phenotype rules of lib/bvbrc.py, and
every isolate that BV-BRC holds is removed, so that no model has seen an external isolate:
an isolate with a record from BV-BRC (`PATRIC` among its sources), and one whose BioSample,
SRA or assembly accession is that of a genome of the snapshots' BV-BRC queries. The
assembly of an eligible isolate is the GenBank assembly CABBAGE names.
"""
from __future__ import annotations

import re

import pandas as pd

from lib import registry
from lib.bvbrc import eucast_or_clsi

COLUMNS = ["BioSample_ID", "SRA_accession", "assembly_ID", "species", "database",
           "antibiotic_name", "ast_standard", "resistance_phenotype", "used_ECOFF"]
LABELS = {"resistant": 1, "susceptible": 0}
BVBRC_SOURCE = "PATRIC"
STEPS = ["species", "standard", "phenotype", "ecoff", "conflict", "bvbrc_source",
         "bvbrc_genome", "assembly"]
_SEP = re.compile(r"[,;\s]+")
_SLUG = re.compile(r"[\s/\-]+")


def species_index(organisms: dict) -> dict[str, str]:
    """CABBAGE species name -> organism id, from the registry's species_names."""
    return {name.strip(): org for org, block in organisms.items()
            for name in block.get("species_names") or []}


def antibiotic(name) -> str | None:
    """The registry's spelling of a CABBAGE antibiotic name. A name the registry does not
    know as written is tried in snake case ('cefpodoxime-clavulanic acid'); an unknown
    drug keeps that form and matches no panel drug."""
    raw = "" if name is None else str(name).strip()
    if not raw or raw.lower() == "nan":
        return None
    canon = registry.normalize_antibiotic(raw)
    if canon != raw:
        return canon
    return registry.normalize_antibiotic(_SLUG.sub("_", raw.lower()))


def accessions(values) -> set[str]:
    """Upper-case accessions of a column whose cells may list several."""
    out: set[str] = set()
    for v in values:
        if v is None or pd.isna(v):
            continue
        out |= {t for t in _SEP.split(str(v).strip().upper()) if t}
    return out


def unversioned(acc: str) -> str:
    """GCA_000001405.15 -> GCA_000001405."""
    return str(acc).strip().upper().split(".")[0]


def bvbrc_ids(genomes: pd.DataFrame) -> dict[str, set[str]]:
    """The accessions of the genomes of the snapshots' BV-BRC queries (genomes.csv)."""
    return {"biosample": accessions(genomes["biosample_accession"]),
            "sra": accessions(genomes["sra_accession"]),
            "assembly": {unversioned(a) for a in accessions(genomes["assembly_accession"])}}


def _assembly_of(values) -> str | None:
    """The one GenBank assembly of an isolate: its latest version; None when the isolate
    has none or names two different assemblies."""
    accs = sorted(a for a in accessions(values) if a.startswith("GCA_"))
    if len({unversioned(a) for a in accs}) != 1:
        return None
    return max(accs, key=lambda a: int(a.split(".")[1]) if "." in a else 0)


def select(ph: pd.DataFrame, organisms: dict, ids: dict[str, set[str]]
           ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(isolates, phenotypes, counts) of the eligible external isolates.

    isolates: organism, biosample_id, assembly_id, sra_accession, sources.
    phenotypes: organism, biosample_id, antibiotic, label (1 R, 0 S), n_records, how
    ('single' or 'majority'). counts: per organism, the isolates left after every step
    (STEPS) and the isolate × antibiotic cells resolved, tied and kept."""
    missing = sorted(set(COLUMNS) - set(ph.columns))
    if missing:
        raise ValueError(f"CABBAGE columns missing: {missing}")
    full = ph[COLUMNS].copy()
    full["BioSample_ID"] = full["BioSample_ID"].astype(str).str.strip()
    full["organism"] = full["species"].astype(str).str.strip().map(species_index(organisms))
    counts: dict[str, dict] = {o: {} for o in organisms}

    def tally(step: str, frame: pd.DataFrame) -> None:
        n = frame.groupby("organism")["BioSample_ID"].nunique()
        for o in counts:
            counts[o][step] = int(n.get(o, 0))

    d = full[full["organism"].notna()]
    tally("species", d)
    d = d[d["ast_standard"].fillna("").map(eucast_or_clsi)]
    tally("standard", d)
    pheno = d["resistance_phenotype"].fillna("").astype(str).str.strip().str.lower()
    d = d.assign(label=pheno.map(LABELS))[pheno.isin(LABELS)]
    tally("phenotype", d)
    d = d[d["used_ECOFF"].fillna("").astype(str).str.strip().str.lower() != "yes"]
    tally("ecoff", d)

    # conflicting records of an isolate and antibiotic: majority; no year, so a tie drops
    d = d.assign(antibiotic=d["antibiotic_name"].map(antibiotic))
    d = d[d["antibiotic"].notna()]
    cells = (d.groupby(["organism", "BioSample_ID", "antibiotic"])["label"]
             .agg(n_r="sum", n_records="size").reset_index())
    tied = cells["n_r"] * 2 == cells["n_records"]
    for o in counts:
        counts[o]["cells_tied"] = int((tied & (cells["organism"] == o)).sum())
    cells = cells[~tied].assign(
        label=lambda c: (c["n_r"] * 2 > c["n_records"]).astype(int),
        how=lambda c: ((c["n_r"] == 0) | (c["n_r"] == c["n_records"])).map(
            {True: "single", False: "majority"}))
    tally("conflict", cells)

    # independence from BV-BRC, over every record of the isolate (filtered or not)
    by_iso = full[full["organism"].notna()].groupby("BioSample_ID")
    from_bvbrc = by_iso["database"].agg(
        lambda s: s.fillna("").astype(str).str.split(";").map(
            lambda parts: BVBRC_SOURCE in [p.strip() for p in parts]).any())
    cells = cells[~cells["BioSample_ID"].map(from_bvbrc).fillna(False).astype(bool)]
    tally("bvbrc_source", cells)
    sra = by_iso["SRA_accession"].agg(accessions)
    asm = by_iso["assembly_ID"].agg(accessions)

    def known(b: str) -> bool:
        return (b.upper() in ids["biosample"] or bool(sra.get(b, set()) & ids["sra"])
                or bool({unversioned(a) for a in asm.get(b, set())} & ids["assembly"]))

    keep = {b: not known(b) for b in cells["BioSample_ID"].unique()}
    cells = cells[cells["BioSample_ID"].map(keep)]
    tally("bvbrc_genome", cells)

    assembly = by_iso["assembly_ID"].agg(_assembly_of)
    cells = cells.assign(assembly_id=cells["BioSample_ID"].map(assembly))
    cells = cells[cells["assembly_id"].notna()]
    tally("assembly", cells)
    for o in counts:
        counts[o]["cells"] = int((cells["organism"] == o).sum())

    isolates = (cells.drop_duplicates("BioSample_ID")[["organism", "BioSample_ID", "assembly_id"]]
                .rename(columns={"BioSample_ID": "biosample_id"}))
    isolates["sra_accession"] = isolates["biosample_id"].map(
        lambda b: ";".join(sorted(sra.get(b, set()))))
    isolates["sources"] = isolates["biosample_id"].map(
        by_iso["database"].agg(lambda s: ";".join(sorted(
            {p.strip() for v in s.dropna() for p in str(v).split(";") if p.strip()}))))
    phenotypes = cells.rename(columns={"BioSample_ID": "biosample_id"})[
        ["organism", "biosample_id", "antibiotic", "label", "n_records", "how"]]
    return (isolates.sort_values(["organism", "biosample_id"], ignore_index=True),
            phenotypes.sort_values(["organism", "biosample_id", "antibiotic"], ignore_index=True),
            counts)


def taxon_ok(tax_id: int | None, lineages: dict[int, set[int]], taxids) -> bool:
    """Whether an NCBI taxon is at or below one of the organism's taxa."""
    return tax_id is not None and bool(lineages.get(tax_id, {tax_id}) & {int(t) for t in taxids})


def pair_counts(phenotypes: pd.DataFrame) -> pd.DataFrame:
    """organism, antibiotic, n_resistant, n_susceptible of the eligible isolates."""
    g = phenotypes.groupby(["organism", "antibiotic"])["label"]
    t = g.agg(n_resistant="sum", n="size").reset_index()
    t["n_susceptible"] = t["n"] - t["n_resistant"]
    return t[["organism", "antibiotic", "n_resistant", "n_susceptible"]]
