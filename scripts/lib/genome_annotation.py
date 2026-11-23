"""The full annotation of a KANIT search report (optional): the genes of the assembly (Bakta) and
the class of each contig, plasmid or chromosome (MOB-suite's mob_recon or geNomad).

Each tool runs when it is on PATH, or its finished output is read. The genes are sorted into
resistance genes, mobile element and plasmid genes, other and hypothetical proteins from their
gene symbol and product (``category``); the sort serves the figure and is not an annotation of
its own. Contig classes from short-read assemblies are predictions: MOB-suite's sensitivity and
specificity for plasmid contigs were about 95% and 88% (Robertson & Nash 2018), and large
plasmids with repeats are often split.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

GENE_COLUMNS = ["contig", "start", "end", "strand", "gene", "product", "type", "category", "label"]

_AMR_GENE = re.compile(r"^(bla|aac|aad|aph|ant\(|armA|rmt|tet|sul\d|dfr|qnr|qep|oqx|erm|mph|msr|mef|lnu|"
                       r"vat|vga|cfr|optrA|poxtA|van|mcr|cat|cml|floR|fos|arr|ble|sat|str[AB]|mec)",
                       re.IGNORECASE)
_AMR_PRODUCT = re.compile(r"beta-lactamase|carbapenem|resistan|aminoglycoside|chloramphenicol|"
                          r"florfenicol|tetracycline|quinolone|macrolide|lincosamide|streptogramin|"
                          r"vancomycin|glycopeptide|fosfomycin|colistin|polymyxin|trimethoprim|"
                          r"sulfonamide|rifamp|bleomycin|methicillin|penicillin-binding protein 2a",
                          re.IGNORECASE)
_MGE = re.compile(r"transposase|integrase|recombinase|resolvase|invertase|insertion (sequence|element)|"
                  r"\bIS\d|mobile element|relaxase|mobilization|conjugal|conjugative|type IV secretion|"
                  r"\btra[A-Z]\b|plasmid|phage",
                  re.IGNORECASE)


def category(gene: str, product: str) -> str:
    gene, product = (gene or "").strip(), (product or "").strip()
    if (gene and _AMR_GENE.match(gene)) or _AMR_PRODUCT.search(product):
        return "amr"
    if _MGE.search(product) or _MGE.search(gene):
        return "mge"
    if not product or "hypothetical" in product.lower():
        return "hypothetical"
    return "other"


def _run(cmd: Sequence[object], log: Path) -> None:
    if not shutil.which(str(cmd[0])):
        raise FileNotFoundError(f"{cmd[0]} not found on PATH")
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, check=False)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed (exit {r.returncode}); see {log}")


# ---- Bakta ----------------------------------------------------------------------------------------
def run_bakta(genome: Path, out: Path, db: Path, threads: int = 4) -> Path:
    """Bakta with the contig headers kept (so contigs keep their ids); returns the TSV."""
    out.mkdir(parents=True, exist_ok=True)
    _run(["bakta", "--db", db, "--output", out, "--prefix", "kanit", "--threads", threads,
          "--keep-contig-headers", "--skip-plot", "--force", genome], out / "bakta.log")
    return out / "kanit.tsv"


def read_bakta_tsv(path) -> pd.DataFrame:
    """Bakta's TSV (``#Sequence Id  Type  Start  Stop  Strand  Locus Tag  Gene  Product  DbXrefs``,
    1-based closed coordinates) as 0-based half-open genes with a category."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            contig, typ, start, stop, strand, _, gene, product = p[:8]
            rows.append((contig, int(start) - 1, int(stop), strand if strand in "+-" else "+", gene,
                         product, typ, category(gene, product), gene or product))
    return pd.DataFrame(rows, columns=GENE_COLUMNS)


# ---- contig class ----------------------------------------------------------------------------------
def run_mob_recon(genome: Path, out: Path, threads: int = 4) -> Path:
    _run(["mob_recon", "--infile", genome, "--outdir", out, "--force", "--num_threads", threads],
         out.parent / "mob_recon.log")
    return out / "contig_report.txt"


def read_mob_contig_report(path) -> dict[str, str]:
    """{contig: "plasmid (cluster)" or "chromosome"} from mob_recon's contig_report.txt."""
    d = pd.read_csv(path, sep="\t", dtype=str)
    for col in ("contig_id", "molecule_type"):
        if col not in d.columns:
            raise ValueError(f"{path}: no column {col}")
    out = {}
    for r in d.itertuples(index=False):
        kind = str(r.molecule_type).strip().lower()
        cl = getattr(r, "primary_cluster_id", "")
        out[str(r.contig_id).split()[0]] = (f"plasmid ({cl}, MOB-suite)" if kind == "plasmid" and
                                            isinstance(cl, str) and cl and cl != "-"
                                            else f"{kind} (MOB-suite)")
    return out


def run_genomad(genome: Path, out: Path, db: Path, threads: int = 4) -> Path:
    _run(["genomad", "end-to-end", "--cleanup", "--threads", threads, genome, out, db],
         out.parent / "genomad.log")
    stem = Path(genome).name
    for ext in (".gz", ".fasta", ".fna", ".fa"):
        stem = stem.removesuffix(ext)
    return out / f"{stem}_summary" / f"{stem}_plasmid_summary.tsv"


def read_genomad_plasmids(path, contigs) -> dict[str, str]:
    """{contig: "plasmid (geNomad score s)" or "not classified as plasmid (geNomad)"}."""
    d = pd.read_csv(path, sep="\t", dtype={"seq_name": str})
    if "plasmid_score" not in d.columns:
        d = d.assign(plasmid_score=float("nan"))
    score = dict(zip(d["seq_name"].astype(str), d["plasmid_score"], strict=True))
    return {c: (f"plasmid (geNomad score {float(score[c]):.2f})" if c in score
                else "not classified as plasmid (geNomad)") for c in contigs}
