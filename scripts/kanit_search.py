#!/usr/bin/env python3
"""KANIT search of one assembly: the knowledge base's biomarkers in the genome, a call for every
antibiotic of its organism, and a report (lib/genome_search.py, lib/genome_report.py).

    python scripts/kanit_search.py --genome isolate.fna --organism ecoli [--db kanit.sqlite]
        [--rule upto_strong_novel] [--out DIR] [--annotate kb|full] [--threads N]
        [--bakta-db DIR | --bakta-tsv FILE] [--plasmids mob|genomad] [--genomad-db DIR]
        [--mob-report FILE | --genomad-summary FILE]

--annotate kb (default) draws the neighbourhoods from the knowledge base alone: the markers,
their CARD gene and NCBI context. --annotate full adds the genes of the windows (Bakta) and the
class of each contig (MOB-suite or geNomad), from the tools on PATH or from their finished
output (--bakta-tsv, --mob-report, --genomad-summary); lib/genome_annotation.py.

Outputs in --out (default kanit_<genome name>): calls.csv, markers.csv (every marker, found or
not), hits.csv (the occurrences of the found markers), report.html, figures/*.svg and run.json.
Runs locally; not part of the workflow. A call is a research result, not a susceptibility test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import genome_annotation as ga  # noqa: E402
from lib import genome_report as gr  # noqa: E402
from lib import genome_search as gs  # noqa: E402

DEFAULT_DB = os.environ.get("KANIT_DB", str(PROJECT_ROOT / "results" / "kb" / "kanit.sqlite"))


def _stem(path: Path) -> str:
    s = path.name
    for ext in (".gz", ".fasta", ".fna", ".fa", ".fas"):
        s = s.removesuffix(ext)
    return s


def annotate(args, genome: Path, out: Path, contigs) -> tuple:
    """(genes or None, {contig: class} or None, the note on the neighbourhood figure)."""
    if args.annotate == "kb":
        return None, None, ("Knowledge base only: the markers with their CARD gene or NCBI context; the "
                            "genes between them are not annotated.")
    tools = out / "annotation"
    tools.mkdir(parents=True, exist_ok=True)
    if args.bakta_tsv:
        genes = ga.read_bakta_tsv(args.bakta_tsv)
    elif args.bakta_db:
        genes = ga.read_bakta_tsv(ga.run_bakta(genome, tools / "bakta", Path(args.bakta_db), args.threads))
    else:
        sys.exit("ERROR: --annotate full needs --bakta-db (to run Bakta) or --bakta-tsv.")
    if args.mob_report:
        cls = ga.read_mob_contig_report(args.mob_report)
    elif args.genomad_summary:
        cls = ga.read_genomad_plasmids(args.genomad_summary, contigs)
    elif args.plasmids == "mob":
        cls = ga.read_mob_contig_report(ga.run_mob_recon(genome, tools / "mob_recon", args.threads))
    elif args.plasmids == "genomad":
        if not args.genomad_db:
            sys.exit("ERROR: --plasmids genomad needs --genomad-db.")
        cls = ga.read_genomad_plasmids(ga.run_genomad(genome, tools / "genomad", Path(args.genomad_db),
                                                      args.threads), contigs)
    else:
        cls = None
    note = ("Genes from Bakta; resistance and mobile element genes are sorted by gene symbol and product. "
            + ("The contig class is a prediction. " if cls else ""))
    return genes, cls, note


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KANIT search of one assembly.")
    ap.add_argument("--genome", required=True, type=Path, help="assembly, FASTA (plain or gzip)")
    ap.add_argument("--organism", required=True, help="organism id of the knowledge base, e.g. ecoli")
    ap.add_argument("--db", default=DEFAULT_DB, help="kanit.sqlite (default: KANIT_DB or results/kb)")
    ap.add_argument("--rule", default=gs.DEFAULT_RULE, choices=gs.RULES)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--annotate", choices=["kb", "full"], default="kb")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bakta-db")
    ap.add_argument("--bakta-tsv")
    ap.add_argument("--plasmids", choices=["mob", "genomad", "none"], default="mob")
    ap.add_argument("--genomad-db")
    ap.add_argument("--mob-report")
    ap.add_argument("--genomad-summary")
    args = ap.parse_args(argv)
    if not Path(args.db).exists():
        sys.exit(f"ERROR: knowledge base not found: {args.db}")
    out = args.out or Path(f"kanit_{_stem(args.genome)}")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    contigs = gs.read_assembly(args.genome)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        res = gs.search(conn, contigs, args.organism, args.rule)
        org_name = (conn.execute("SELECT name FROM organism WHERE organism_id = ?",
                                 (args.organism,)).fetchone() or [args.organism])[0]
    genes, cls, note = annotate(args, args.genome, out, contigs)

    res.calls.to_csv(out / "calls.csv", index=False)
    res.markers.to_csv(out / "markers.csv", index=False)
    res.hits.to_csv(out / "hits.csv", index=False)
    win = gr.windows(res.hits, res.contigs)
    figs = {"signature": gr.signature(res.markers, res.calls),
            "map": gr.genome_map(res.contigs, res.hits),
            "neighbourhood": gr.neighbourhoods(res.hits, res.contigs, win, genes, cls)}
    svgs = {}
    for k, fig in figs.items():
        svgs[k] = gr.svg(fig)
        if svgs[k]:
            (out / "figures" / f"{k}.svg").write_text(svgs[k], encoding="utf-8")
    (out / "report.html").write_text(gr.report_html(res, args.genome.name, org_name, svgs, note),
                                     encoding="utf-8")
    (out / "run.json").write_text(json.dumps({
        "genome": str(args.genome), "genome_sha256": hashlib.sha256(args.genome.read_bytes()).hexdigest(),
        "organism": args.organism, "rule": args.rule, "annotate": args.annotate,
        "knowledge_base": {k: res.release.get(k) for k in ("kb_version", "card_version", "doi",
                                                           "code_commit")},
        "n_contigs": len(res.contigs), "n_markers_found": int(res.markers["present"].sum()),
        "calls": dict(zip(res.calls["antibiotic_id"], res.calls["call"], strict=True))}, indent=2) + "\n")
    n_r = int((res.calls["call"] == "R").sum())
    print(f"KANIT search — {args.genome.name} ({args.organism}): {int(res.markers['present'].sum())} "
          f"markers found; {n_r} of {len(res.calls)} antibiotics called resistant -> {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
