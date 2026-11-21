#!/usr/bin/env python3
"""Step 16 — comparison with genotype-based prediction, per organism.

Subcommands, so that the tools can run in their own container:

  prep     the genomes of the organism's panel models (genomes.txt) and
           run_external.sh, which runs AMRFinderPlus and ResFinder on each genome
           (a finished genome is not run again), records the tool versions and
           the class and subclass vocabulary of AMRFinderPlus's database
  run      runs run_external.sh with the tools found on PATH
  collect  every genome's reports into amrfinder_calls.csv, amrfinder_genomes.csv
           and resfinder_calls.csv; a missing report stops it
  compare  sensitivity, specificity, balanced accuracy and error rates of
           AMRFinderPlus, ResFinder, RGI (two variants) and the model for every
           panel model of the organism (comparison.csv)
  all      prep, run, collect and compare

Inputs: the panel decisions, the panel inputs (00, 02c, 02d), 08's rgi_hits.csv,
the unitig store, the model matrices and 04's out-of-fold predictions. The rules
are in lib/external.py. Outputs: paths_organism.external_dir.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import external as ex  # noqa: E402
from lib import panel, registry  # noqa: E402
from lib.card_layer import near_universal_aros  # noqa: E402
from lib.config import load_config, resolve_path, resolve_tool  # noqa: E402
from lib.io_utils import run_logged  # noqa: E402
from lib.matrix_store import Store  # noqa: E402


def _included(organism, config) -> list[str]:
    d = panel.read_panel(resolve_path("panel_dir", config=config) / "panel_decisions.csv")
    return sorted(d.loc[(d["organism"] == organism) & (d["decision"] == panel.INCLUDED),
                        "antibiotic"])


def panel_genomes(organism, config) -> list[str]:
    """Genomes of every panel model of the organism."""
    ph, qc, cl = panel.read_inputs(organism, config)
    out: set[str] = set()
    for ab in _included(organism, config):
        out |= set(panel.pair_genomes(ph, qc, cl, ab)["Genome ID"])
    return sorted(out)


def command(out_dir: Path, genomes_dir: Path, *, amrfinder_organism: str, species: str,
            threads: int) -> str:
    """run_external.sh [K N]: AMRFinderPlus (--plus; core is selected later) and ResFinder
    with PointFinder on every N-th genome of genomes.txt from the K-th (all by default);
    shard 0 also records the versions and the class/subclass pairs of every table of
    AMRFinderPlus's database (amrfinder_catalog.tsv; amrfinder_update does not fetch the
    ReferenceGeneCatalog, which is read too when present). $AMRFINDER
    and $RESFINDER override the executables; AMRFinderPlus's database is $AMRFINDER_DB
    when set, ResFinder's databases come from its CGE_* variables. ResFinder's BLAST
    outputs (*_blast: about 20 files per genome, 2026-10-11) are removed once a genome is
    done; collect reads only its phenotype table."""
    return "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        'AMRFINDER="${AMRFINDER:-amrfinder}"', 'RESFINDER="${RESFINDER:-python -m resfinder}"',
        'DBARG=()', 'if [ -n "${AMRFINDER_DB:-}" ]; then DBARG=(--database "$AMRFINDER_DB"); fi',
        'K="${1:-0}"', 'N="${2:-1}"',
        f'cd "{out_dir.resolve()}"', "mkdir -p amrfinder resfinder",
        'if [ "$K" = 0 ]; then',
        '  "$AMRFINDER" --version > amrfinder_version.txt 2>&1',
        '  "$AMRFINDER" --database_version ${DBARG[@]+"${DBARG[@]}"} > amrfinder_database_version.txt 2>&1',
        "  DB=$(sed -n \"s/^Database directory: '\\(.*\\)'.*/\\1/p\" amrfinder_database_version.txt)",
        '  { printf "class\\tsubclass\\n"; for f in "$DB"/*.tsv "$DB"/*.txt; do',
        '      [ -e "$f" ] || continue',
        "      awk -F'\\t' 'NR == 1 { for (i = 1; i <= NF; i++) { h = tolower($i); "
        'sub(/^#/, "", h); if (h == "class") c = i; if (h == "subclass") s = i } next } '
        'c && s { print $c "\\t" $s }\' "$f"',
        '  done; } > amrfinder_catalog.tsv',
        '  [ "$(wc -l < amrfinder_catalog.tsv)" -gt 1 ] || '
        '{ echo "no class/subclass table in $DB" >&2; exit 1; }',
        "  $RESFINDER --version > resfinder_version.txt 2>&1",
        "fi",
        "awk -v k=\"$K\" -v n=\"$N\" '(NR - 1) % n == k' genomes.txt | while read -r g; do",
        '  if [ ! -e "amrfinder/$g.done" ]; then',
        f'    "$AMRFINDER" --nucleotide "{genomes_dir.resolve()}/$g.fna" --organism '
        f'{amrfinder_organism} --plus --threads {threads} --name "$g" ${{DBARG[@]+"${{DBARG[@]}}"}} '
        '--output "amrfinder/$g.tsv" > "amrfinder/$g.log" 2>&1',
        '    touch "amrfinder/$g.done"', "  fi",
        '  if [ ! -e "resfinder/$g.done" ]; then',
        '    rm -rf "resfinder/$g"',
        f'    $RESFINDER -ifa "{genomes_dir.resolve()}/$g.fna" -o "resfinder/$g" -s "{species}" '
        '--acquired --point --ignore_missing_species > "resfinder/$g.log" 2>&1',
        '    rm -rf "resfinder/$g/"*_blast',
        '    touch "resfinder/$g.done"', "  fi",
        "done", ""])


def prep(organism, config, out_dir: Path, threads: int) -> list[str]:
    genomes = panel_genomes(organism, config)
    org = registry.get_organism(organism)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "genomes.txt").write_text("\n".join(genomes) + "\n")
    script = out_dir / "run_external.sh"
    script.write_text(command(out_dir, resolve_path("raw_genomes_dir", organism=organism,
                                                    config=config),
                              amrfinder_organism=org["amrfinder_organism"],
                              species=org["resfinder_species"], threads=threads))
    script.chmod(0o755)
    return genomes


def run(out_dir: Path) -> None:
    tool = resolve_tool("amrfinder")
    if not tool:
        sys.exit("ERROR: amrfinder not found on PATH (or set AMR_AMRFINDER_BIN); run "
                 f"{out_dir / 'run_external.sh'} in the tools container instead.")
    os.environ["AMRFINDER"] = tool
    run_logged(["bash", out_dir / "run_external.sh"], out_dir / "run_external.log")


def _first_line(path: Path):
    return path.read_text().strip().splitlines()[0].strip() if path.exists() else None


def collect(organism, config, out_dir: Path) -> dict:
    genomes = (out_dir / "genomes.txt").read_text().split()
    missing = [g for g in genomes if not (out_dir / "amrfinder" / f"{g}.done").exists()
               or not (out_dir / "resfinder" / f"{g}.done").exists()]
    if missing:
        sys.exit(f"ERROR: tools not finished for {len(missing)} genome(s), e.g. {missing[:3]}")
    afp = pd.concat([ex.read_amrfinder(out_dir / "amrfinder" / f"{g}.tsv", g) for g in genomes],
                    ignore_index=True)
    afp.to_csv(out_dir / "amrfinder_calls.csv", index=False)
    pd.DataFrame({"genome_id": genomes}).to_csv(out_dir / "amrfinder_genomes.csv", index=False)
    abs_ = _included(organism, config)
    rows = []
    for g in genomes:
        table = ex.resfinder_table(out_dir / "resfinder" / g)
        if table is None:
            sys.exit(f"ERROR: ResFinder wrote no phenotype table for {g}")
        rows += [(g, a, r) for a, r in ex.read_resfinder(table, abs_).items()]
    pd.DataFrame(rows, columns=["genome_id", "antibiotic", "resistant"]).to_csv(
        out_dir / "resfinder_calls.csv", index=False)
    if not (out_dir / "amrfinder_catalog.tsv").exists():
        sys.exit("ERROR: amrfinder_catalog.tsv missing (run_external.sh writes it).")
    db = _first_line(out_dir / "amrfinder_database_version.txt")
    versions = {"amrfinderplus": _first_line(out_dir / "amrfinder_version.txt"),
                "amrfinderplus_database": next(
                    (ln.split(":", 1)[1].strip() for ln in
                     (out_dir / "amrfinder_database_version.txt").read_text().splitlines()
                     if ln.startswith("Database version")), db),
                "resfinder": _first_line(out_dir / "resfinder_version.txt"),
                "n_genomes": len(genomes)}
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    return versions


def compare(organism, config, out_dir: Path) -> pd.DataFrame:
    calls = pd.read_csv(out_dir / "amrfinder_calls.csv", dtype={"genome_id": str},
                        keep_default_na=False)
    rf = pd.read_csv(out_dir / "resfinder_calls.csv", dtype={"genome_id": str})
    catalog = ex.catalog_tokens(out_dir / "amrfinder_catalog.tsv")
    rgi = pd.read_csv(resolve_path("rgi_dir", organism=organism, config=config) / "rgi_hits.csv",
                      dtype={"genome_id": str, "aro": str}, keep_default_na=False,
                      usecols=["genome_id", "aro", "model_type", "drug_class"])
    store_genomes = Store(resolve_path("unitig_store_dir", organism=organism,
                                       config=config)).genomes
    near = near_universal_aros(rgi, store_genomes, config["card"]["near_universal"])
    keywords = registry.load_amrfinder_keywords()
    parts = []
    for ab in _included(organism, config):
        def path(key, ab=ab):
            return resolve_path(key, organism=organism, antibiotic=ab, config=config)

        mg = pd.read_csv(path("matrix_dir") / "genomes.csv", dtype={"Genome ID": str})
        truth = pd.Series(mg["label"].astype(int).to_numpy(), index=mg["Genome ID"])
        g = truth.index.tolist()
        targets = registry.card_drug_classes(ab)
        oof = path("cv_dir") / "oof_predictions.csv"
        preds = {
            "amrfinderplus": ex.predict_amrfinder(calls, g, keywords.get(ab, set()), catalog),
            "resfinder": ex.predict_resfinder(rf, g, ab),
            "rgi_all": ex.predict_rgi(rgi, g, targets, near, drop_near_universal=False),
            "rgi_without_near_universal": ex.predict_rgi(rgi, g, targets, near,
                                                         drop_near_universal=True),
            "model": (ex.predict_model(pd.read_csv(oof, dtype={"genome_id": str}), g,
                                       config["cv"]["threshold"]) if oof.exists() else None)}
        parts.append(ex.compare_model(f"{organism}__{ab}", truth, preds))
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    ok = table[table["assessable"] == 1]
    (out_dir / "external_summary.json").write_text(json.dumps({
        "organism": organism, "n_models": int(table["model_id"].nunique()),
        "assessable_models": {t: int(n) for t, n in ok.groupby("tool").size().items()},
        "median_balanced_accuracy": {t: round(float(v), 4) for t, v in
                                     ok.groupby("tool")["balanced_accuracy"].median().items()}},
        indent=2) + "\n")
    return table


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="Comparison with genotype-based prediction.")
    ap.add_argument("command", choices=["prep", "run", "collect", "compare", "all"])
    ap.add_argument("--organism", required=True)
    ap.add_argument("--threads", type=int, default=int(config["preprocessing"]["threads"]))
    args = ap.parse_args()
    out_dir = resolve_path("external_dir", organism=args.organism, config=config)
    print(f"EXTERNAL — {args.organism} — {args.command}")
    if args.command in ("prep", "all"):
        print(f"  {len(prep(args.organism, config, out_dir, args.threads))} genomes")
    if args.command in ("run", "all"):
        run(out_dir)
    if args.command in ("collect", "all"):
        print(f"  versions: {collect(args.organism, config, out_dir)}")
    if args.command in ("compare", "all"):
        t = compare(args.organism, config, out_dir)
        print(t[["model_id", "tool", "assessable", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
