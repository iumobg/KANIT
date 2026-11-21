#!/usr/bin/env python3
"""Step 00a — the data snapshot of one organism from BV-BRC.

Subcommands (internet; the DOWNLOAD entry runs them on a login node):

  query   the public genomes at or below the organism's taxa (registry taxids) with
          their NCBI identifiers, their susceptibility records with laboratory
          evidence, and the binary phenotypes (rules in lib/bvbrc.py)
  fetch   the assembly of every phenotyped genome, checked against the genome's
          number of sequences and length; passed genomes are skipped, so it resumes
  freeze  the phenotype table of the genomes whose assembly passed and snapshot.json:
          query date, API version, filters, counts and the checksum of every
          snapshot file (the assemblies' checksums are in download_report.csv); it
          warns of single drugs without a registry class whose smaller class already
          reaches the panel threshold before quality control
  verify  every checksum of the frozen snapshot, the assemblies included
  all     query (unless done), fetch and freeze; with a frozen snapshot, verify only

A frozen snapshot is not queried again: a new query needs its files removed.

Outputs, in the directory of paths_organism.metadata_file: genomes.csv (the
phenotyped genomes and their NCBI identifiers), amr_records.csv (the records as
BV-BRC gave them), amr_cleaned_long.csv, query.json, download_report.csv,
amr_phenotypes.csv (Genome ID and one 0/1 column per antibiotic) and
snapshot.json; the assemblies in paths_organism.raw_genomes_dir.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import bvbrc, registry  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402
from lib.matrix_store import sha256_file  # noqa: E402

REPORT_COLUMNS = ["genome_id", "status", "sequences", "length", "bytes", "sha256", "problem"]
SNAPSHOT_FILES = ["genomes.csv", "amr_records.csv", "amr_cleaned_long.csv", "query.json",
                  "download_report.csv", "amr_phenotypes.csv"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_json(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def query(organism: str, meta: Path, api: bvbrc.Api) -> dict:
    if (meta / "snapshot.json").exists():
        sys.exit(f"ERROR: {meta / 'snapshot.json'} exists: the snapshot is frozen.")
    taxids = registry.organism_taxids(organism)
    queried_at, version = _now(), api.version()
    print(f"  BV-BRC API {version}; taxa {taxids}")
    g = bvbrc.genomes(api, taxids)
    print(f"  {len(g)} public genomes at or below the taxa")
    rec = bvbrc.records(api, g["genome_id"].tolist(),
                        progress=lambda i, n: print(f"  records: {i}/{n} genomes", flush=True))
    cleaned, rep = bvbrc.clean_amr_table(rec)
    phenotyped = g[g["genome_id"].isin(set(cleaned["genome_id"]))]
    meta.mkdir(parents=True, exist_ok=True)
    _write_csv(phenotyped, meta / "genomes.csv")
    _write_csv(rec, meta / "amr_records.csv")
    _write_csv(cleaned, meta / "amr_cleaned_long.csv")
    q = {"organism": organism, "taxids": taxids, "source": bvbrc.API, "api_version": version,
         "queried_at": queried_at,
         "genome_query": bvbrc.genome_filter(taxids),
         "record_query": bvbrc.record_filter(["<genome ids>"]),
         "n_genomes_at_or_below": len(g),
         "n_genomes_with_records": int(rec["genome_id"].nunique()),
         "n_genomes_phenotyped": len(phenotyped), "phenotypes": rep}
    _write_json(q, meta / "query.json")
    return q


def _report(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).reindex(columns=REPORT_COLUMNS)
    for c in ("sequences", "length", "bytes"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def _read_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REPORT_COLUMNS)
    return pd.read_csv(path, dtype={"genome_id": str}, keep_default_na=False)


def fetch_one(api: bvbrc.Api, row, genomes_dir: Path) -> dict:
    gid = row["genome_id"]
    try:
        data = api.fasta(gid)
    except Exception as e:                          # recorded; the next fetch tries again
        return {"genome_id": gid, "status": "failed", "problem": f"{type(e).__name__}: {e}"}
    n, length, problem = bvbrc.check_fasta(data, row["contigs"], row["genome_length"])
    out = {"genome_id": gid, "sequences": n, "length": length, "problem": problem}
    if problem:
        return {**out, "status": "failed"}
    path = genomes_dir / f"{gid}.fna"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return {**out, "status": "passed", "bytes": len(data), "sha256": bvbrc.sha256_bytes(data)}


def fetch(meta: Path, genomes_dir: Path, api: bvbrc.Api, *, workers: int) -> pd.DataFrame:
    """Assemblies of the phenotyped genomes; download_report.csv grows as they finish."""
    if not (meta / "query.json").exists():
        sys.exit("ERROR: run the query first.")
    if (meta / "snapshot.json").exists():
        sys.exit(f"ERROR: {meta / 'snapshot.json'} exists: the snapshot is frozen.")
    g = pd.read_csv(meta / "genomes.csv", dtype=str, keep_default_na=False)
    report_file = meta / "download_report.csv"
    done = _read_report(report_file)
    done = done[(done["status"] == "passed") & done["genome_id"].isin(set(g["genome_id"]))]
    done = done.drop_duplicates("genome_id", keep="last")
    on_disk = np.array([(genomes_dir / f"{gid}.fna").exists()
                        and (genomes_dir / f"{gid}.fna").stat().st_size == int(b)
                        for gid, b in zip(done["genome_id"], done["bytes"], strict=True)], dtype=bool)
    done = done.loc[on_disk]
    todo = g[~g["genome_id"].isin(set(done["genome_id"]))]
    print(f"  assemblies: {len(done)} passed before, {len(todo)} to fetch")
    genomes_dir.mkdir(parents=True, exist_ok=True)
    rows = done.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, api, r, genomes_dir) for r in todo.to_dict("records")]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(futs):
                _write_csv(_report(rows), report_file)
                print(f"  fetched {i}/{len(futs)}", flush=True)
    report = (_report(rows).drop_duplicates("genome_id", keep="last")
              .sort_values("genome_id", ignore_index=True))
    report = report[report["genome_id"].isin(set(g["genome_id"]))]
    _write_csv(report, report_file)
    print(f"  {int((report['status'] == 'passed').sum())} passed, "
          f"{int((report['status'] == 'failed').sum())} failed")
    return report


def freeze(meta: Path, phenotypes_file: Path, *, min_minority: int) -> dict:
    g = pd.read_csv(meta / "genomes.csv", dtype=str, keep_default_na=False)
    report = _read_report(meta / "download_report.csv")
    unfetched = sorted(set(g["genome_id"]) - set(report["genome_id"]))
    if unfetched:
        sys.exit(f"ERROR: {len(unfetched)} genome(s) not fetched yet, e.g. {unfetched[:3]}")
    passed = set(report.loc[report["status"] == "passed", "genome_id"]) & set(g["genome_id"])
    cleaned = pd.read_csv(meta / "amr_cleaned_long.csv", dtype={"genome_id": str})
    wide = bvbrc.pivot_binary(cleaned[cleaned["genome_id"].isin(passed)])
    _write_csv(wide, phenotypes_file)
    files = {}
    for name in SNAPSHOT_FILES:
        p = phenotypes_file if name == "amr_phenotypes.csv" else meta / name
        files[_rel(p)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    no_class = bvbrc.drugs_without_class(wide, min_minority)
    if no_class:
        print(f"  WARNING: drugs without a registry class may pass the panel: {no_class}; "
              "register their class (with its CARD drug-class terms) before the main run")
    snap = {"frozen_at": _now(), "query": json.loads((meta / "query.json").read_text()),
            "n_genomes": len(wide), "n_assemblies_failed": len(g) - len(passed),
            "n_antibiotics": len(wide.columns) - 1, "drugs_without_class": no_class,
            "files": files}
    _write_json(snap, meta / "snapshot.json")
    return snap


def verify(meta: Path, genomes_dir: Path) -> list[str]:
    snap = json.loads((meta / "snapshot.json").read_text())
    bad = [p for p, e in snap["files"].items()
           if not (PROJECT_ROOT / p).exists() or sha256_file(PROJECT_ROOT / p) != e["sha256"]]
    report = _read_report(meta / "download_report.csv")
    for gid, sha in report.loc[report["status"] == "passed", ["genome_id", "sha256"]].itertuples(
            index=False):
        f = genomes_dir / f"{gid}.fna"
        if not f.exists() or sha256_file(f) != sha:
            bad.append(_rel(f))
    return bad


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="BV-BRC data snapshot of one organism.")
    ap.add_argument("command", choices=["query", "fetch", "freeze", "verify", "all"])
    ap.add_argument("--organism", required=True)
    ap.add_argument("--workers", type=int, default=8, help="assemblies fetched at once")
    args = ap.parse_args()
    phenotypes_file = resolve_path("metadata_file", organism=args.organism, config=config)
    meta = phenotypes_file.parent
    genomes_dir = resolve_path("raw_genomes_dir", organism=args.organism, config=config)
    api = bvbrc.Api()
    print(f"BV-BRC SNAPSHOT — {args.organism} — {args.command}")
    frozen = (meta / "snapshot.json").exists()
    if args.command == "verify" or (args.command == "all" and frozen):
        bad = verify(meta, genomes_dir)
        if bad:
            sys.exit(f"ERROR: {len(bad)} file(s) differ from the snapshot, e.g. {bad[:3]}")
        print("  snapshot verified")
        return
    if args.command == "query" or (args.command == "all" and not (meta / "query.json").exists()):
        q = query(args.organism, meta, api)
        print(f"  {q['n_genomes_phenotyped']} phenotyped genomes; {q['phenotypes']}")
    if args.command in ("fetch", "all"):
        report = fetch(meta, genomes_dir, api, workers=args.workers)
        if args.command == "all" and (report["status"] == "failed").any():
            time.sleep(60)                          # a second pass for transient failures
            fetch(meta, genomes_dir, api, workers=args.workers)
    if args.command in ("freeze", "all"):
        s = freeze(meta, phenotypes_file, min_minority=int(config["panel"]["min_minority"]))
        print(f"  frozen: {s['n_genomes']} genomes, {s['n_antibiotics']} antibiotics, "
              f"{s['n_assemblies_failed']} assemblies failed")


if __name__ == "__main__":
    main()
