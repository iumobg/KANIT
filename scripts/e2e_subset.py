#!/usr/bin/env python3
"""The data of the end-to-end test: a subset of a frozen BV-BRC snapshot.

Takes n genomes of one antibiotic from the organism's snapshot (half resistant, half
susceptible where possible, seeded), whose assemblies passed, and writes them as a
snapshot of their own in the locations of the overlay (e2e/...): the phenotypes of that
antibiotic only, the genomes, the download report, the records and a snapshot.json that
names its source. The assemblies are linked, not copied.

  python scripts/e2e_subset.py --organism enterobacter_cloacae --antibiotic ciprofloxacin \
      --overlay tests/data/e2e_overlay.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import bvbrc  # noqa: E402
from lib.config import CONFIG_FILE, deep_merge, load_config, resolve_path  # noqa: E402
from lib.matrix_store import sha256_file  # noqa: E402


def subset(src_cfg: dict, dst_cfg: dict, organism: str, antibiotic: str, n: int,
           seed: int = 0) -> list[str]:
    src = resolve_path("metadata_file", organism=organism, config=src_cfg).parent
    dst_file = resolve_path("metadata_file", organism=organism, config=dst_cfg)
    dst = dst_file.parent
    if not (src / "snapshot.json").exists():
        sys.exit(f"ERROR: no frozen snapshot in {src}; run the DOWNLOAD entry first.")
    long = pd.read_csv(src / "amr_cleaned_long.csv", dtype={"genome_id": str})
    report = pd.read_csv(src / "download_report.csv", dtype={"genome_id": str},
                         keep_default_na=False)
    passed = set(report.loc[report["status"] == "passed", "genome_id"])
    ab = long[(long["antibiotic"] == antibiotic) & long["genome_id"].isin(passed)]
    if ab.empty:
        sys.exit(f"ERROR: no genome of {organism} has a phenotype for {antibiotic}.")
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for label in (1, 0):
        ids = sorted(ab.loc[ab["label"] == label, "genome_id"])
        k = min(n // 2, len(ids))
        picked += list(rng.choice(ids, size=k, replace=False))
    picked = sorted(picked)
    dst.mkdir(parents=True, exist_ok=True)
    rows = ab[ab["genome_id"].isin(picked)]
    rows.to_csv(dst / "amr_cleaned_long.csv", index=False)
    bvbrc.pivot_binary(rows).to_csv(dst_file, index=False)
    for name in ("genomes.csv", "download_report.csv", "amr_records.csv"):
        t = pd.read_csv(src / name, dtype=str, keep_default_na=False)
        t[t["genome_id"].isin(picked)].to_csv(dst / name, index=False)
    (dst / "query.json").write_text((src / "query.json").read_text())
    genomes_src = resolve_path("raw_genomes_dir", organism=organism, config=src_cfg)
    genomes_dst = resolve_path("raw_genomes_dir", organism=organism, config=dst_cfg)
    genomes_dst.mkdir(parents=True, exist_ok=True)
    for g in picked:
        link = genomes_dst / f"{g}.fna"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink((genomes_src / f"{g}.fna").resolve(), link)
    files = {name: {"sha256": sha256_file(dst / name), "bytes": (dst / name).stat().st_size}
             for name in ("amr_cleaned_long.csv", dst_file.name, "genomes.csv",
                          "download_report.csv", "amr_records.csv", "query.json")}
    source = json.loads((src / "snapshot.json").read_text())
    (dst / "snapshot.json").write_text(json.dumps({
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": source["query"], "n_genomes": len(picked), "n_assemblies_failed": 0,
        "n_antibiotics": 1, "drugs_without_class": {}, "files": files,
        "e2e_subset": {"source": str(src / "snapshot.json"),
                       "source_sha256": sha256_file(src / "snapshot.json"),
                       "antibiotic": antibiotic, "n": n, "seed": seed}}, indent=2) + "\n")
    return picked


def main():
    ap = argparse.ArgumentParser(description="Data of the end-to-end test.")
    ap.add_argument("--organism", required=True)
    ap.add_argument("--antibiotic", required=True)
    ap.add_argument("--overlay", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=CONFIG_FILE, help="base config.yaml")
    args = ap.parse_args()
    src_cfg = load_config(args.config)
    dst_cfg = deep_merge(src_cfg, yaml.safe_load(args.overlay.read_text()) or {})
    picked = subset(src_cfg, dst_cfg, args.organism, args.antibiotic, args.n, args.seed)
    print(f"E2E SUBSET — {args.organism} / {args.antibiotic}: {len(picked)} genomes -> "
          f"{resolve_path('metadata_file', organism=args.organism, config=dst_cfg).parent}")


if __name__ == "__main__":
    main()
