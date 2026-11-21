#!/usr/bin/env python3
"""The checksums of the outputs at the end of a workflow run.

Every file under the output roots, with its size, modification time and SHA-256, and
the config location it belongs to: the paths_organism key of the most specific
template that holds it, with the organism and antibiotic in the path. So the outputs
of every step and model can be read from one table, and a backup can be verified
against it.

The roots are results/, data/processed/, data/external/ and models/. The assemblies
under data/raw/ are left out: their checksums are in the data snapshot's
download_report.csv. A checksum in the latest earlier manifest is reused while the
file's size and modification time are unchanged.

  run_outputs.py --out runs/nextflow/<run>/outputs.csv

Writes outputs.csv and outputs_summary.json next to it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.config import load_config  # noqa: E402
from lib.matrix_store import sha256_file  # noqa: E402

ROOTS = ["results", "data/processed", "data/external", "models"]
COLUMNS = ["path", "bytes", "mtime_ns", "modified", "sha256", "location", "organism",
           "antibiotic"]


def locations(config: dict) -> list[tuple[str, re.Pattern]]:
    """(key, pattern) of every paths_organism template, the most specific first."""
    out = []
    for key, template in (config.get("paths_organism") or {}).items():
        absolute = str(PROJECT_ROOT / template).rstrip("/")
        rx = re.escape(absolute)
        for name in ("organism", "antibiotic"):
            rx = rx.replace(re.escape("{" + name + "}"), f"(?P<{name}>[^/]+)")
        rx = rx.replace(re.escape("{run_id}"), "[^/]+")
        tail = "$" if key.endswith("_file") else "(?:/|$)"
        out.append((len(absolute), key, re.compile("^" + rx + tail)))
    return [(k, p) for _, k, p in sorted(out, key=lambda t: -t[0])]


def locate(path: Path, locs: list[tuple[str, re.Pattern]]) -> tuple[str, str, str]:
    for key, pattern in locs:
        m = pattern.match(str(path))
        if m:
            g = m.groupdict()
            return key, g.get("organism") or "", g.get("antibiotic") or ""
    return "other", "", ""


def previous_checksums(out: Path) -> dict[str, tuple[int, int, str]]:
    """path -> (bytes, mtime_ns, sha256) of the latest earlier manifest beside this run."""
    runs = sorted(p for p in out.parent.parent.glob("*/outputs.csv") if p != out)
    if not runs:
        return {}
    prev = pd.read_csv(runs[-1], dtype={"path": str, "sha256": str})
    return {p: (int(b), int(t), s) for p, b, t, s in
            zip(prev["path"], prev["bytes"], prev["mtime_ns"], prev["sha256"], strict=True)}


def manifest(config: dict, out: Path, *, roots=ROOTS, threads: int = 8) -> pd.DataFrame:
    files = sorted(f for r in roots if (PROJECT_ROOT / r).is_dir()
                   for f in (PROJECT_ROOT / r).rglob("*") if f.is_file())
    cache = previous_checksums(out)
    locs = locations(config)

    def row(f: Path) -> dict:
        st = f.stat()
        rel = str(f.relative_to(PROJECT_ROOT)) if f.is_relative_to(PROJECT_ROOT) else str(f)
        hit = cache.get(rel)
        sha = hit[2] if hit and hit[:2] == (st.st_size, st.st_mtime_ns) else sha256_file(f)
        key, org, ab = locate(f, locs)
        return {"path": rel, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
                    timespec="seconds"),
                "sha256": sha, "location": key, "organism": org, "antibiotic": ab}

    with ThreadPoolExecutor(max_workers=threads) as ex:
        rows = list(ex.map(row, files))
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser(description="Checksums of the outputs at the end of a run.")
    ap.add_argument("--out", type=Path, required=True, help="outputs.csv in the run directory")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    config = load_config()
    prefix = config.get("paths_prefix")                 # an overlay's separate tree (e2e/)
    roots = [f"{prefix.rstrip('/')}/{r}" for r in ROOTS] if prefix else ROOTS
    table = manifest(config, out, roots=roots, threads=args.threads)
    tmp = out.with_name(out.name + ".tmp")
    table.to_csv(tmp, index=False)
    os.replace(tmp, out)
    summary = {"n_files": len(table), "bytes": int(table["bytes"].sum()),
               "sha256": sha256_file(out), "roots": roots,
               "by_location": {k: int(n) for k, n in table["location"].value_counts().items()}}
    (out.parent / "outputs_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"RUN OUTPUTS — {summary['n_files']} files, {summary['bytes'] / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
