#!/usr/bin/env python3
"""Verified backup of the results tree at the end of every stage.

The tree is cut into units: every directory two levels under a root (for example
results/ecoli/ampicillin, data/processed/ecoli/unitig_store, data/raw/ecoli/genomes)
is one unit with everything below it, and the files lying directly in a root or in a
directory one level under it form one more unit there. Every finished run directory
of runs/nextflow is a unit too (the current run is backed up by the next stage). A
unit's fingerprint is the SHA-256 of its files' paths, sizes and modification times.

  pack    every unit whose fingerprint differs from its last verified backup in the
          ledger becomes one tar.gz in the staging directory (pigz when present); the
          archive must hold exactly the unit's files; backup_manifest.tsv lists unit,
          files, bytes, fingerprint, archive, archive size and MD5
  upload  is scripts/backup_upload.sh on a node with internet: one archive per rclone
          call, then MD5, name and count checks against the remote; only then are the
          rows added to the ledger and the staging directory emptied

The intermediate of unitig-caller (unitig_store/call/, its Rtab of tens of GB) is left
out: the store built from it is backed up and the call can be repeated. Symbolic links are
archived as links (the genome QC links every assembly into one folder; the assemblies are
backed up where they lie), and a link counts with its own size and time.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.config import load_config  # noqa: E402

ROOTS = ["results", "data/processed", "data/external", "data/raw", "models"]
RUNS = "runs/nextflow"
EXCLUDE = ["*/unitig_store/call/*"]
MANIFEST_COLUMNS = ["unit", "mode", "n_files", "bytes", "fingerprint", "archive",
                    "archive_bytes", "md5"]
LEDGER_COLUMNS = MANIFEST_COLUMNS + ["remote", "run", "verified_at"]


@dataclass
class Unit:
    path: str            # relative to the project root
    mode: str            # "tree": everything below; "files": only the files directly in it
    files: list[Path]
    project: Path = PROJECT_ROOT

    @property
    def archive(self) -> str:
        return self.path.replace("/", "__") + ("__files" if self.mode == "files" else "") + ".tar.gz"

    def names(self) -> list[str]:
        return [str(f.relative_to(self.project)) for f in self.files]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for f, name in zip(self.files, self.names(), strict=True):
            st = f.lstat()
            h.update(f"{name}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
        return h.hexdigest()

    def bytes(self) -> int:
        return sum(f.lstat().st_size for f in self.files)


def _excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in EXCLUDE)


def _tree_files(d: Path, project: Path) -> list[Path]:
    return sorted(f for f in d.rglob("*")
                  if f.is_file() and not _excluded(str(f.relative_to(project))))


def _direct_files(d: Path) -> list[Path]:
    return sorted(f for f in d.iterdir() if f.is_file())


def units(project: Path = PROJECT_ROOT, *, current_run: str | None = None,
          roots=ROOTS) -> list[Unit]:
    out: list[Unit] = []

    def rel(p: Path) -> str:
        return str(p.relative_to(project))

    for root in roots:
        r = project / root
        if not r.is_dir():
            continue
        if _direct_files(r):
            out.append(Unit(root, "files", _direct_files(r), project))
        for c in sorted(p for p in r.iterdir() if p.is_dir()):
            if _direct_files(c):
                out.append(Unit(rel(c), "files", _direct_files(c), project))
            for d in sorted(p for p in c.iterdir() if p.is_dir()):
                files = _tree_files(d, project)
                if files:
                    out.append(Unit(rel(d), "tree", files, project))
    runs = project / RUNS
    if runs.is_dir():
        for d in sorted(p for p in runs.iterdir() if p.is_dir() and p.name != current_run):
            files = _tree_files(d, project)
            if files:
                out.append(Unit(rel(d), "tree", files, project))
    return out


def read_ledger(ledger: Path) -> dict[tuple[str, str], dict]:
    """(unit, mode) -> the latest verified row."""
    if not ledger.exists():
        return {}
    latest: dict[tuple[str, str], dict] = {}
    with open(ledger, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            k = (row["unit"], row["mode"])
            if k not in latest or row["verified_at"] >= latest[k]["verified_at"]:
                latest[k] = row
    return latest


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_archive(unit: Unit, out: Path, *, threads: int) -> None:
    """tar.gz of the unit's files, named relative to the project root."""
    tmp = out.with_name(out.name + ".tmp")
    names = unit.names()
    compressor = (["pigz", "-1", "-p", str(threads)] if shutil.which("pigz") else ["gzip", "-1"])
    with open(tmp, "wb") as fh:
        tar = subprocess.Popen(["tar", "-C", str(unit.project), "-cf", "-", "--no-recursion",
                                "-T", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               env={**os.environ, "COPYFILE_DISABLE": "1"})  # no macOS ._ files
        comp = subprocess.Popen(compressor, stdin=tar.stdout, stdout=fh)
        assert tar.stdin is not None and tar.stdout is not None
        tar.stdout.close()
        tar.stdin.write(("\n".join(names) + "\n").encode())
        tar.stdin.close()
        rc_comp, rc_tar = comp.wait(), tar.wait()
    if rc_tar or rc_comp:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{unit.path}: tar exited {rc_tar}, compression {rc_comp}")
    with tarfile.open(tmp, "r:gz") as t:
        members = sorted(m.name for m in t.getmembers() if m.isfile() or m.issym())
    if members != sorted(names):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{unit.path}: the archive holds {len(members)} files, "
                           f"the unit {len(names)}")
    os.replace(tmp, out)


def pack(stage: Path, ledger: Path, *, current_run: str | None, threads: int,
         project: Path = PROJECT_ROOT, roots=ROOTS) -> list[dict]:
    stage.mkdir(parents=True, exist_ok=True)
    done = read_ledger(ledger)
    rows = []
    for u in units(project, current_run=current_run, roots=roots):
        fp = u.fingerprint()
        last = done.get((u.path, u.mode))
        if last and last["fingerprint"] == fp:
            continue
        out = stage / u.archive
        print(f"  pack {u.path} ({u.mode}): {len(u.files)} files", flush=True)
        write_archive(u, out, threads=threads)
        rows.append({"unit": u.path, "mode": u.mode, "n_files": len(u.files), "bytes": u.bytes(),
                     "fingerprint": fp, "archive": u.archive,
                     "archive_bytes": out.stat().st_size, "md5": _md5(out)})
    with open(stage / "backup_manifest.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Verified backup: pack the changed units.")
    ap.add_argument("command", choices=["pack"])
    ap.add_argument("--stage", type=Path, required=True, help="staging directory")
    ap.add_argument("--ledger", type=Path, required=True, help="ledger of verified backups")
    ap.add_argument("--current-run", default=None,
                    help="run directory name left out (it is still being written)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="also copy backup_manifest.tsv here")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    prefix = load_config().get("paths_prefix")          # an overlay's separate tree (e2e/)
    roots = [f"{prefix.rstrip('/')}/{r}" for r in ROOTS] if prefix else ROOTS
    rows = pack(args.stage, args.ledger, current_run=args.current_run, threads=args.threads,
                roots=roots)
    if args.run_dir:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.stage / "backup_manifest.tsv", args.run_dir / "backup_manifest.tsv")
    total = sum(r["archive_bytes"] for r in rows)
    print(f"BACKUP PACK — {len(rows)} unit(s) changed, {total / 1e9:.2f} GB -> {args.stage}")


if __name__ == "__main__":
    main()
