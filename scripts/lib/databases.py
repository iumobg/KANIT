"""Manifest of the reference databases.

CARD (RGI), AMRFinderPlus, ResFinder, PointFinder and CheckM2 are downloaded on
the same day. Each download is recorded once, when it is made: the database's
own version, the download date, the source and the checksum of every file. The
version is read from the database itself where it carries one (CARD's
card.json ``_version``, AMRFinderPlus's version.txt, the VERSION file or git
commit of ResFinder's and PointFinder's databases, CheckM2's .dmnd file name), so
that it cannot drift from what was downloaded. ``verify`` finds a database that
changed after it was recorded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

NAMES = ("card", "amrfinderplus", "resfinder", "pointfinder", "checkm2")


def card_version(card_json) -> str:
    """CARD's own version, from card.json."""
    with open(card_json, encoding="utf-8") as f:
        version = json.load(f).get("_version")
    if not version:
        raise ValueError(f"{card_json}: no _version field")
    return str(version)


def detect_version(name: str, path) -> str | None:
    """The version a database carries itself (None when it carries none)."""
    p = Path(path)
    if name == "card":
        return card_version(p / "card.json" if p.is_dir() else p)
    if name == "amrfinderplus" and (p / "version.txt").exists():
        return (p / "version.txt").read_text().strip()
    if name in ("resfinder", "pointfinder"):
        if (p / "VERSION").exists():
            return (p / "VERSION").read_text().strip()
        try:
            out = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True)
            return out.stdout.strip() or None
        except (OSError, subprocess.CalledProcessError):
            return None
    if name == "checkm2":
        dmnd = sorted(p.glob("*.dmnd")) if p.is_dir() else [p]
        return dmnd[0].name if dmnd else None
    return None


def checksums(path) -> list[dict]:
    """sha256 and size of every file of a database (``.git`` left out), sorted by path."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*")
                                           if f.is_file() and ".git" not in f.parts)
    out = []
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        out.append({"file": str(f.relative_to(p)) if p.is_dir() else f.name,
                    "sha256": h.hexdigest(), "bytes": f.stat().st_size})
    return out


def load(manifest) -> dict:
    m = Path(manifest)
    return json.loads(m.read_text()) if m.exists() else {}


def record(manifest, name: str, path, *, source: str | None = None, version: str | None = None,
           downloaded_on: str | None = None) -> dict:
    """Add or replace one database in the manifest; the version is detected unless given."""
    if name not in NAMES:
        raise ValueError(f"unknown database {name!r}; one of {NAMES}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    entry = {"path": str(p), "version": version or detect_version(name, p),
             "downloaded_on": downloaded_on or datetime.now(timezone.utc).date().isoformat(),
             "source": source, "files": checksums(p)}
    if not entry["version"]:
        raise ValueError(f"{name}: no version found in {p}; give it with --version")
    doc = load(manifest)
    doc[name] = entry
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return entry


def verify(manifest) -> list[str]:
    """Databases whose files changed, disappeared or appeared after they were recorded."""
    problems = []
    for name, e in sorted(load(manifest).items()):
        p = Path(e["path"])
        if not p.exists():
            problems.append(f"{name}: {p} is missing")
            continue
        now = {f["file"]: f["sha256"] for f in checksums(p)}
        then = {f["file"]: f["sha256"] for f in e["files"]}
        changed = sorted(f for f in then if now.get(f) != then[f]) + sorted(set(now) - set(then))
        if changed:
            problems.append(f"{name}: {len(changed)} file(s) differ, e.g. {changed[:3]}")
    return problems


def same_day(manifest) -> bool:
    """Whether every recorded database was downloaded on one day."""
    days = {e["downloaded_on"] for e in load(manifest).values()}
    return len(days) <= 1
