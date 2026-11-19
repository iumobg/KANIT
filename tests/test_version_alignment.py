#!/usr/bin/env python3
"""Guard: every declared release version must match config.yaml's project.version.

Four files carry the release version, and copies drift. .zenodo.json matters most:
it is the text Zenodo mints a permanent DOI over. The files cannot import anything,
so this test is their single source: bump the release and CI tells you what else
to bump. The knowledge-base schema has its own version in code
(lib.knowledge_base.SCHEMA_VERSION), which the run metadata reports.
"""

import json
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

RELEASE = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())["project"]["version"]


def _declared():
    root = PROJECT_ROOT
    return {
        ".zenodo.json": json.loads((root / ".zenodo.json").read_text())["version"],
        "CITATION.cff": yaml.safe_load((root / "CITATION.cff").read_text())["version"],
        "pyproject.toml": re.search(r'^version = "(.*)"',
                                    (root / "pyproject.toml").read_text(), re.M).group(1),
    }


def test_all_declared_versions_match_the_release():
    drift = {f: v for f, v in _declared().items() if v != RELEASE}
    assert not drift, (
        f"version drift vs config.yaml project.version={RELEASE}: {drift}. "
        ".zenodo.json in particular gets a permanent DOI — fix before depositing.")


def test_no_stale_version_in_public_prose():
    """.zenodo.json's description and notes are what a human reads off the DOI landing
    page; any version number stated there must be the current release."""
    z = json.loads((PROJECT_ROOT / ".zenodo.json").read_text())
    prose = f"{z.get('description', '')} {z.get('notes', '')}"
    stale = set(re.findall(r"\b(\d+\.\d+\.\d+)\b", prose)) - {RELEASE}
    assert not stale, f".zenodo.json prose mentions {sorted(stale)}; the release is {RELEASE}."


def test_collect_versions_reports_the_code_schema_version():
    from lib import run_metadata
    from lib.knowledge_base import SCHEMA_VERSION
    v = run_metadata.collect_versions(config={"provenance": {"kb_schema_version": "0.0.1"}})
    assert v["kb_schema_version"] == SCHEMA_VERSION   # config cannot override it
