#!/usr/bin/env python3
"""The report of a workflow run (scripts/run_report.py)."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.unit


def test_run_report(tmp_path):
    spec = importlib.util.spec_from_file_location("amrtest_run_report",
                                                  PROJECT_ROOT / "scripts" / "run_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "run": {"name": "happy_turing", "entry": "main", "organisms": ["ecoli"]},
        "code": {"commit": "abc", "dirty": False}, "config": {},
        "completed": {"at": "2026-10-20T10:00:00Z"}}))
    (tmp_path / "completeness_summary.json").write_text(json.dumps({
        "complete": False, "percent_ok": 95.0, "applicable": 20,
        "failures": [{"unit": "ecoli__ampicillin", "step": "grading", "file": "f",
                      "status": "missing", "detail": ""}]}))
    pd.DataFrame({"unit": ["ecoli__ampicillin"], "cv": ["ok"], "grading": ["missing"]}).to_csv(
        tmp_path / "completeness_matrix.csv", index=False)
    (tmp_path / "resources_summary.json").write_text(json.dumps({
        "total": {"tasks": 3, "core_hours": 60.0, "cpu_hours": 40.0},
        "by_process": {"MODELS:CV_UNIT": {"tasks": 2, "core_hours": 60.0}},
        "jobs_without_accounting": 0}))
    page = m.report(tmp_path).read_text()
    for text in ("happy_turing", "background:#f5b7b1", "MODELS:CV_UNIT", "Missing or invalid",
                 "95.0% of 20"):
        assert text in page, text
