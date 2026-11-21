#!/usr/bin/env python3
"""The report of a workflow run: one HTML page in the run directory.

It joins what the run directory already holds: the run manifest (run, commit,
configuration checksums), the completeness matrix and its failures, the resources per
process and in total (core and CPU hours, the largest memory and the longest task),
the checksums of the outputs and the verified backup.

  run_report.py --run-dir runs/nextflow/<run>      -> run_report.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

COLOURS = {"ok": "#d5f5e3", "n/a": "#eeeeee", "missing": "#f5b7b1", "invalid": "#fad7a0"}
CSS = """body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em;
color:#222}table{border-collapse:collapse;margin:.5em 0 1.5em;font-size:12px}th,td{border:1px
solid #ddd;padding:3px 7px}th{background:#f4f4f4}.meta{color:#666}"""


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _table(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p class='meta'>not available</p>"
    return df.to_html(index=False, border=0, na_rep="", float_format=lambda x: f"{x:.4g}")


def _matrix(path: Path) -> str:
    if not path.exists():
        return "<p class='meta'>not available</p>"
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in m.columns)
    rows = []
    for r in m.itertuples(index=False):
        cells = [f"<td>{html.escape(r[0])}</td>"] + [
            f"<td style='background:{COLOURS.get(v, '#fff')}'>{html.escape(v)}</td>" for v in r[1:]]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def report(run_dir: Path) -> Path:
    run = _json(run_dir / "run_manifest.json") or {}
    comp = _json(run_dir / "completeness_summary.json") or {}
    res = _json(run_dir / "resources_summary.json") or {}
    outs = _json(run_dir / "outputs_summary.json") or {}
    r, code = run.get("run", {}), run.get("code", {})
    facts = pd.DataFrame([
        ("run", r.get("name")), ("entry", r.get("entry")), ("started", r.get("started_at")),
        ("finished", (run.get("completed") or {}).get("at")), ("profile", r.get("profile")),
        ("organisms", ", ".join(r.get("organisms") or [])), ("commit", code.get("commit")),
        ("tracked changes", code.get("dirty")),
        ("complete", comp.get("complete")),
        ("checks ok", f"{comp.get('percent_ok')}% of {comp.get('applicable')}"),
        ("output files", outs.get("n_files")),
        ("output GB", round(outs["bytes"] / 1e9, 2) if outs.get("bytes") else None)],
        columns=["", "value"])
    failures = pd.DataFrame(comp.get("failures") or [])
    by_proc = pd.DataFrame([{"process": p, **v} for p, v in (res.get("by_process") or {}).items()])
    total = pd.DataFrame([res["total"]]) if res.get("total") else None
    backup = (pd.read_csv(run_dir / "backup_manifest.tsv", sep="\t")
              if (run_dir / "backup_manifest.tsv").exists() else None)
    if backup is not None and not backup.empty:
        backup = backup[["unit", "mode", "n_files", "bytes", "archive_bytes", "md5"]]
    parts = [f"<h1>Run {html.escape(str(r.get('name')))}</h1>", _table(facts),
             "<h2>Completeness</h2>", _matrix(run_dir / "completeness_matrix.csv"),
             "<h3>Missing or invalid</h3>", _table(failures),
             "<h2>Resources</h2>", _table(total), _table(by_proc),
             f"<p class='meta'>{res.get('jobs_without_accounting', 0)} SLURM job(s) without "
             "accounting</p>",
             "<h2>Backup of this stage</h2>", _table(backup)]
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Run "
            f"{html.escape(str(r.get('name')))}</title><style>{CSS}</style></head><body>"
            + "\n".join(parts) + "</body></html>")
    out = run_dir / "run_report.html"
    out.write_text(page, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Report of a workflow run.")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    print(f"RUN REPORT -> {report(args.run_dir)}")


if __name__ == "__main__":
    main()
