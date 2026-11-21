#!/usr/bin/env python3
"""Completeness of a workflow run: every output that the output contract
(config/output_contract.yaml, lib/contract.py) lists for the steps of the entry, for
every organism and panel model, exists and is valid.

Valid: a JSON file parses; a CSV file passes its table schema; any other file is not
empty. A step a model does not reach is "n/a": the steps after the cross-validation
design for a model that is not evaluable.

  completeness.py check --entry main --organisms ecoli,kpneumoniae --out-dir runs/nextflow/<run>
  completeness.py gate --out-dir runs/nextflow/<run>

Writes completeness.csv (one row per check), completeness_matrix.csv (unit × step:
ok, missing, invalid or n/a) and completeness_summary.json ("complete": every check
ok or n/a). The run's last task fails when the run is not complete.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import contract, panel  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402


def _status(path: Path, kind: str, table: dict | None = None) -> tuple[str, str]:
    if not path.exists():
        return "missing", ""
    if path.stat().st_size == 0:
        return "invalid", "empty file"
    try:
        if kind == "json":
            json.loads(path.read_text())
        elif kind == "csv" and table is not None:
            problems = contract.validate_csv(path, table)
            if problems:
                return "invalid", "; ".join(problems[:3])
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
        return "invalid", f"{type(e).__name__}: {e}"
    return "ok", ""


def step_checks(step: dict) -> list[dict]:
    """Every output of a step: {kind, location, file, table, evaluable_only}."""
    c = contract.load()
    model_skip = bool(step.get("evaluable_only"))
    evaluable_tables = step.get("evaluable_tables")
    out = []
    for t in step.get("tables") or []:
        spec = c["tables"][t]
        only = model_skip or (evaluable_tables is not None and t in evaluable_tables)
        out.append({"kind": "csv", "location": spec["location"], "file": spec["file"],
                    "table": t, "evaluable_only": only})
    if step.get("summary"):
        s = step["summary"]
        out.append({"kind": "json", "location": s["location"], "file": s["file"], "table": None,
                    "evaluable_only": model_skip or bool(s.get("evaluable_only"))})
    for f in step.get("files") or []:
        out.append({"kind": f["kind"], "location": f["location"], "file": f["file"],
                    "table": None, "evaluable_only": model_skip or bool(f.get("evaluable_only"))})
    return out


def run_checks(entry: str, organisms: list[str], config: dict) -> pd.DataFrame:
    c = contract.load()
    steps = {k: v for k, v in c["steps"].items() if v["entry"] == entry}
    rows: list[dict] = []

    def check(unit, name, step, *, organism=None, antibiotic=None, evaluable=True):
        for x in step_checks(step):
            p = contract.output_path(x["location"], x["file"], config, organism, antibiotic)
            if x["evaluable_only"] and not evaluable:
                status, detail = "n/a", "not evaluable"
            else:
                table = c["tables"][x["table"]] if x["table"] else None
                status, detail = _status(p, x["kind"], table)
            rows.append({"unit": unit, "step": name, "file": str(p), "status": status,
                         "detail": detail})

    decisions_file = resolve_path("panel_dir", config=config) / "panel_decisions.csv"
    decisions = (panel.read_panel(decisions_file) if decisions_file.exists()
                 else pd.DataFrame(columns=["organism", "antibiotic", "decision"]))
    included = decisions[(decisions["decision"] == panel.INCLUDED)
                         & decisions["organism"].isin(organisms)]
    modelled = sorted(set(included["organism"]))
    evaluable = {}
    for row in included.itertuples():
        design = resolve_path("cv_dir", organism=row.organism, antibiotic=row.antibiotic,
                              config=config) / "cv_design.json"
        evaluable[(row.organism, row.antibiotic)] = (
            not design.exists()) or bool(json.loads(design.read_text()).get("evaluable"))
    for name, step in steps.items():
        level = step["level"]
        if level == "global":
            if step.get("needs_evaluable") and not any(evaluable.values()):
                continue
            check("all", name, step)
        elif level == "organism":
            for org in organisms:
                check(org, name, step, organism=org)
        elif level == "modelled_organism":
            for org in modelled:
                check(org, name, step, organism=org)
        elif level == "model":
            for (org, ab), ev in evaluable.items():
                check(f"{org}__{ab}", name, step, organism=org, antibiotic=ab, evaluable=ev)
    return pd.DataFrame(rows, columns=["unit", "step", "file", "status", "detail"])


def matrix(table: pd.DataFrame) -> pd.DataFrame:
    order = {"missing": 3, "invalid": 2, "ok": 1, "n/a": 0}
    worst = (table.assign(rank=table["status"].map(order))
             .groupby(["unit", "step"], sort=False)["rank"].max().map({v: k for k, v in
                                                                      order.items()}))
    steps = list(dict.fromkeys(table["step"]))
    return worst.unstack("step").reindex(columns=steps).fillna("").reset_index()


def gate(out_dir: Path) -> None:
    """Fail when the run is not complete (the workflow's last task)."""
    s = json.loads((out_dir / "completeness_summary.json").read_text())
    if not s["complete"]:
        lines = [f"  {f['unit']} {f['step']}: {f['status']} {f['file']} {f['detail']}"
                 for f in s["failures"][:20]]
        sys.exit(f"INCOMPLETE RUN — {len(s['failures'])} missing or invalid output(s):\n"
                 + "\n".join(lines))
    print(f"COMPLETE — {s['percent_ok']}% of {s['applicable']} checks ok")


def main():
    ap = argparse.ArgumentParser(description="Completeness of a workflow run.")
    ap.add_argument("command", choices=["check", "gate"])
    ap.add_argument("--entry", choices=["main", "DOWNLOAD", "CONTEXT", "KB", "CABBAGE"])
    ap.add_argument("--organisms", help="comma-separated registry ids")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "gate":
        gate(args.out_dir)
        return
    if not args.entry or not args.organisms:
        ap.error("check needs --entry and --organisms")
    table = run_checks(args.entry, args.organisms.split(","), load_config())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "completeness.csv", index=False)
    if not table.empty:
        matrix(table).to_csv(args.out_dir / "completeness_matrix.csv", index=False)
    bad = table[table["status"].isin(["missing", "invalid"])]
    applicable = table[table["status"] != "n/a"]
    summary = {"entry": args.entry, "complete": bad.empty, "checks": len(table),
               "applicable": len(applicable),
               "percent_ok": round(100 * (applicable["status"] == "ok").mean(), 2)
               if len(applicable) else 100.0,
               "failures": bad[["unit", "step", "file", "status", "detail"]].to_dict("records")}
    (args.out_dir / "completeness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"COMPLETENESS — {args.entry}: {summary['percent_ok']}% of {summary['applicable']} "
          f"checks ok; {len(bad)} missing or invalid")


if __name__ == "__main__":
    main()
