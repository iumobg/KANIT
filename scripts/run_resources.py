#!/usr/bin/env python3
"""Resources of every task of a workflow run.

Joins, on the SLURM job, three records of each task (a receipt without the job's id, which
happens when SLURM_JOB_ID does not reach the task, takes it from the trace through the task's work
directory: the trace's hash is its last two path parts, the second cut to six characters):
- its receipt: process, step, key, attempt, cores and memory asked for, start, end;
- Nextflow's trace (raw values): real time, %CPU, peak RSS;
- SLURM's accounting (sacct.tsv, which the SACCT_DUMP task writes): state, elapsed
  time, CPU time, MaxRSS, memory asked for, node.

  run_resources.py --run-dir runs/nextflow/<run>

Writes resources.csv (one row per task) and resources_summary.json, with these per
process and in total: tasks, core hours allocated, CPU hours used, the largest
MaxRSS and the longest elapsed time. MaxRSS can include the page cache, so it is
read next to the trace's peak RSS.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SACCT_FORMAT = ("JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocCPUS,ReqMem,"
                "MaxRSS,TotalCPU,NodeList")
_UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def seconds(text) -> float | None:
    """sacct time: [D-]HH:MM:SS[.fff] or MM:SS[.fff]."""
    if not isinstance(text, str) or not text.strip() or text in ("INVALID", "UNLIMITED"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def size_bytes(text) -> float | None:
    """sacct size: 123456K, 1.5G, 4000M, 4000Mc or 0 (bytes when no unit)."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = re.fullmatch(r"([\d.]+)([KMGT]?)[cn]?", text.strip())
    if not m:
        return None
    return float(m.group(1)) * _UNITS.get(m.group(2), 1)


def read_sacct(path: Path) -> pd.DataFrame:
    """One row per job: the job line's state, times and request, the largest MaxRSS of
    its steps."""
    cols = ["job_id", "state", "partition", "node", "elapsed_s", "alloc_cpus", "req_mem_bytes",
            "total_cpu_s", "max_rss_bytes"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    s = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    if s.empty:
        return pd.DataFrame(columns=cols)
    s["job_id"] = s["JobID"].str.split(".").str[0]
    rss = s["MaxRSS"].map(size_bytes).groupby(s["job_id"]).max()
    jobs = s[~s["JobID"].str.contains(".", regex=False)].set_index("job_id")
    return pd.DataFrame({
        "job_id": jobs.index, "state": jobs["State"].to_numpy(),
        "partition": jobs["Partition"].to_numpy(), "node": jobs["NodeList"].to_numpy(),
        "elapsed_s": jobs["Elapsed"].map(seconds).to_numpy(),
        "alloc_cpus": pd.to_numeric(jobs["AllocCPUS"], errors="coerce").to_numpy(),
        "req_mem_bytes": jobs["ReqMem"].map(size_bytes).to_numpy(),
        "total_cpu_s": jobs["TotalCPU"].map(seconds).to_numpy(),
        "max_rss_bytes": rss.reindex(jobs.index).to_numpy()})


def read_receipts(run_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((run_dir / "tasks").glob("*/*/receipt.json")):
        r = json.loads(f.read_text())
        rows.append({"process": r.get("process") or f.parent.parent.name, "step": r.get("step"),
                     "key": r.get("key"), "attempt": r.get("attempt"), "cpus": r.get("cpus"),
                     "memory": r.get("memory"), "started_at": r.get("started_at"),
                     "finished_at": r.get("finished_at"), "seconds": r.get("seconds"),
                     "job_id": (r.get("slurm") or {}).get("job_id"), "stub": r.get("stub"),
                     "hash": _hash(r.get("work_dir"))})
    return pd.DataFrame(rows, columns=["process", "step", "key", "attempt", "cpus", "memory",
                                       "started_at", "finished_at", "seconds", "job_id", "stub",
                                       "hash"])


def _hash(work_dir) -> str | None:
    """Nextflow's short hash of a task work directory: '.../nf-work/53/e32f57abc…' → '53/e32f57'."""
    parts = Path(str(work_dir)).parts if work_dir else ()
    return f"{parts[-2]}/{parts[-1][:6]}" if len(parts) >= 2 else None


def read_trace(run_dir: Path) -> pd.DataFrame:
    cols = ["job_id", "hash", "trace_realtime_s", "trace_cpu_percent", "trace_peak_rss_bytes"]
    f = run_dir / "trace.tsv"
    if not f.exists():
        return pd.DataFrame(columns=cols)
    t = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    num = {c: pd.to_numeric(t[c].str.rstrip("%"), errors="coerce")
           for c in ("realtime", "%cpu", "peak_rss") if c in t}
    return pd.DataFrame({"job_id": t["native_id"],
                         "hash": t["hash"] if "hash" in t else None,
                         "trace_realtime_s": num.get("realtime", pd.Series(dtype=float)) / 1000,
                         "trace_cpu_percent": num.get("%cpu"),
                         "trace_peak_rss_bytes": num.get("peak_rss")})


def resources(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    r = read_receipts(run_dir)
    trace = read_trace(run_dir)
    by_hash = trace.dropna(subset=["hash"]).drop_duplicates("hash").set_index("hash")["job_id"]
    r["job_id"] = r["job_id"].astype(object)
    missing = r["job_id"].isna() | (r["job_id"].astype(str).str.strip() == "")
    found = r.loc[missing, "hash"].map(by_hash).astype(object)
    r.loc[missing, "job_id"] = found.where(found.notna(), None)
    table = (r.drop(columns="hash").merge(read_sacct(run_dir / "sacct.tsv"), on="job_id", how="left")
              .merge(trace.drop(columns="hash").drop_duplicates("job_id"), on="job_id", how="left"))
    table["core_hours"] = table["alloc_cpus"] * table["elapsed_s"] / 3600
    table["cpu_hours"] = table["total_cpu_s"] / 3600

    def summary(g: pd.DataFrame) -> dict:
        def top(col):
            v = g[col].max()
            return None if pd.isna(v) else float(v)
        return {"tasks": len(g), "core_hours": round(float(g["core_hours"].sum()), 2),
                "cpu_hours": round(float(g["cpu_hours"].sum()), 2),
                "max_rss_gib": None if top("max_rss_bytes") is None
                else round(top("max_rss_bytes") / 1024 ** 3, 2),
                "max_trace_peak_rss_gib": None if top("trace_peak_rss_bytes") is None
                else round(top("trace_peak_rss_bytes") / 1024 ** 3, 2),
                "max_elapsed_h": None if top("elapsed_s") is None
                else round(top("elapsed_s") / 3600, 2)}

    by_process = {str(p): summary(g) for p, g in table.groupby("process", sort=True)}
    return table, {"total": summary(table), "by_process": by_process,
                   "jobs_without_accounting": int(table["job_id"].notna().sum()
                                                  - table["state"].notna().sum())}


def main():
    ap = argparse.ArgumentParser(description="Resources of every task of a run.")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    table, summary = resources(args.run_dir)
    table.to_csv(args.run_dir / "resources.csv", index=False)
    (args.run_dir / "resources_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    t = summary["total"]
    print(f"RESOURCES — {t['tasks']} tasks, {t['core_hours']} core hours allocated, "
          f"{t['cpu_hours']} CPU hours used")


if __name__ == "__main__":
    main()
