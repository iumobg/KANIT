#!/usr/bin/env python3
"""M3-0c — how much of the unitig matrix's reduction is the frequency filter and how
much is the pattern collapse?

The unitig–k-mer comparison needs the two reductions
separated, because only the first one is shared with the k-mer arm:

  unitigs seen  --[frequency filter, min_support]-->  store unitigs
  store unitigs --[frequency filter over the pair]-->  kept unitigs
  kept unitigs  --[pattern collapse]---------------->  patterns

The k-mer arm keeps the filter and drops the collapse, so collapse_ratio =
kept / patterns is the factor the k-mer arm gives up. A ratio near 1 would mean the
unitig abstraction buys almost nothing here; a large ratio means the opposite.

Reads only the summaries the finished run already wrote. Runs in seconds, touches
nothing, needs no compute node.

Outputs (default results/tables/): m3_0c_collapse_ratio.csv, m3_0c_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from lib import panel  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402


def store_rows(organisms, config):
    """One row per organism: what the store-level filter removed."""
    rows = []
    for org in organisms:
        path = resolve_path("unitig_store_dir", organism=org, config=config) / "store_summary.json"
        if not path.exists():
            print(f"  ! no store summary for {org} ({path})")
            continue
        s = json.loads(path.read_text())
        seen, kept = int(s["n_unitigs_seen"]), int(s["n_unitigs"])
        rows.append({
            "organism": org,
            "n_genomes_store": int(s["n_genomes"]),
            "min_support": int(s["min_support"]),
            "n_unitigs_seen": seen,
            "n_unitigs_store": kept,
            "filter_kept_frac": kept / seen if seen else float("nan"),
        })
    return pd.DataFrame(rows)


def model_rows(pairs, config):
    """One row per panel pair: what the pair filter and the collapse left."""
    rows = []
    for org, ab in pairs:
        path = (resolve_path("matrix_dir", organism=org, antibiotic=ab, config=config)
                / "matrix_summary.json")
        if not path.exists():
            print(f"  ! no matrix summary for {org}/{ab}")
            continue
        s = json.loads(path.read_text())
        kept, patterns = int(s["n_unitigs_kept"]), int(s["n_patterns"])
        rows.append({
            "organism": org,
            "antibiotic": ab,
            "n_genomes": int(s["n_genomes"]),
            "n_resistant": int(s["n_resistant"]),
            "n_susceptible": int(s["n_susceptible"]),
            "min_support": int(s["min_support"]),
            "n_unitigs_kept": kept,
            "n_patterns": patterns,
            # > 1 means several kept unitigs share one presence/absence vector
            "collapse_ratio": kept / patterns if patterns else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()

    config = load_config()
    panel_csv = resolve_path("panel_dir", config=config) / "panel_decisions.csv"
    if not panel_csv.exists():
        sys.exit(f"ERROR: no panel at {panel_csv}.")
    pairs = panel.included_pairs(panel_csv)
    if not pairs:
        sys.exit("ERROR: the panel includes no pair.")

    stores = store_rows(sorted({org for org, _ in pairs}), config)
    models = model_rows(pairs, config)
    if models.empty:
        sys.exit("ERROR: no matrix summary found; has the unitig run written the matrices?")

    table = models.merge(stores.drop(columns=["min_support"]), on="organism", how="left")
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "m3_0c_collapse_ratio.csv", index=False)

    r = table["collapse_ratio"]
    summary = {
        "n_pairs_in_panel": len(pairs),
        "n_pairs_measured": int(len(table)),
        "min_support": sorted(table["min_support"].unique().tolist()),
        "collapse_ratio": {
            "min": float(r.min()), "q1": float(r.quantile(0.25)),
            "median": float(r.median()), "q3": float(r.quantile(0.75)),
            "max": float(r.max()),
        },
        "store_filter_kept_frac": {
            org: (None if pd.isna(v) else float(v))
            for org, v in zip(stores["organism"], stores["filter_kept_frac"], strict=True)
        } if not stores.empty else {},
        "totals": {
            "n_unitigs_kept": int(table["n_unitigs_kept"].sum()),
            "n_patterns": int(table["n_patterns"].sum()),
        },
    }
    (args.out / "m3_0c_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{len(table)}/{len(pairs)} pair(s) measured, min_support="
          f"{summary['min_support']}")
    print(f"  collapse ratio (kept unitigs per pattern): median {r.median():.2f}, "
          f"range {r.min():.2f}–{r.max():.2f}")
    worst = table.loc[r.idxmin()]
    print(f"  least collapsed pair: {worst['organism']}/{worst['antibiotic']} "
          f"{worst['n_unitigs_kept']:,} -> {worst['n_patterns']:,} ({r.min():.2f}x)")
    print(f"  -> {args.out / 'm3_0c_collapse_ratio.csv'}")


if __name__ == "__main__":
    main()
