#!/usr/bin/env python3
"""Step 14 — pyseer LMM of one panel pair (protocol §8.5).

Subcommands, so that pyseer can run in its own container:

  prep  pyseer's inputs in pyseer_dir: phenotypes.tsv, tested.Rtab (the CPSS
        prefilter and the candidates), background.Rtab, kinship.tsv (every
        pyseer.kinship_every-th unitig of the model) and run_pyseer.sh
  lmm   runs run_pyseer.sh with the pyseer found on PATH (or AMR_PYSEER_BIN); in
        the workflow the tools container runs the script itself
  post  Bonferroni over the tested patterns, λ and QQ points of the tested and of
        the background patterns, and the pyseer layer (layers_dir/pyseer.csv)
  all   prep, lmm and post

Inputs: candidates_file, cpss_dir/prefilter.csv, the model matrix. A model that
is not evaluable exits with status 3, as in 04. The rule is in lib/pyseer_lmm.py.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import folds  # noqa: E402
from lib import pyseer_lmm as pl  # noqa: E402
from lib.config import get_target, load_config, resolve_path, resolve_tool  # noqa: E402
from lib.io_utils import run_logged  # noqa: E402
from lib.matrix_store import ModelMatrix  # noqa: E402


def prep(mm, out_dir: Path, cfg, *, prefilter_file: Path, candidates_file: Path,
         cpu: int, structure=None) -> dict:
    tested = set(pd.read_csv(prefilter_file)["pattern_id"].astype(int))
    tested |= set(pd.read_csv(candidates_file)["pattern_id"].astype(int))
    info = pl.write_inputs(mm, out_dir, tested, every=cfg["kinship_every"],
                           background_max=cfg["background_max"], structure=structure)
    if structure is not None:
        info["structure_matrix"] = str(structure.dir)
    script = out_dir / "run_pyseer.sh"
    script.write_text(pl.command(out_dir, cpu))
    script.chmod(0o755)
    (out_dir / "inputs.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def lmm(out_dir: Path) -> None:
    tool = resolve_tool("pyseer")
    if not tool:
        sys.exit("ERROR: pyseer not found on PATH (or set AMR_PYSEER_BIN); run "
                 f"{out_dir / 'run_pyseer.sh'} in the tools container instead.")
    os.environ["PYSEER"] = tool
    run_logged(["bash", out_dir / "run_pyseer.sh"], out_dir / "run_pyseer.log")


def post(out_dir: Path, cfg, *, candidates_file: Path, layers_dir: Path) -> dict:
    tested = pl.read_assoc(out_dir / "tested_assoc.tsv")
    submitted = set(pd.read_csv(out_dir / "tested_patterns.csv")["pattern_id"].astype(int))
    absent = sorted(submitted - set(tested["pattern_id"]))
    if absent:
        sys.exit(f"ERROR: pyseer tested {len(tested)} of {len(submitted)} patterns "
                 f"(missing e.g. {absent[:5]}); see tested_assoc.log.")
    candidates = pd.read_csv(candidates_file)["pattern_id"].astype(int)
    layer, threshold = pl.results(tested, candidates, cfg["alpha"])
    background = pl.read_assoc(out_dir / "background_assoc.tsv")
    layers_dir.mkdir(parents=True, exist_ok=True)
    layer.to_csv(layers_dir / "pyseer.csv", index=False)
    tested.assign(significant=tested["lrt-pvalue"].lt(threshold)).to_csv(
        out_dir / "pyseer_tested.csv", index=False)
    pd.concat([pl.qq_points(tested["lrt-pvalue"]).assign(set="tested"),
               pl.qq_points(background["lrt-pvalue"]).assign(set="background")]).to_csv(
        out_dir / "pyseer_qq.csv", index=False)
    version = out_dir / "pyseer_version.txt"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **json.loads((out_dir / "inputs.json").read_text()), **cfg,
        "bonferroni_threshold": threshold,
        "n_significant": int(tested["lrt-pvalue"].lt(threshold).sum()),
        "n_candidates": int(len(layer)), "n_candidates_passing": int(layer["passes"].sum()),
        "lambda_tested": pl.genomic_lambda(tested["lrt-pvalue"]),
        "lambda_background": pl.genomic_lambda(background["lrt-pvalue"]),
        "pyseer_version": version.read_text().strip() if version.exists() else None,
    }
    (out_dir / "pyseer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    config = load_config()
    default_org, default_ab = get_target(config=config)
    ap = argparse.ArgumentParser(description="pyseer LMM of one panel pair.")
    ap.add_argument("command", choices=["prep", "lmm", "post", "all"])
    ap.add_argument("--organism", default=default_org)
    ap.add_argument("--antibiotic", default=default_ab)
    ap.add_argument("--threads", type=int, default=int(config["preprocessing"]["threads"]))
    args = ap.parse_args()
    cfg = config["pyseer"]

    def path(key):
        return resolve_path(key, organism=args.organism, antibiotic=args.antibiotic,
                            config=config)

    out_dir = path("pyseer_dir")
    print(f"PYSEER — {args.organism} / {args.antibiotic} — {args.command}")
    if not json.loads((path("cv_dir") / "cv_design.json").read_text())["evaluable"]:
        print("Model not evaluable (see 04's seeds.csv): no pyseer.")
        sys.exit(folds.NOT_EVALUABLE)
    if args.command in ("prep", "all"):
        # The budgeted arms of the unitig–k-mer comparison: population structure from arm A's full matrix
        structure = ModelMatrix(path("source_matrix_dir")) if config.get("m3") else None
        info = prep(ModelMatrix(path("matrix_dir")), out_dir, cfg,
                    prefilter_file=path("cpss_dir") / "prefilter.csv",
                    candidates_file=path("candidates_file"), cpu=args.threads,
                    structure=structure)
        print(f"  inputs: {info['n_tested']} tested, {info['n_background']} background, "
              f"kinship of {info['n_kinship_unitigs']} unitigs")
    if args.command in ("lmm", "all"):
        lmm(out_dir)
    if args.command in ("post", "all"):
        s = post(out_dir, cfg, candidates_file=path("candidates_file"),
                 layers_dir=path("layers_dir"))
        print(f"  {s['n_significant']} of {s['n_tested']} significant (Bonferroni "
              f"{s['bonferroni_threshold']:.2e}); candidates {s['n_candidates_passing']} of "
              f"{s['n_candidates']}; λ tested {s['lambda_tested']:.2f}, background "
              f"{s['lambda_background']:.2f}")


if __name__ == "__main__":
    main()
