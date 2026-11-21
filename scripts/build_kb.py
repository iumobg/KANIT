#!/usr/bin/env python3
"""Build the knowledge base from the steps' outputs.

Reads the registry, the panel decisions and, for every organism and included
pair, the outputs of the steps (00, 02c, 02d, 03u, 04, 08–14c, 12b); loads them
into a new SQLite file inside one transaction and checks it
(lib.knowledge_base.validate). Nothing is computed but identifiers, carrier bit
strings and the release record. The file is replaced only after the new one has
passed its checks.

Output: paths_organism.kb_dir / kanit.sqlite and kb_report.json.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import knowledge_base as kb  # noqa: E402
from lib import panel, pattern_sets, registry  # noqa: E402
from lib.card_layer import MODES  # noqa: E402
from lib.config import CONFIG_FILE, load_config, resolve_path  # noqa: E402
from lib.matrix_store import ModelMatrix, sha256_file  # noqa: E402
from lib.run_metadata import git_commit_hash, git_is_dirty, version_line  # noqa: E402


class Sources:
    """Reads step outputs and records each file's checksum for source_file."""

    def __init__(self):
        self.files: dict[str, tuple[str, int]] = {}

    @staticmethod
    def key(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(Path(path).resolve())

    def _record(self, path: Path) -> Path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"input missing: {path}")
        self.files[self.key(path)] = (sha256_file(path), path.stat().st_size)
        return path

    def csv(self, path, **kw) -> pd.DataFrame:
        return pd.read_csv(self._record(path), **kw)

    def json(self, path) -> dict:
        return json.loads(self._record(path).read_text())

    def text(self, path) -> str:
        return self._record(path).read_text()

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([(k, s, b) for k, (s, b) in sorted(self.files.items())],
                            columns=["path", "sha256", "bytes"])


def qc_versions(qc_dir: Path, src: Sources) -> dict:
    """CheckM2 and QUAST versions of an organism's genome QC. The version files the
    workflow writes beside the reports are read where they exist: versions.json kept the
    first line of each, which for QUAST is a locale warning, not the version."""
    out = dict(src.json(qc_dir / "versions.json"))
    for tool in ("checkm2", "quast"):
        f = qc_dir / tool / "version.txt"
        if f.exists():
            out[tool] = version_line(src.text(f))
    return out


def _bool(series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The rows of every frame; an empty frame (an empty association set) adds nothing."""
    full = [f for f in frames if len(f)]
    return pd.concat(full or frames[:1], ignore_index=True)


def _graded_set(name: str, path, src: Sources, mid: str) -> dict:
    """One set of graded patterns (lib/pattern_sets.py): its candidate rows with the route,
    and its layers, CARD layer and grades, each holding exactly the set's patterns."""
    ps = pattern_sets.paths(name, path)
    pats = src.csv(ps["patterns"])
    ids = pats["pattern_id"].astype(int)
    out = {"prevalence": src.csv(ps["layers_dir"] / "prevalence.csv"),
           "mda": src.csv(ps["layers_dir"] / "mda.csv"),
           "mda clusters": src.csv(ps["layers_dir"] / "mda_clusters.csv"),
           "grades": src.csv(ps["grades_dir"] / "grades_patterns.csv", keep_default_na=False),
           "card_u": src.csv(ps["card_layer_dir"] / "card_unitigs.csv", keep_default_na=False)}
    for table in ("prevalence", "mda", "mda clusters", "grades"):
        if set(out[table]["pattern_id"].astype(int)) != set(ids.tolist()):
            raise ValueError(f"{mid}: the {table} table does not hold exactly the "
                             f"{pattern_sets.LABEL[name]} patterns")
    if name == pattern_sets.CANDIDATES:
        cand = pd.DataFrame({"model_id": mid, "pattern_id": ids, "route": "model",
                             "source": pats["source"], "gain_rank": pats["gain_rank"],
                             "total_gain": pats["total_gain"]})
    else:
        cand = pd.DataFrame({"model_id": mid, "pattern_id": ids, "route": "association",
                             "source": "pyseer", "gain_rank": None, "total_gain": None})
    return {"ids": ids.to_numpy(dtype=np.int64), "candidate": cand, **out}


# ---- reference data ---------------------------------------------------------------
def organisms(ids) -> pd.DataFrame:
    rows = []
    for o in sorted(ids):
        d = registry.get_organism(o)
        rows.append({"organism_id": o, "name": d["display_name"],
                     "ncbi_taxids": ";".join(str(t) for t in registry.organism_taxids(o)),
                     "gram_stain": d.get("gram_stain"), "phylum": d.get("phylum")})
    return pd.DataFrame(rows)


def antibiotics(ids) -> pd.DataFrame:
    return pd.DataFrame([{"antibiotic_id": a, "drug_class": registry.antibiotic_to_class(a),
                          "card_drug_classes": ";".join(sorted(registry.card_drug_classes(a))),
                          "who_aware": registry.antibiotic_who_aware(a)} for a in sorted(ids)])


def snapshot_of(org: str, config: dict, src: Sources) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(data_snapshot row, NCBI identifiers of the genomes) of one organism (00a)."""
    meta = resolve_path("metadata_file", organism=org, config=config).parent
    snap = src.json(meta / "snapshot.json")
    q = snap["query"]
    row = pd.DataFrame([{
        "organism_id": org, "source": q["source"], "api_version": q["api_version"],
        "queried_at": q["queried_at"], "frozen_at": snap["frozen_at"],
        "filters": json.dumps({"genomes": q["genome_query"], "records": q["record_query"]}),
        "n_genomes": snap["n_genomes"], "sha256": src.files[src.key(meta / "snapshot.json")][0]}])
    ids = src.csv(meta / "genomes.csv", dtype=str, keep_default_na=False)
    ids = ids.set_index("genome_id")
    return row, ids.mask(ids == "")


def genomes_of(org: str, config: dict,
               src: Sources) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(genome rows, phenotype rows, data_snapshot row) of one organism."""
    paths = panel.input_paths(org, config)
    snapshot, ncbi = snapshot_of(org, config, src)
    qc = src.csv(paths["qc_table"], dtype={"genome_id": str})
    unknown = sorted(set(qc["genome_id"]) - set(ncbi.index))
    if unknown:
        raise ValueError(f"{org}: {len(unknown)} genome(s) of the QC table are not in the data "
                         f"snapshot, e.g. {unknown[:3]}")
    ncbi = ncbi.loc[qc["genome_id"]]
    clusters = src.csv(paths["clusters"], dtype={"Genome ID": str})
    lineage = dict(zip(clusters["Genome ID"], clusters["Cluster"].astype(str), strict=True))
    g = pd.DataFrame({
        "genome_id": qc["genome_id"], "organism_id": org,
        "checkm2_completeness": qc["completeness"], "checkm2_contamination": qc["contamination"],
        "n50": qc["n50"], "n_contigs": qc["n_contigs"], "total_length": qc["total_length"],
        "qc_pass": _bool(qc["pass_overall"]), "lineage_cluster": qc["genome_id"].map(lineage),
        "ncbi_taxid": ncbi["taxon_id"].astype(int).to_numpy(),
        "assembly_accession": ncbi["assembly_accession"].to_numpy(),
        "sra_accession": ncbi["sra_accession"].to_numpy(),
        "biosample_accession": ncbi["biosample_accession"].to_numpy()})
    ph = src.csv(paths["phenotypes"], dtype={"Genome ID": str})
    long = ph.melt(id_vars="Genome ID", var_name="antibiotic_id", value_name="resistant").dropna()
    long = long[long["Genome ID"].isin(set(g["genome_id"]))]
    return g, pd.DataFrame({"genome_id": long["Genome ID"], "antibiotic_id": long["antibiotic_id"],
                            "resistant": long["resistant"].astype(int)}), snapshot


# ---- one model -------------------------------------------------------------------
# Settings the build adds to the parameter table beside lib.knowledge_base.PARAMETERS:
# the genome quality gate, read from 02d's summary, and the lineage clustering, from
# config.yaml and the registry's overrides (as 02c merges them): (name, key, what it sets).
QC_SETTINGS = (
    ("qc.completeness_min", "completeness_min",
     "CheckM2 completeness (%) at or above which a genome passes quality control"),
    ("qc.contamination_max", "contamination_max",
     "CheckM2 contamination (%) at or below which a genome passes quality control"))
LINEAGE_SETTINGS = (
    ("lineage.model", "model", "PopPUNK model fitted to the core and accessory distances"),
    ("lineage.refine", "refine", "Whether a boundary refinement follows the PopPUNK fit"),
    ("lineage.min_k", "min_k", "Smallest k-mer length of the PopPUNK sketches"),
    ("lineage.max_k", "max_k", "Largest k-mer length of the PopPUNK sketches"),
    ("lineage.k_step", "k_step", "Step between the k-mer lengths of the PopPUNK sketches"),
    ("lineage.sketch_size", "sketch_size", "Sketch size of PopPUNK"),
    ("lineage.qc", "qc", "Whether PopPUNK's quality control of the sketches runs"),
    ("lineage.max_a_dist", "max_a_dist",
     "PopPUNK quality control: largest accessory distance permitted"),
    ("lineage.length_sigma", "length_sigma",
     "PopPUNK quality control: standard deviations of the length distribution beyond which a "
     "genome is removed"),
    ("lineage.prop_n", "prop_n", "PopPUNK quality control: largest proportion of ambiguous bases"),
    ("lineage.length_range", "length_range",
     "PopPUNK quality control: permitted genome length in bp, [min, max] (null: none)"))


def _setting(values: dict) -> str:
    """One value when every organism has it, else a JSON object by organism."""
    def one(v):
        return v if isinstance(v, str) else json.dumps(v)
    distinct = {json.dumps(v, sort_keys=True) for v in values.values()}
    if len(distinct) == 1:
        return one(next(iter(values.values())))
    return json.dumps(values, sort_keys=True)


def settings(config: dict, orgs: list[str], src: Sources) -> list[tuple[str, str, str]]:
    """The parameter rows of the quality gate and of the lineage clustering."""
    rows = []
    qc = {org: src.json(resolve_path("genome_qc_dir", organism=org, config=config)
                        / f"02d_genome_qc_summary_{org}.json")["thresholds"] for org in orgs}
    for name, key, what in QC_SETTINGS:
        rows.append((name, _setting({o: qc[o][key] for o in orgs}), what))
    lineage = {}
    for org in orgs:
        p = dict(config.get("lineage") or {})
        try:
            p.update(registry.get_organism(org).get("lineage") or {})
        except KeyError:            # not in the registry: the globals, as in 02c
            pass
        lineage[org] = p
    for name, key, what in LINEAGE_SETTINGS:
        rows.append((name, _setting({o: lineage[o].get(key) for o in orgs}), what))
    return rows


def model_tables(org: str, ab: str, config: dict, src: Sources) -> dict[str, pd.DataFrame]:
    def path(key):
        return resolve_path(key, organism=org, antibiotic=ab, config=config)

    mid = f"{org}__{ab}"
    mm = ModelMatrix(path("matrix_dir"))
    summary = src.json(mm.dir / "matrix_summary.json")
    mg = src.csv(mm.dir / "genomes.csv", dtype={"Genome ID": str})
    design = src.json(path("cv_dir") / "cv_design.json")
    out: dict[str, pd.DataFrame] = {}
    out["model_genome"] = pd.DataFrame({
        "model_id": mid, "genome_id": mg["Genome ID"], "row_index": np.arange(len(mg)),
        "resistant": mg["label"].astype(int), "lineage_cluster": mg["lineage"].astype(str)})
    model = {"model_id": mid, "organism_id": org, "antibiotic_id": ab,
             "n_genomes": len(mg), "n_resistant": int(mg["label"].sum()),
             "n_susceptible": int((mg["label"] == 0).sum()),
             "n_lineages": int(design["n_lineages"]),
             "largest_lineage_share": float(design["largest_lineage_share"]),
             "n_unitigs": int(summary["n_unitigs_kept"]), "n_patterns": int(summary["n_patterns"]),
             "evaluable": int(bool(design["evaluable"])), "final_n_trees": None,
             "final_params": None}
    out["model"] = pd.DataFrame([model])
    if not design["evaluable"]:
        return out

    final = src.json(path("cv_dir") / "final" / "record.json")
    out["model"].loc[0, ["final_n_trees", "final_params"]] = [
        int(final["n_trees"]), json.dumps(final["params"], sort_keys=True)]
    metrics = src.json(path("cv_dir") / "metrics.json")
    rows = []
    for arm, m in metrics["arms"].items():
        for k, v in m.items():
            if k != "ci" and isinstance(v, (int, float)):
                ci = m["ci"] if k == "roc_auc" else {}
                rows.append({"model_id": mid, "arm": arm, "metric": k, "value": float(v),
                             "ci_low": ci.get("low"), "ci_high": ci.get("high")})
    for k, ci in metrics.get("bootstrap", {}).items():
        if not (isinstance(ci, dict) and "_minus_" in k):      # n_boot, seed, level, ...
            continue
        a, b = k.split("_minus_")
        rows.append({"model_id": mid, "arm": k, "metric": "roc_auc",
                     "value": metrics["arms"][a]["roc_auc"] - metrics["arms"][b]["roc_auc"],
                     "ci_low": ci["low"], "ci_high": ci["high"]})
    out["model_metric"] = pd.DataFrame(rows)

    lp = src.json(path("label_permutation_dir") / "label_permutation.json")
    across = src.csv(path("cross_model_dir") / "label_permutation.csv", keep_default_na=False)
    row = across[(across["organism"] == org) & (across["antibiotic"] == ab)]
    if len(row) != 1 or row["status"].iloc[0] != "tested":
        raise ValueError(f"{mid}: no tested row in the cross-model label permutation table")
    out["label_permutation"] = pd.DataFrame([{
        "model_id": mid, **{k: lp[k] for k in ("auc_observed", "null_mean", "null_sd", "z",
                                                "n_permutations", "p")},
        "q": float(row["q"].iloc[0]), "flag": row["flag"].iloc[0]}])

    cpss = src.csv(path("cpss_dir") / "cpss.csv")
    tested = src.csv(path("pyseer_dir") / "pyseer_tested.csv", keep_default_na=False,
                     na_values=[""])
    pyseer = src.json(path("pyseer_dir") / "pyseer_summary.json")
    if pyseer["n_tested"] != len(tested):
        raise ValueError(f"{mid}: pyseer_tested.csv disagrees with pyseer_summary.json")
    out["tool_versions"] = pd.DataFrame(
        [{"tool": "pyseer", "version": pyseer["pyseer_version"]}]
        + [{"tool": t, "version": v} for t, v in final.get("versions", {}).items()])
    # the graded patterns: the candidates (route model) and the patterns nominated by
    # association (route association), each with its own layers
    sets = {name: _graded_set(name, path, src, mid) for name in pattern_sets.SETS}
    cids, aids = sets[pattern_sets.CANDIDATES]["ids"], sets[pattern_sets.ASSOCIATION]["ids"]
    if set(cids.tolist()) & set(aids.tolist()):
        raise ValueError(f"{mid}: a pattern is both a candidate and nominated by association")
    if not set(aids.tolist()) <= set(tested["pattern_id"].astype(int)):
        raise ValueError(f"{mid}: a pattern nominated by association was not tested by pyseer")
    gids = np.concatenate([cids, aids])
    prev, mda, clus, card_u, grades = (
        _concat([s[k] for s in sets.values()])
        for k in ("prevalence", "mda", "mda clusters", "card_u", "grades"))

    # patterns: the graded ones (with carriers and members) and every tested pattern
    pats = src.csv(mm.dir / "patterns.csv").set_index("pattern_id")
    tested_ids = set(cpss["pattern_id"].astype(int)) | set(tested["pattern_id"].astype(int))
    all_ids = np.array(sorted(tested_ids | set(gids.tolist())), dtype=np.int64)
    carriers = mm.columns(gids)
    blob = {int(p): np.packbits(carriers[:, i]).tobytes() for i, p in enumerate(gids)}
    out["pattern"] = pd.DataFrame({
        "model_id": mid, "pattern_id": all_ids, "n_members": pats.loc[all_ids, "n_members"].to_numpy(),
        "n_present": pats.loc[all_ids, "n_present"].to_numpy(),
        "carriers": [blob.get(int(p)) for p in all_ids]})
    out["candidate"] = _concat([s["candidate"] for s in sets.values()])

    out["prevalence_result"] = prev.assign(model_id=mid)[[
        "model_id", "pattern_id", "present_resistant", "present_susceptible", "prev_resistant",
        "prev_susceptible", "delta", "direction", "fisher_p", "q"]].assign(
        passes=_bool(prev["passes"]).to_numpy())
    out["mda_result"] = mda.assign(model_id=mid)[[
        "model_id", "pattern_id", "n_fold_models_using", "auc_observed", "auc_permuted", "mda",
        "n_permuted_ge_observed", "p", "q"]].assign(passes=_bool(mda["passes"]).to_numpy())
    out["mda_cluster"] = clus.assign(model_id=mid)[[
        "model_id", "pattern_id", "cluster", "cluster_size", "cluster_patterns", "mda", "p",
        "q"]].assign(passes=_bool(clus["passes"]).to_numpy())

    extra = sorted(set(gids.tolist()) - set(cpss["pattern_id"].astype(int)))
    out["cpss_result"] = pd.concat([
        pd.DataFrame({"model_id": mid, "pattern_id": cpss["pattern_id"], "in_prefilter": 1,
                      "chi2": cpss["chi2"], "n_selected": cpss["n_selected"], "pi": cpss["pi"],
                      "passes": _bool(cpss["stable"])}),
        pd.DataFrame({"model_id": mid, "pattern_id": extra, "in_prefilter": 0, "chi2": None,
                      "n_selected": 0, "pi": 0.0, "passes": 0})], ignore_index=True)
    out["pyseer_result"] = pd.DataFrame({
        "model_id": mid, "pattern_id": tested["pattern_id"], "af": tested["af"],
        "beta": tested["beta"], "beta_se": tested["beta-std-err"], "lrt_p": tested["lrt-pvalue"],
        "notes": tested["notes"].fillna("").astype(str), "passes": _bool(tested["significant"])})

    # unitigs, members and CARD annotations of the graded patterns
    ids = [kb.unitig_id(s) for s in card_u["sequence"]]
    card_u = card_u.assign(unitig_id=[i for i, _ in ids], canonical=[c for _, c in ids])
    out["unitig"] = pd.DataFrame({"unitig_id": card_u["unitig_id"], "sequence": card_u["canonical"],
                                  "length": card_u["canonical"].str.len()})
    out["pattern_member"] = card_u.assign(model_id=mid)[["model_id", "pattern_id", "unitig_id"]]
    out["card_annotation"] = pd.concat([pd.DataFrame({
        "model_id": mid, "unitig_id": card_u["unitig_id"], "mode": mode,
        "state": card_u[f"{mode}_state"], "reasons": card_u[f"{mode}_reasons"].astype(str),
        "n_b": card_u[f"{mode}_n_b"].astype(int), "located_genomes": card_u["located_genomes"]})
        for mode in MODES], ignore_index=True)
    hits = [(u, f"ARO:{a}") for u, aros in zip(card_u["unitig_id"], card_u["allele_aware_aros"],
                                                strict=True) for a in str(aros).split(";") if a]
    out["card_hit"] = pd.DataFrame(sorted(set(hits)), columns=["unitig_id", "aro_accession"]).assign(
        model_id=mid)[["model_id", "unitig_id", "aro_accession"]]

    primary = config["grading"]["primary_mode"]
    out["grade"] = pd.concat([pd.DataFrame({
        "model_id": mid, "pattern_id": grades["pattern_id"], "mode": mode,
        "primary_mode": int(mode == primary), "card_state": grades[f"{mode}_card_state"],
        "card_reasons": grades[f"{mode}_card_reasons"].astype(str),
        "n_layers": grades["n_layers"].astype(int), "layers_passed": grades["layers_passed"],
        "grade": grades[f"{mode}_grade"]}) for mode in MODES], ignore_index=True)
    return out


def aro_table(org: str, config: dict, src: Sources) -> tuple[pd.DataFrame, dict]:
    rgi = resolve_path("rgi_dir", organism=org, config=config)
    hits = src.csv(rgi / "rgi_hits.csv", dtype={"aro": str}, keep_default_na=False,
                   usecols=["aro", "aro_name", "model_type", "gene_family", "drug_class",
                            "mechanism"])
    a = hits.drop_duplicates("aro").rename(columns={"aro_name": "name"})
    a.insert(0, "aro_accession", "ARO:" + a.pop("aro"))
    return a, src.json(rgi / "rgi_summary.json")


# ---- build ----------------------------------------------------------------------
def build(config: dict, out_file: Path, *, kb_version: str, allow_missing_context: bool = False,
          license: str | None = None, doi: str | None = None) -> dict:
    src = Sources()
    decisions = src.csv(resolve_path("panel_dir", config=config) / "panel_decisions.csv")
    included = decisions[decisions["decision"] == panel.INCLUDED]
    orgs = sorted(set(decisions["organism"]))
    tables: dict[str, list[pd.DataFrame]] = {}

    def add(name, df):
        tables.setdefault(name, []).append(df)

    phenos, rgi_summaries = [], {}
    for org in orgs:
        g, ph, snapshot = genomes_of(org, config, src)
        add("genome", g)
        add("data_snapshot", snapshot)
        phenos.append(ph)
    ext_versions, no_context = [], []
    for org in sorted(set(included["organism"])):
        a, rgi_summaries[org] = aro_table(org, config, src)
        add("aro", a)
        ext = resolve_path("external_dir", organism=org, config=config)
        calls = src.csv(ext / "amrfinder_calls.csv", dtype={"genome_id": str},
                        keep_default_na=False)
        add("external_call", calls.assign(call_index=calls.groupby("genome_id").cumcount()))
        add("external_comparison", src.csv(ext / "comparison.csv"))
        ext_versions.append(src.json(ext / "versions.json"))
        ext_versions.append(src.json(resolve_path("lineage_dir", organism=org, config=config)
                                     / "versions.json"))
        ext_versions.append(qc_versions(resolve_path("genome_qc_dir", organism=org,
                                                     config=config), src))
        ext_versions.append(src.json(resolve_path("unitig_store_dir", organism=org,
                                                  config=config) / "store_summary.json"
                                     ).get("tools", {}))
        ctx = resolve_path("context_dir", organism=org, config=config) / "unitig_context.csv"
        if ctx.exists() or not allow_missing_context:
            add("unitig_context", src.csv(ctx))
        else:
            no_context.append(org)
    for row in included.itertuples():
        for name, df in model_tables(row.organism, row.antibiotic, config, src).items():
            add(name, df)
    versions = pd.concat(tables.pop("tool_versions", [pd.DataFrame(columns=["tool", "version"])]))

    ab_ids = set(decisions["antibiotic"]) | set(pd.concat(phenos)["antibiotic_id"])
    add("organism", organisms(orgs))
    add("antibiotic", antibiotics(ab_ids))
    add("phenotype", pd.concat(phenos, ignore_index=True))
    add("panel_decision", pd.DataFrame({
        "organism_id": decisions["organism"], "antibiotic_id": decisions["antibiotic"],
        "decision": decisions["decision"], "reason": decisions["reason"],
        "n_resistant": decisions["n_resistant"], "n_susceptible": decisions["n_susceptible"]}))
    frames = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    if "unitig_context" in frames:          # cache rows of unitigs that are no longer candidates
        ctx = frames["unitig_context"]
        frames["unitig_context"] = ctx[ctx["unitig_id"].isin(set(frames["unitig"]["unitig_id"]))]
    for key, cols in (("unitig", ["unitig_id"]), ("aro", ["aro_accession"])):
        if key in frames:
            frames[key] = frames[key].drop_duplicates()
            if frames[key].duplicated(cols).any():
                raise ValueError(f"{key}: one identifier with two different records")

    cards = {s.get("card_version") for s in rgi_summaries.values()}
    if len(cards) > 1:
        raise ValueError(f"organisms were annotated with different CARD versions: {cards}")
    tools: dict = {}
    rows = [{"tool": "rgi", "version": s.get("rgi_version")} for s in rgi_summaries.values()]
    for v in ext_versions:
        rows += [{"tool": t, "version": x} for t, x in v.items() if t != "n_genomes"]
    versions = pd.concat([versions, pd.DataFrame(rows, columns=["tool", "version"])],
                         ignore_index=True)
    for tool, vs in versions.groupby("tool")["version"]:
        if vs.nunique() > 1:
            raise ValueError(f"models were built with different {tool} versions: {sorted(set(vs))}")
        tools[tool] = vs.iloc[0]
    frames["parameter"] = pd.DataFrame(kb.parameters(config) + settings(config, orgs, src),
                                       columns=["name", "value", "description"])
    manifest = src.json(resolve_path("databases_manifest", config=config))
    frames["reference_database"] = pd.DataFrame([{
        "name": n, "version": e["version"], "downloaded_on": e["downloaded_on"],
        "source": e.get("source"), "n_files": len(e["files"]),
        "sha256": hashlib.sha256(json.dumps(e["files"], sort_keys=True).encode()).hexdigest()}
        for n, e in sorted(manifest.items())])
    if "card" in manifest and cards and manifest["card"]["version"] not in cards:
        raise ValueError(f"RGI ran with CARD {cards}, the manifest records "
                         f"{manifest['card']['version']}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.name + ".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        kb.create(conn)
        order = ["organism", "data_snapshot", "antibiotic", "aro", "genome", "phenotype", "panel_decision",
                 "model", "model_genome", "model_metric", "label_permutation", "unitig",
                 "pattern", "pattern_member", "candidate", "prevalence_result", "mda_result",
                 "mda_cluster", "cpss_result", "pyseer_result", "card_hit", "card_annotation",
                 "unitig_context", "grade", "external_call", "external_comparison",
                 "reference_database", "parameter"]
        with conn:
            for name in order:
                if name in frames:
                    kb.insert(conn, name, frames[name])
            kb.insert(conn, "source_file", src.table())
            kb.insert(conn, "release", pd.DataFrame([{
                "kb_version": kb_version, "schema_version": kb.SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config_sha256": hashlib.sha256(Path(CONFIG_FILE).read_bytes()).hexdigest(),
                "code_commit": git_commit_hash(), "code_dirty": git_is_dirty(),
                "card_version": next(iter(cards)) if cards else None,
                "tools": json.dumps(tools, sort_keys=True), "license": license, "doi": doi}]))
        report = kb.validate(conn)
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp, out_file)
    days = sorted({e["downloaded_on"] for e in manifest.values()})
    report = {"kb_version": kb_version, "file": str(out_file),
              "sha256": sha256_file(out_file), "organisms_without_context": no_context,
              "databases_downloaded_on": days, **report}
    (out_file.parent / "kb_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="Build the KANIT knowledge base.")
    ap.add_argument("--kb-version", default="1.0.0")
    ap.add_argument("--allow-missing-context", action="store_true",
                    help="build without the NCBI context of an organism (development only)")
    ap.add_argument("--license", default=None,
                    help="data licence of the release, an SPDX identifier (e.g. CC-BY-4.0)")
    ap.add_argument("--doi", default=None, help="DOI reserved for the deposit of the release")
    ap.add_argument("--out", type=Path, default=None,
                    help="SQLite file (default: paths_organism.kb_dir/kanit.sqlite)")
    args = ap.parse_args()
    out = args.out or resolve_path("kb_dir", config=config) / "kanit.sqlite"
    r = build(config, out, kb_version=args.kb_version,
              allow_missing_context=args.allow_missing_context, license=args.license, doi=args.doi)
    t = r["tables"]
    routes = r["patterns_by_route"]
    print(f"KANIT {args.kb_version} -> {out}: {t['model']} models, {routes['model']} candidates "
          f"and {routes['association']} patterns nominated by association, {t['unitig']} "
          f"unitigs; {r['grades_rechecked']} grades rechecked")


if __name__ == "__main__":
    main()
