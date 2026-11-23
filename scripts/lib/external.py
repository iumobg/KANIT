"""Comparison with genotype-based prediction.

AMRFinderPlus and ResFinder (with PointFinder) predict a genome resistant to an
antibiotic when they report at least one qualifying determinant:

* AMRFinderPlus: Type AMR, Scope core, class or subclass among the antibiotic's
  registry keywords (lib.hypotheses.reference_calls). An antibiotic is not
  assessable when none of its keywords is a class or subclass of the tool's own
  reference catalogue.
* ResFinder: its predicted phenotype for the antibiotic; not assessable when the
  phenotype table does not report the antibiotic.
* The model: the mean of the lineage-aware out-of-fold probabilities of a genome
  over the repeats, resistant at ≥ the threshold.

RGI is not among them. It reports CARD models at the level of drug class, without a
drug-level phenotype rule, and counts intrinsic multidrug efflux genes and their
regulators: read as a predictor it calls almost every genome resistant. CARD,
through RGI, is the reference of the concordance criterion (lib.card_layer), so
comparing the knowledge base with RGI would also compare CARD with itself.

Metrics against the laboratory phenotype: sensitivity, specificity, balanced
accuracy, very major error rate (resistant predicted susceptible, over the
resistant genomes) and major error rate (susceptible predicted resistant, over
the susceptible genomes).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from lib import concordance
from lib.hypotheses import class_tokens, reference_calls

TOOLS = ("amrfinderplus", "resfinder", "model")
CALL_COLUMNS = ["genome_id", "element_symbol", "type", "subtype", "scope", "class", "subclass"]
_AFP_NAMES = {"element_symbol": ("Element symbol", "Gene symbol"),
              "type": ("Type", "Element type"), "subtype": ("Subtype", "Element subtype"),
              "scope": ("Scope",), "class": ("Class",), "subclass": ("Subclass",)}


def read_amrfinder(path, genome_id: str) -> pd.DataFrame:
    """One AMRFinderPlus report as calls (column names of versions 3 and 4)."""
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {"genome_id": [str(genome_id)] * len(d)}
    for col, names in _AFP_NAMES.items():
        name = next((n for n in names if n in d.columns), None)
        if name is None:
            raise ValueError(f"{path}: AMRFinderPlus column missing: {names[0]}")
        out[col] = d[name].tolist()
    return pd.DataFrame(out, columns=CALL_COLUMNS)


def catalog_tokens(path) -> set[str]:
    """Class and subclass tokens of AMRFinderPlus's database (amrfinder_catalog.tsv)."""
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    cols = {c.lower(): c for c in d.columns}
    if "class" not in cols or "subclass" not in cols:
        raise ValueError(f"{path}: class and subclass columns expected")
    toks: set[str] = set()
    for c in (cols["class"], cols["subclass"]):
        for v in d[c].unique():
            toks |= class_tokens(v)
    return toks


def _ab_tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", str(name).strip().lower()) if t)


def read_resfinder(path, antibiotics) -> dict[str, int]:
    """ResFinder's phenotype table -> {antibiotic: 1 resistant / 0 susceptible} for
    the antibiotics it reports (matched on the set of name tokens, so that
    'amoxicillin+clavulanic acid' is amoxicillin_clavulanic_acid)."""
    wanted = {_ab_tokens(a): a for a in antibiotics}
    calls = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and _ab_tokens(parts[0]) in wanted:
            calls[wanted[_ab_tokens(parts[0])]] = int(parts[2].strip().lower().startswith("resistant"))
    return calls


def resfinder_table(genome_dir: Path) -> Path | None:
    """The species-specific phenotype table, or the generic one."""
    hits = sorted(Path(genome_dir).glob("pheno_table_*.txt"))
    if hits:
        return hits[0]
    generic = Path(genome_dir) / "pheno_table.txt"
    return generic if generic.exists() else None


# ---- predictions ---------------------------------------------------------------------
def predict_amrfinder(calls: pd.DataFrame, genomes, keywords: set[str],
                      catalog: set[str]) -> pd.Series | None:
    """1/0 per genome, or None when the antibiotic is not assessable."""
    if not keywords & catalog:
        return None
    hit = set(reference_calls(calls, keywords)["genome_id"].astype(str))
    return pd.Series([int(g in hit) for g in genomes], index=list(genomes))


def predict_resfinder(calls: pd.DataFrame, genomes, antibiotic: str) -> pd.Series | None:
    """1/0 per genome from resfinder_calls (genome_id, antibiotic, resistant); None when
    ResFinder reports the antibiotic for no genome. Every genome must be reported."""
    d = calls[calls["antibiotic"] == antibiotic]
    if d.empty:
        return None
    by = dict(zip(d["genome_id"].astype(str), d["resistant"].astype(int), strict=True))
    missing = [g for g in genomes if g not in by]
    if missing:
        raise ValueError(f"ResFinder reports {antibiotic} for some genomes but not "
                         f"{len(missing)}, e.g. {missing[:3]}")
    return pd.Series([by[g] for g in genomes], index=list(genomes))


def predict_model(oof: pd.DataFrame, genomes, threshold: float) -> pd.Series:
    """Mean lineage-aware out-of-fold probability over the repeats, at the threshold."""
    d = oof[oof["arm"] == "lineage_aware"]
    p = d.groupby(d["genome_id"].astype(str))["p"].mean()
    missing = [g for g in genomes if g not in p.index]
    if missing:
        raise ValueError(f"no out-of-fold prediction for {len(missing)} genome(s)")
    return pd.Series((p.reindex(list(genomes)).to_numpy() >= threshold).astype(int),
                     index=list(genomes))


def metrics(truth: pd.Series, pred: pd.Series | None) -> dict:
    if pred is None:
        return {"assessable": 0}
    cm = concordance.confusion(truth.tolist(), pred.reindex(truth.index).tolist())
    err = concordance.fda_errors(cm)
    return {"assessable": 1, "n": cm["n"], "n_resistant": cm["TP"] + cm["FN"],
            "tp": cm["TP"], "fp": cm["FP"], "tn": cm["TN"], "fn": cm["FN"],
            "sensitivity": concordance.sensitivity(cm),
            "specificity": concordance.specificity(cm),
            "balanced_accuracy": concordance.balanced_accuracy(cm),
            "very_major_error_rate": err["very_major_error_rate"],
            "major_error_rate": err["major_error_rate"]}


def compare_model(model_id: str, truth: pd.Series, preds: dict[str, pd.Series | None]
                  ) -> pd.DataFrame:
    """One row per tool for one model."""
    cols = ["model_id", "tool", "assessable", "n", "n_resistant", "tp", "fp", "tn", "fn",
            "sensitivity", "specificity", "balanced_accuracy", "very_major_error_rate",
            "major_error_rate"]
    return pd.DataFrame([{"model_id": model_id, "tool": t, **metrics(truth, preds.get(t))}
                         for t in TOOLS], columns=cols)

