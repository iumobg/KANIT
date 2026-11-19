"""Population-structure-corrected association with pyseer (protocol §8.5).

pyseer's linear mixed model (``pyseer --lmm``) tests the union of the CPSS
prefilter (§8.4) and the candidate patterns. The kinship matrix is K = G Gᵀ, the
number of sampled unitigs two genomes share (the shared-variant kinship of
pyseer's similarity_pyseer), over a systematic subsample of all unitigs of the
model, every ``kinship_every``-th one, so that it does not depend on the
candidates. Significance: Bonferroni, alpha / number of patterns tested. The
genomic inflation λ (median χ²₁ of the p-values over its null expectation) and
the QQ points are reported.

The tested patterns are chosen for their association (χ² prefilter, model
importance), so λ over them is inflated by design. λ is therefore also computed
over a background sample of the model's unitigs that lies between the kinship
sample (offset kinship_every / 2) and is tested with the same model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

PYSEER_COLUMNS = ("variant", "af", "filter-pvalue", "lrt-pvalue", "beta", "beta-std-err",
                  "variant_h2", "notes")


def unitig_samples(mm, every: int, background_max: int) -> tuple[np.ndarray, np.ndarray]:
    """(kinship, background): pattern ids of every ``every``-th unitig of the model
    (one per unitig, repeats kept), and the distinct patterns of the unitigs halfway
    between them, at most ``background_max`` evenly spaced."""
    m = mm.members().sort_values("unitig_index")["pattern_id"].to_numpy()
    kin = m[::every]
    bg = np.unique(m[every // 2::every])
    if bg.size > background_max:
        bg = bg[np.linspace(0, bg.size - 1, background_max).round().astype(int)]
    return kin, bg


def kinship(mm, pattern_ids, block: int = 4096) -> np.ndarray:
    """K = G Gᵀ over the given patterns (one column per id, repeats counted)."""
    ids = np.asarray(pattern_ids, dtype=np.int64)
    k = np.zeros((mm.n_genomes, mm.n_genomes), dtype=np.float64)
    for s in range(0, ids.size, block):
        g = mm.columns(ids[s:s + block]).astype(np.float32)
        k += (g @ g.T).astype(np.float64)
    return np.rint(k).astype(np.int64)


def sample_names(genome_ids) -> list[str]:
    """pyseer's sample names: the genome ids behind a letter. pyseer reads the phenotype
    and kinship tables with pandas' type inference, which turns numeric-looking ids into
    floats: BV-BRC's 550.2110 becomes 550.211 and no longer matches its column."""
    return [f"s{g}" for g in genome_ids]


def write_rtab(path, names, genome_ids, bits) -> None:
    """pyseer's --pres format: a header of sample names, one 0/1 row per variant.
    ``bits`` is variants × genomes."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.shape[1]
    line = np.full((bits.shape[0], 2 * n), ord("\t"), dtype=np.uint8)
    line[:, 0::2] = bits + ord("0")
    line[:, -1] = ord("\n")
    with open(path, "wb") as f:
        f.write(("pattern\t" + "\t".join(genome_ids) + "\n").encode())
        for name, row in zip(names, line, strict=True):
            f.write(name.encode() + b"\t" + row.tobytes())


def write_inputs(mm, out_dir, tested, *, every: int, background_max: int,
                 structure=None) -> dict:
    """phenotypes.tsv, tested.Rtab, background.Rtab and kinship.tsv for pyseer.

    ``structure`` is the model matrix the kinship and the background patterns come from,
    by default ``mm``. The budgeted arms of the unitig–k-mer comparison pass arm A's: their own matrix holds only
    the K chosen features, too few and too label-bound to describe population structure."""
    structure = mm if structure is None else structure
    if list(structure.genomes["Genome ID"].astype(str)) != list(mm.genomes["Genome ID"].astype(str)):
        raise ValueError("the structure matrix must hold the model's genomes in its order")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sample_names(mm.genomes["Genome ID"].astype(str))
    pd.DataFrame({"samples": ids, "resistant": mm.labels.astype(int)}).to_csv(
        out / "phenotypes.tsv", sep="\t", index=False)
    tested = np.asarray(sorted(set(int(p) for p in tested)), dtype=np.int64)
    pd.DataFrame({"pattern_id": tested}).to_csv(out / "tested_patterns.csv", index=False)
    kin_ids, bg = unitig_samples(structure, every, background_max)
    for name, pats, src in (("tested", tested, mm), ("background", bg, structure)):
        write_rtab(out / f"{name}.Rtab", [f"p{p}" for p in pats], ids, src.columns(pats).T)
    pd.DataFrame(kinship(structure, kin_ids), index=ids, columns=ids).to_csv(
        out / "kinship.tsv", sep="\t")
    return {"n_tested": int(tested.size), "n_background": int(bg.size),
            "n_kinship_unitigs": int(kin_ids.size)}


def command(out_dir, cpu: int) -> str:
    """The shell script that runs both LMMs in ``out_dir`` (the tools container runs it;
    $PYSEER overrides the pyseer executable). --min-af 0 --max-af 1: every pattern is
    tested, rare ones included."""
    lmm = ('"$PYSEER" --lmm --phenotypes phenotypes.tsv --similarity kinship.tsv '
           f"--min-af 0 --max-af 1 --cpu {cpu}")
    return "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail", 'PYSEER="${PYSEER:-pyseer}"',
        f'cd "{Path(out_dir).resolve()}"',
        '"$PYSEER" --version > pyseer_version.txt 2>&1',
        f"{lmm} --pres tested.Rtab > tested_assoc.tsv 2> tested_assoc.log",
        f"{lmm} --pres background.Rtab > background_assoc.tsv 2> background_assoc.log", ""])


def read_assoc(path) -> pd.DataFrame:
    """pyseer's LMM output with the pattern id of every variant."""
    d = pd.read_csv(path, sep="\t")
    missing = sorted({"variant", "af", "lrt-pvalue", "beta"} - set(d.columns))
    if missing:
        raise ValueError(f"{path}: pyseer columns missing: {missing}")
    if not d["variant"].astype(str).str.fullmatch(r"p\d+").all():
        raise ValueError(f"{path}: variant names must be p<pattern id>")
    d.insert(0, "pattern_id", d["variant"].str[1:].astype(int))
    d["lrt-pvalue"] = pd.to_numeric(d["lrt-pvalue"], errors="coerce")
    return d


def genomic_lambda(p) -> float:
    """Median χ²₁ statistic of the p-values over its null median."""
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if not p.size:
        return float("nan")
    return float(np.median(chi2.isf(p, 1)) / chi2.ppf(0.5, 1))


def qq_points(p) -> pd.DataFrame:
    p = np.sort(np.asarray(p, dtype=float)[np.isfinite(p)])
    expected = (np.arange(1, p.size + 1) - 0.5) / p.size
    return pd.DataFrame({"expected": -np.log10(expected), "observed": -np.log10(p)})


def results(tested: pd.DataFrame, candidates, alpha: float) -> tuple[pd.DataFrame, float]:
    """(layer, threshold): every candidate's test and whether it passes Bonferroni."""
    threshold = alpha / len(tested)
    cand = np.asarray(sorted(set(int(c) for c in candidates)), dtype=np.int64)
    by = tested.set_index("pattern_id")
    absent = sorted(set(cand.tolist()) - set(by.index.tolist()))
    if absent:
        raise ValueError(f"pyseer has no result for {len(absent)} candidate(s), e.g. {absent[:5]}")
    layer = by.loc[cand, ["af", "beta", "lrt-pvalue"]].rename(columns={"lrt-pvalue": "lrt_pvalue"})
    layer = layer.reset_index().assign(threshold=threshold)
    layer["passes"] = layer["lrt_pvalue"].lt(threshold).fillna(False).astype(bool)
    if "notes" in by.columns:
        layer["notes"] = by.loc[cand, "notes"].fillna("").to_numpy()
    return layer, threshold
