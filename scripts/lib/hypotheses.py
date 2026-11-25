"""The hypothesis tests: H1, H2, H3, H7 and the explanatory H6.

Every test reads the knowledge base (build_kb.py), only the candidates (route
``model``): the patterns nominated by association enter no hypothesis
test. H7 and H6 also read the AMRFinderPlus calls of the genomes, the reference that is
curated independently
of CARD (step 16 writes them: one row per call with genome_id, element_symbol,
type, subtype, scope, class, subclass, and the list of analysed genomes). The
criteria were set before the analysis.

AMRFinderPlus symbols are mapped to CARD AMR gene families through CARD's ARO
index: the normalised symbol (lower case, without "bla" and punctuation) equal to
an ARO name or CARD short name; otherwise the symbol without its allele number;
otherwise the one family shared by every ARO that extends the allele-less symbol
with an allele number; van cluster genes (vanH-A) through CARD's cluster terms.
Unmapped symbols are reported with their counts. The index must come from the CARD
release the knowledge base was built with (index_mismatches()).

The sensitivity analysis runs every test again without the
models the label permutation flags (sensitivity()).
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, mannwhitneyu

GRADE_RANK = {"confirmed": 4, "strong_novel": 3, "candidate": 2, "weak": 1, "none": 0}
MUTATION_MODELS = ("variant", "overexpression", "rrna_variant")
PERMUTATION_FLAG = "permutation_not_significant"     # label_permutation.flag


def _param(conn, name) -> float:
    return float(conn.execute("SELECT value FROM parameter WHERE name = ?", (name,)).fetchone()[0])


def _models(conn, exclude=()) -> pd.DataFrame:
    m = pd.read_sql_query(
        "SELECT m.model_id, m.organism_id, m.antibiotic_id, a.drug_class FROM model m "
        "JOIN antibiotic a USING (antibiotic_id) WHERE m.evaluable = 1 ORDER BY m.model_id", conn)
    return m[~m["model_id"].isin(exclude)].reset_index(drop=True)


# ---- H1 and H2 ---------------------------------------------------------------------
def h1(conn, exclude=()) -> tuple[pd.DataFrame, dict]:
    """Every model has a stable pattern; the Meinshausen–Bühlmann bound of each."""
    q, pi = _param(conn, "cpss.q"), _param(conn, "cpss.pi_threshold")
    t = pd.read_sql_query(
        "SELECT model_id, sum(in_prefilter) AS n_prefilter, "
        "sum(CASE WHEN passes = 1 THEN 1 ELSE 0 END) AS n_stable FROM cpss_result "
        "GROUP BY model_id", conn)
    t = _models(conn, exclude)[["model_id"]].merge(t, on="model_id", how="left").fillna(0)
    t["mb_bound"] = q ** 2 / ((2 * pi - 1) * t["n_prefilter"].where(t["n_prefilter"] > 0))
    t["has_stable"] = t["n_stable"] >= 1
    return t, {"criterion": "every model has a pattern with pi >= threshold",
               "n_models": int(len(t)), "n_with_stable": int(t["has_stable"].sum()),
               "supported": bool(len(t)) and bool(t["has_stable"].all())}


def h2(conn, exclude=()) -> tuple[pd.DataFrame, dict]:
    """Share of the stable patterns of every model with b = 1; median >= 0.40."""
    d = pd.read_sql_query(
        "SELECT c.model_id, g.mode, g.card_state FROM cpss_result c JOIN grade g "
        "ON g.model_id = c.model_id AND g.pattern_id = c.pattern_id "
        "JOIN candidate k ON k.model_id = c.model_id AND k.pattern_id = c.pattern_id "
        "WHERE c.passes = 1 AND k.route = 'model'", conn)
    rows = []
    for mid in _models(conn, exclude)["model_id"]:
        row = {"model_id": mid}
        for mode in ("allele_aware", "homolog_only"):
            s = d[(d["model_id"] == mid) & (d["mode"] == mode)]["card_state"]
            row["n_stable"] = len(s)
            row[f"share_b_{mode}"] = float((s == "b").mean()) if len(s) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows, columns=["model_id", "n_stable", "share_b_allele_aware",
                                    "share_b_homolog_only"])
    med = float(t["share_b_allele_aware"].median()) if t["share_b_allele_aware"].notna().any() \
        else np.nan
    return t, {"criterion": "median share of stable patterns with b = 1 >= 0.40",
               "n_defined": int(t["share_b_allele_aware"].notna().sum()),
               "median": med,
               "median_homolog_only": float(t["share_b_homolog_only"].median()),
               "supported": bool(med >= 0.40) if np.isfinite(med) else False}


# ---- H3 ------------------------------------------------------------------------------
def families(conn) -> dict[str, set[str]]:
    """CARD AMR gene families of every model's candidate unitigs (all hits, any class)."""
    out: dict[str, set[str]] = defaultdict(set)
    for mid, fam in conn.execute(
            "SELECT DISTINCT pm.model_id, a.gene_family FROM pattern_member pm "
            "JOIN candidate c ON c.model_id = pm.model_id AND c.pattern_id = pm.pattern_id "
            "JOIN card_hit h ON h.model_id = pm.model_id AND h.unitig_id = pm.unitig_id "
            "JOIN aro a ON a.aro_accession = h.aro_accession WHERE c.route = 'model'"):
        out[mid] |= {f.strip() for f in str(fam or "").split(";") if f.strip()}
    return out


def benjamini_yekutieli(p) -> np.ndarray:
    """BY-adjusted p-values, valid under arbitrary dependence; NaN passes through."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return out
    order = np.argsort(p[ok])
    adj = p[ok][order] * m * np.sum(1.0 / np.arange(1, m + 1)) / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def h3(conn, *, n_mc: int = 10_000, seed: int = 0, alpha: float = 0.05,
       exclude=()) -> tuple[pd.DataFrame, dict]:
    """Overlap of CARD gene families between the models of an organism; within-class
    pairs against cross-class pairs (one-sided Mann–Whitney U)."""
    fam, models = families(conn), _models(conn, exclude)
    rng = np.random.default_rng(seed)
    rows = []
    for org, ms in models.groupby("organism_id"):
        universe = set().union(*(fam.get(m, set()) for m in ms["model_id"]))
        n_univ = len(universe)
        for a, b in combinations(ms.itertuples(index=False), 2):
            fa, fb = fam.get(a.model_id, set()), fam.get(b.model_id, set())
            row = {"organism_id": org, "model_a": a.model_id, "model_b": b.model_id,
                   "same_class": bool(a.drug_class and a.drug_class == b.drug_class),
                   "n_a": len(fa), "n_b": len(fb), "n_shared": len(fa & fb),
                   "n_universe": n_univ}
            if fa and fb:
                k, big_k, n = len(fa & fb), len(fa), len(fb)
                draws = rng.hypergeometric(big_k, n_univ - big_k, n, size=n_mc)
                row.update(overlap=k / min(big_k, n), fold_enrichment=k * n_univ / (big_k * n),
                           fisher_p=float(hypergeom.sf(k - 1, n_univ, big_k, n)),
                           mc_p=(1 + int((draws >= k).sum())) / (n_mc + 1))
            rows.append(row)
    cols = ["organism_id", "model_a", "model_b", "same_class", "n_a", "n_b", "n_shared",
            "n_universe", "overlap", "fold_enrichment", "fisher_p", "mc_p"]
    t = pd.DataFrame(rows).reindex(columns=cols)
    t["q_by"] = benjamini_yekutieli(t["fisher_p"].to_numpy())
    within = t.loc[t["same_class"] & t["overlap"].notna(), "overlap"]
    cross = t.loc[~t["same_class"] & t["overlap"].notna(), "overlap"]
    if len(within) and len(cross):
        p = float(mannwhitneyu(within, cross, alternative="greater").pvalue)
    else:
        p = np.nan
    return t, {"criterion": "within-class overlap > cross-class overlap (one-sided MWU)",
               "n_within": int(len(within)), "n_cross": int(len(cross)),
               "median_within": float(within.median()) if len(within) else np.nan,
               "median_cross": float(cross.median()) if len(cross) else np.nan,
               "p": p, "supported": bool(np.isfinite(p) and p < alpha)}


# ---- AMRFinderPlus reference (H7, H6) -----------------------------------------------
def _norm(x: str) -> str:
    x = (x or "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", x[3:] if x.startswith("bla") else x)


def _family_set(field) -> set[str]:
    """The AMR gene families of a semicolon-separated cell (none for an empty one)."""
    return set() if pd.isna(field) else {f.strip() for f in str(field).split(";") if f.strip()}


def card_family_index(aro_index: pd.DataFrame) -> dict[str, set[str]]:
    """{normalised ARO name or CARD short name: CARD AMR gene families}."""
    idx: dict[str, set[str]] = defaultdict(set)
    for name, short, fams in aro_index[["ARO Name", "CARD Short Name", "AMR Gene Family"]].itertuples(
            index=False):
        fs = _family_set(fams)
        for key in (name, short):
            if isinstance(key, str) and key:
                idx[_norm(key)] |= fs
    return idx


def index_mismatches(conn, aro_index: pd.DataFrame) -> list[str]:
    """CARD terms of the knowledge base that the ARO index lacks or files under other
    AMR gene families.

    H7 compares the families the index gives AMRFinderPlus symbols with the families
    the knowledge base records for its CARD hits. An index from another CARD release
    names some families differently ("Erm 23S ribosomal RNA methyltransferase" and
    "Erm-like 23S rRNA methyltransferase"), which changes H7 without an error, so the
    two must agree on every term the knowledge base holds."""
    index: dict[str, set[str]] = defaultdict(set)
    for acc, fams in aro_index[["ARO Accession", "AMR Gene Family"]].itertuples(index=False):
        index[str(acc).strip()] |= _family_set(fams)
    return sorted(acc for acc, fams in conn.execute("SELECT aro_accession, gene_family FROM aro")
                  if acc not in index or index[acc] != _family_set(fams))


def map_symbol(symbol: str, idx: dict[str, set[str]]) -> set[str]:
    """CARD AMR gene families of an AMRFinderPlus element symbol (empty if unmapped)."""
    if idx.get(_norm(symbol)):
        return idx[_norm(symbol)]
    m = re.match(r"^van([A-Z])-([A-Z])$", symbol)
    if m and idx.get(_norm(f"van{m.group(1)}_in_van{m.group(2)}_cl")):
        return idx[_norm(f"van{m.group(1)}_in_van{m.group(2)}_cl")]
    stripped = re.sub(r"[-_]?\d+[A-Za-z]?$", "", symbol)
    if stripped != symbol and idx.get(_norm(stripped)):
        return idx[_norm(stripped)]
    pre = _norm(stripped)
    found = {frozenset(f) for k, f in idx.items()
             if k.startswith(pre) and k[len(pre):len(pre) + 1].isdigit() and f}
    return set(next(iter(found))) if len(found) == 1 else set()


def class_tokens(field) -> set[str]:
    """Upper-case tokens of an AMRFinderPlus Class or Subclass cell."""
    s = str(field or "")
    if s.upper() in ("", "NA", "NAN"):
        return set()
    return {t.strip().upper() for t in s.replace(",", "/").split("/") if t.strip()}


def reference_calls(calls: pd.DataFrame, keywords: set[str]) -> pd.DataFrame:
    """AMRFinderPlus calls that count for an antibiotic: Type AMR, Scope core, and a
    class or subclass token among the antibiotic's keywords."""
    matching = np.array([bool((class_tokens(c) | class_tokens(s)) & keywords)
                         for c, s in zip(calls["class"], calls["subclass"], strict=True)],
                        dtype=bool)
    keep = (calls["type"].astype(str).str.upper().eq("AMR").to_numpy()
            & calls["scope"].astype(str).str.lower().eq("core").to_numpy() & matching)
    out = calls[keep].copy()
    out["point"] = out["subtype"].astype(str).str.upper().str.startswith("POINT")
    return out


def carriers(calls: pd.DataFrame, idx: dict[str, set[str]]) -> tuple[dict, dict, dict]:
    """Determinant -> carrier genomes, for acquired families (FAM:), point-mutation
    genes (GENE:) and AMRFinderPlus elements (for H6); and unmapped symbol counts."""
    det: dict[str, set[str]] = defaultdict(set)
    element: dict[tuple[str, bool], set[str]] = defaultdict(set)
    unmapped: dict[str, int] = defaultdict(int)
    for g, sym, point in calls[["genome_id", "element_symbol", "point"]].itertuples(index=False):
        sym = str(sym)
        element[(sym, bool(point))].add(str(g))
        if point:
            det["GENE:" + sym.split("_")[0]].add(str(g))
            continue
        fams = map_symbol(sym, idx)
        if not fams:
            unmapped[sym] += 1
        for f in fams:
            det["FAM:" + f].add(str(g))
    return det, element, unmapped


def model_patterns(conn, mid: str) -> pd.DataFrame:
    """Candidate patterns with their best grade, CARD families and mutation genes."""
    grades = pd.read_sql_query(
        "SELECT g.pattern_id, g.grade, g.card_state FROM grade g "
        "JOIN candidate c USING (model_id, pattern_id) "
        "WHERE g.model_id = ? AND g.primary_mode = 1 AND c.route = 'model'", conn,
        params=(mid,))
    hits = pd.read_sql_query(
        "SELECT pm.pattern_id, a.gene_family, a.name, a.model_type, ca.state "
        "FROM pattern_member pm "
        "JOIN candidate c ON c.model_id = pm.model_id AND c.pattern_id = pm.pattern_id "
        "JOIN card_hit h ON h.model_id = pm.model_id AND h.unitig_id = pm.unitig_id "
        "JOIN aro a ON a.aro_accession = h.aro_accession "
        "JOIN card_annotation ca ON ca.model_id = pm.model_id AND ca.unitig_id = pm.unitig_id "
        "AND ca.mode = 'allele_aware' WHERE pm.model_id = ? AND c.route = 'model'", conn,
        params=(mid,))
    fam: dict[int, set[str]] = defaultdict(set)
    mut: dict[int, list[str]] = defaultdict(list)
    for pid, gf, name, mtype, state in hits.itertuples(index=False):
        fam[pid] |= {f.strip() for f in str(gf or "").split(";") if f.strip()}
        if mtype in MUTATION_MODELS and state == "b":
            mut[pid].append(str(name))
    grades["families"] = grades["pattern_id"].map(lambda p: fam.get(p, set()))
    grades["mutation_aros"] = grades["pattern_id"].map(lambda p: mut.get(p, []))
    grades["has_card_hit"] = grades["families"].map(bool)
    return grades


def matches(det: str, families: set[str], mutation_aros: list[str]) -> bool:
    """Whether a pattern maps to a determinant."""
    if det.startswith("FAM:"):
        return det[4:] in families
    gene = re.escape(det[5:])
    return any(re.search(rf"(?<![A-Za-z0-9]){gene}(?![A-Za-z0-9])", n, re.IGNORECASE)
               for n in mutation_aros)


def h7(conn, reference: dict[str, pd.DataFrame], idx: dict[str, set[str]], keywords: dict,
       *, thresholds=(0.10, 0.05, 0.20), alpha: float = 0.05,
       exclude=()) -> tuple[pd.DataFrame, dict, dict]:
    """Expected determinants of every model and whether they get the top grade.

    ``reference``: {organism: AMRFinderPlus calls of its analysed genomes}. Returns
    (determinant table, summary per threshold, unmapped symbol counts)."""
    rows: list[dict] = []
    pattern_rows: list[dict] = []
    unmapped_all: dict[str, int] = defaultdict(int)
    for m in _models(conn, exclude).itertuples(index=False):
        kw = keywords.get(m.antibiotic_id, set())
        if not kw or m.organism_id not in reference:
            continue
        mg = pd.read_sql_query("SELECT genome_id, resistant FROM model_genome WHERE model_id = ?",
                               conn, params=(m.model_id,))
        res = set(mg.loc[mg["resistant"] == 1, "genome_id"])
        sus = set(mg.loc[mg["resistant"] == 0, "genome_id"])
        calls = reference_calls(reference[m.organism_id], kw)
        calls = calls[calls["genome_id"].astype(str).isin(res | sus)]
        det, _, unmapped = carriers(calls, idx)
        for s, c in unmapped.items():
            unmapped_all[s] += c
        pats = model_patterns(conn, m.model_id)
        for d, gs in sorted(det.items()):
            pr, ps = len(gs & res) / len(res), len(gs & sus) / len(sus)
            hit = pats[[matches(d, f, a) for f, a in zip(pats["families"], pats["mutation_aros"],
                                                          strict=True)]]
            best = max(hit["grade"], key=lambda g: GRADE_RANK[g]) if len(hit) else None
            row = {"model_id": m.model_id, "determinant": d,
                   "kind": "point" if d.startswith("GENE:") else "acquired",
                   "prev_resistant": pr, "prev_susceptible": ps, "recovered": bool(len(hit)),
                   "best_grade": best, "n_patterns": int(len(hit))}
            for thr in thresholds:
                row[f"expected_{thr:g}"] = bool(pr >= thr and pr > ps)
            rows.append(row)
        for p in pats.itertuples(index=False):
            pattern_rows.append({"model_id": m.model_id, "pattern_id": p.pattern_id,
                                 "grade": p.grade, "has_card_hit": p.has_card_hit,
                                 "card_state": p.card_state,
                                 "determinants": [d for d in det if matches(
                                     d, p.families, p.mutation_aros)]})
    t = pd.DataFrame(rows)
    pats_all = pd.DataFrame(pattern_rows)
    summary: dict[str, dict] = {}
    for thr in thresholds:
        col = f"expected_{thr:g}"
        if t.empty:
            summary[f"{thr:g}"] = {"n_expected": 0, "supported_primary": False}
            continue
        exp = t[t[col]]
        rec = exp[exp["recovered"]]
        share = float((rec["best_grade"] == "confirmed").mean()) if len(rec) else np.nan
        expected_set = {(r.model_id, r.determinant) for r in exp.itertuples()}
        is_exp = pd.Series([any((mid, d) in expected_set for d in dets) for mid, dets in
                            zip(pats_all.get("model_id", []), pats_all.get("determinants", []),
                                strict=True)], index=pats_all.index, dtype=bool)
        a = pats_all[is_exp]
        b = pats_all[~is_exp & pats_all["has_card_hit"]] if len(pats_all) else pats_all
        table = [[int((a["grade"] == "confirmed").sum()), int((a["grade"] != "confirmed").sum())],
                 [int((b["grade"] == "confirmed").sum()), int((b["grade"] != "confirmed").sum())]]
        p = float(fisher_exact(table, alternative="greater")[1]) if len(a) and len(b) else np.nan
        # the same two groups split by CARD state: only b = 1 can be confirmed
        by_state = {st: [[int(((g["card_state"] == st) & (g["grade"] == "confirmed")).sum()),
                          int(((g["card_state"] == st) & (g["grade"] != "confirmed")).sum())]
                         for g in (a, b)]
                    for st in ("b", "card_hit_without_b")} if len(pats_all) else {}
        summary[f"{thr:g}"] = {
            "n_expected": int(len(exp)), "n_recovered": int(len(rec)),
            "share_confirmed": share, "supported_primary": bool(np.isfinite(share) and share >= 0.5),
            "secondary_table": table, "secondary_p": p, "secondary_table_by_state": by_state,
            "supported_secondary": bool(np.isfinite(p) and p < alpha)}
    return t, summary, dict(unmapped_all)


def h6(conn, reference: dict[str, pd.DataFrame], keywords: dict, h2_table: pd.DataFrame,
       exclude=()) -> tuple[pd.DataFrame, dict]:
    """Acquired, mutational or mixed phenotype; H2 share acquired > mutational."""
    share = dict(zip(h2_table["model_id"], h2_table["share_b_allele_aware"], strict=True))
    rows = []
    for m in _models(conn, exclude).itertuples(index=False):
        kw = keywords.get(m.antibiotic_id, set())
        if not kw or m.organism_id not in reference:
            continue
        mg = pd.read_sql_query("SELECT genome_id, resistant FROM model_genome WHERE model_id = ?",
                               conn, params=(m.model_id,))
        res = set(mg.loc[mg["resistant"] == 1, "genome_id"])
        sus = set(mg.loc[mg["resistant"] == 0, "genome_id"])
        calls = reference_calls(reference[m.organism_id], kw)
        calls = calls[calls["genome_id"].astype(str).isin(res | sus)]
        _, element, _ = carriers(calls, {})
        carried: dict[bool, set[str]] = {False: set(), True: set()}
        for (_, point), gs in element.items():
            pr, ps = len(gs & res) / len(res), len(gs & sus) / len(sus)
            if pr >= 0.10 and pr - ps >= 0.10:
                carried[point] |= gs & res
        a_share, m_share = len(carried[False]) / len(res), len(carried[True]) / len(res)
        cls = ("acquired" if a_share >= 0.5 and a_share >= m_share else
               "mutational" if m_share >= 0.5 and m_share > a_share else "mixed")
        rows.append({"model_id": m.model_id, "A": a_share, "M": m_share, "class": cls,
                     "h2_share": share.get(m.model_id, np.nan)})
    t = pd.DataFrame(rows, columns=["model_id", "A", "M", "class", "h2_share"])
    acq = t.loc[(t["class"] == "acquired") & t["h2_share"].notna(), "h2_share"]
    mut = t.loc[(t["class"] == "mutational") & t["h2_share"].notna(), "h2_share"]
    p = float(mannwhitneyu(acq, mut, alternative="greater").pvalue) if len(acq) and len(mut) \
        else np.nan
    return t, {"analysis": "explanatory (not a hypothesis)", "n_acquired": int(len(acq)),
               "n_mutational": int(len(mut)),
               "n_mixed": int((t["class"] == "mixed").sum()), "p_acquired_gt_mutational": p}


def run(conn: sqlite3.Connection, reference, idx, keywords, *, n_mc: int, seed: int,
        exclude=()) -> dict:
    t1, s1 = h1(conn, exclude)
    t2, s2 = h2(conn, exclude)
    t3, s3 = h3(conn, n_mc=n_mc, seed=seed, exclude=exclude)
    t7, s7, unmapped = h7(conn, reference, idx, keywords, exclude=exclude)
    t6, s6 = h6(conn, reference, keywords, t2, exclude)
    return {"tables": {"h1_models": t1, "h2_models": t2, "h3_pairs": t3,
                       "h7_determinants": t7, "h6_models": t6,
                       "h7_unmapped_symbols": pd.DataFrame(sorted(unmapped.items()),
                                                           columns=["symbol", "n_calls"])},
            "summary": {"H1": s1, "H2": s2, "H3": s3, "H7": s7, "H6": s6}}


# ---- sensitivity analysis without the flagged models ---------------------------------
def flagged_models(conn) -> list[str] | None:
    """Evaluable models the label permutation flags (q >= alpha); None while one of
    them has no test, since the q values are corrected across all of them."""
    rows = conn.execute(
        "SELECT m.model_id, lp.flag FROM model m LEFT JOIN label_permutation lp "
        "USING (model_id) WHERE m.evaluable = 1 ORDER BY m.model_id").fetchall()
    if not rows or any(flag is None for _, flag in rows):
        return None
    return [mid for mid, flag in rows if flag == PERMUTATION_FLAG]


def sensitivity(conn: sqlite3.Connection, reference, idx, keywords, *, n_mc: int,
                seed: int) -> dict:
    """The hypotheses again without the flagged models, reported
    next to the primary results. The per-model rows are those of the primary tables; only
    the summaries change, and H3 through the families of each organism's models."""
    excluded = flagged_models(conn)
    if excluded is None:
        return {"status": "not computed: an evaluable model has no label permutation test",
                "excluded_models": []}
    if not excluded:
        return {"status": "no model flagged: the primary analysis", "excluded_models": []}
    res = run(conn, reference, idx, keywords, n_mc=n_mc, seed=seed, exclude=excluded)
    return {"status": "computed", "excluded_models": excluded, **res["summary"]}
