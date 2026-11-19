#!/usr/bin/env python3
"""Figures, model reports and the numbers file (plan F4.2), from the steps' CSV tables.

  reports.py --entry main --organisms ecoli,kpneumoniae

At the end of every entry it draws the QC figure of every step whose tables exist
(lib/figures.py; Appendix C), writes for every panel model a self-contained HTML
report (data, cross-validation, label permutation, evidence layers, grades, the best
candidates, apart from them the patterns nominated by association (protocol §14 item 6),
the comparison with genotype-based tools and the model's figures), and
rewrites tez_sayilari.csv: every number the thesis and the article quote, with the
table and column it comes from. A figure whose table is missing is skipped and named
in reports_summary.json (completeness.py reports the missing table).

Outputs: paths_organism.reports_dir (global and organism figures, the numbers file)
and model_reports_dir (a model's figures and report.html).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import card_tiers, concordance, contract, panel, registry  # noqa: E402
from lib import figures as F  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402
from lib.run_metadata import git_commit_hash  # noqa: E402

GRADE_ORDER = {g: i for i, g in enumerate(F.GRADES)}


def _gram_stain(organism_id):
    """The organism's Gram stain from the registry, or None when it has none."""
    try:
        return registry.get_organism(organism_id).get("gram_stain")
    except KeyError:
        return None


class _Rounded(float):
    """A rounded float that remembers the value it was rounded from (the numbers file's
    exact column)."""

    raw: float


def _four_significant(x):
    """Round to four significant digits. A p-value can be far below the fourth
    decimal, where the usual rounding of the numbers file would report it as 0."""
    if x is None or x == 0:
        return x
    r = _Rounded(round(x, -int(math.floor(math.log10(abs(x)))) + 3))
    r.raw = float(x)
    return r


class Tables:
    """The contract's tables of one run, read when asked for (None when absent)."""

    def __init__(self, config: dict):
        self.config = config
        self.skipped: list[str] = []

    def get(self, tid: str, organism=None, antibiotic=None, **read_csv) -> pd.DataFrame | None:
        try:
            p = contract.table_path(tid, self.config, organism, antibiotic)
        except KeyError:
            return None
        if not p.exists():
            return None
        # Genome IDs stay text: read as numbers, "1352.20830" becomes 1352.2083 and merges
        # with genome 1352.2083, so counts of distinct genomes come out short.
        read_csv["dtype"] = {**GENOME_ID_DTYPE, **read_csv.get("dtype", {})}
        return pd.read_csv(p, keep_default_na=True, low_memory=False, **read_csv)

    def json(self, location, file, organism=None, antibiotic=None) -> dict | None:
        p = contract.output_path(location, file, self.config, organism, antibiotic)
        return json.loads(p.read_text()) if p.exists() else None


GENOME_ID_DTYPE = {"genome_id": str, "Genome ID": str}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    F.plt.close(fig)
    return path


# ---- figures -----------------------------------------------------------------------
def organism_figures(t: Tables, org: str, out: Path) -> list[Path]:
    done = []
    long = t.get("phenotypes_long", org)
    if long is not None:
        done.append(_save(F.phenotype_counts(long, f"{org} — phenotypes"), out / "snapshot.png"))
    qc = t.get("genome_qc", org)
    if qc is not None:
        summ = t.json("genome_qc_dir", "02d_genome_qc_summary_{organism}.json", org) or {}
        done.append(_save(F.genome_qc(qc, summ.get("thresholds", {}), f"{org} — genome QC"),
                          out / "genome_qc.png"))
    cl = t.get("lineage_clusters", org)
    if cl is not None:
        done.append(_save(F.lineage_sizes(cl, f"{org} — lineages"), out / "lineages.png"))
    hits = t.get("rgi_hits", org)
    if hits is not None:
        done.append(_save(F.rgi_hits(hits, f"{org} — RGI"), out / "rgi.png"))
    comp = t.get("external_comparison", org)
    if comp is not None:
        done.append(_save(F.external_agreement(comp, f"{org} — genotype-based prediction"),
                          out / "external.png"))
    ctx = t.get("unitig_context", org)
    if ctx is not None:
        done.append(_save(F.context_plasmid(ctx, f"{org} — NCBI context"), out / "context.png"))
    return done


def model_figures(t: Tables, org: str, ab: str, out: Path) -> list[tuple[str, Path]]:
    title = f"{org} / {ab}"
    done: list[tuple[str, Path]] = []

    def add(name, fig):
        done.append((name, _save(fig, out / f"{name}.png")))

    pats, mg = t.get("model_patterns", org, ab), t.get("model_genomes", org, ab)
    if pats is not None and mg is not None:
        add("matrix", F.matrix_patterns(pats, len(mg), title))
    oof = t.get("oof_predictions", org, ab)
    rel, fm = t.get("reliability", org, ab), t.get("fold_metrics", org, ab)
    comp = t.get("fold_composition", org, ab)
    if all(x is not None for x in (oof, rel, fm, comp)):
        add("cross_validation", F.cv_quality(oof, rel, fm, comp, title))
    null, lp = t.get("label_permutation_null", org, ab), t.get("label_permutation_models")
    if null is not None:
        row = (lp[(lp["organism"] == org) & (lp["antibiotic"] == ab)]
               if lp is not None else pd.DataFrame())
        obs = float(row["auc_observed"].iloc[0]) if len(row) else None
        add("label_permutation", F.permutation_null(null, obs, title))
    cpss = t.get("cpss_patterns", org, ab)
    if cpss is not None:
        add("cpss", F.cpss_frequency(cpss, float(t.config["cpss"]["pi_threshold"]), title))
    prev = t.get("prevalence_layer", org, ab)
    if prev is not None:
        add("prevalence", F.prevalence_volcano(prev, title))
    mda = t.get("mda_layer", org, ab)
    if mda is not None:
        add("mda", F.mda_scatter(mda, title))
    qq = t.get("pyseer_qq", org, ab)
    if qq is not None:
        s = t.json("pyseer_dir", "pyseer_summary.json", org, ab) or {}
        lam = {"tested": s.get("lambda_tested"), "background": s.get("lambda_background")}
        add("pyseer", F.pyseer_qq(qq, title, lam))
    card = t.get("card_patterns", org, ab)
    if card is not None:
        add("card", F.card_states(card, title))
    grades = t.get("grades_patterns", org, ab)
    if grades is not None:
        add("grading", F.evidence_funnel(grades, title))
    return done


def global_figures(t: Tables, out: Path) -> list[Path]:
    done = []
    dec = t.get("panel_decisions")
    if dec is not None:
        done.append(_save(F.panel_minority(dec, int(t.config["panel"]["min_minority"]), "Panel"),
                          out / "panel.png"))
    lp = t.get("label_permutation_models")
    if lp is not None:
        done.append(_save(F.permutation_models(lp, "Label permutation"),
                          out / "label_permutation.png"))
    h3 = t.get("h3_pairs")
    if h3 is not None:
        for org in sorted(h3["organism_id"].unique()):
            done.append(_save(F.overlap_heatmap(h3, org), out / org / "candidate_overlap.png"))
    return done


# ---- model report ------------------------------------------------------------------
CSS = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;
color:#222}h1{margin-bottom:0}.meta{color:#666}table{border-collapse:collapse;margin:.5em 0 1.5em;
font-size:13px}th,td{border:1px solid #ddd;padding:3px 8px;text-align:right}th{background:#f4f4f4}
td:first-child,th:first-child{text-align:left}img{max-width:100%;margin:.5em 0 1.5em}"""


def _table(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p class='meta'>not available</p>"
    return df.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0, na_rep="")


def model_report(t: Tables, org: str, ab: str, figs: list[tuple[str, Path]], out: Path) -> Path:
    mg = t.get("model_genomes", org, ab)
    data = None
    if mg is not None:
        data = pd.DataFrame([{"genomes": len(mg), "resistant": int((mg["label"] == 1).sum()),
                              "susceptible": int((mg["label"] == 0).sum()),
                              "lineages": mg["lineage"].nunique()}])
    rm = t.get("repeat_metrics", org, ab)
    cv = None
    if rm is not None:
        cols = [c for c in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "brier") if c in rm]
        g = rm.groupby("arm")[cols]
        cv = (g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).reset_index()
    lp = t.get("label_permutation_models")
    lp_row = lp[(lp["organism"] == org) & (lp["antibiotic"] == ab)] if lp is not None else None
    grades = t.get("grades_patterns", org, ab)
    layers = counts = best = None
    if grades is not None:
        layers = pd.DataFrame([{k: int(F._bool(grades[k]).sum())
                                for k in ("prevalence", "mda", "cpss", "pyseer")}])
        counts = pd.DataFrame({m: grades[f"{m}_grade"].value_counts().reindex(F.GRADES,
                                                                            fill_value=0)
                               for m in ("allele_aware", "homolog_only")}).reset_index(
            names="grade")
        best = (grades.assign(_o=grades["grade"].map(GRADE_ORDER))
                .sort_values(["_o", "n_layers"], ascending=[True, False]).head(25)
                [["pattern_id", "n_members", "layers_passed", "allele_aware_card_state",
                  "allele_aware_grade", "homolog_only_grade"]])
    assoc = t.get("association_grades_patterns", org, ab)
    ranks = t.get("association_patterns", org, ab)
    assoc_counts = assoc_best = None
    if assoc is not None and ranks is not None:
        assoc_counts = pd.DataFrame({m: assoc[f"{m}_grade"].value_counts().reindex(
            F.GRADES, fill_value=0) for m in ("allele_aware", "homolog_only")}).reset_index(
            names="grade")
        assoc_best = (assoc.merge(ranks, on="pattern_id", how="left")
                      .assign(_o=lambda d: d["grade"].map(GRADE_ORDER))
                      .sort_values(["_o", "rank"]).head(25)
                      [["pattern_id", "rank", "lrt_pvalue", "n_members", "layers_passed",
                        "allele_aware_card_state", "allele_aware_grade", "homolog_only_grade"]])
    comp = t.get("external_comparison", org)
    ext = comp[comp["model_id"] == f"{org}__{ab}"] if comp is not None else None
    parts = [f"<h1>{html.escape(org)} / {html.escape(ab)}</h1>",
             f"<p class='meta'>KANIT v1.0 · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
             f"commit {html.escape(str(git_commit_hash() or 'unknown'))}</p>",
             "<h2>Data</h2>", _table(data),
             "<h2>Cross-validation (mean ± SD over repeats)</h2>", _table(cv),
             "<h2>Label permutation</h2>", _table(lp_row),
             "<h2>Evidence layers that fire</h2>", _table(layers),
             "<h2>Grades</h2>", _table(counts),
             "<h2>Best candidates</h2>", _table(best),
             "<h2>Patterns nominated by association (secondary analysis)</h2>",
             "<p class='meta'>pyseer-significant patterns that are not candidates (protocol "
             "§14 item 6). They pass the pyseer layer by the way they are chosen, so their "
             "grades are not comparable with those of the candidates.</p>",
             _table(assoc_counts), _table(assoc_best),
             "<h2>Genotype-based prediction</h2>", _table(ext), "<h2>Figures</h2>"]
    for name, path in figs:
        b64 = base64.b64encode(path.read_bytes()).decode()
        parts.append(f"<h3>{html.escape(name)}</h3><img alt='{name}' "
                     f"src='data:image/png;base64,{b64}'>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>"
            f"{html.escape(org)} / {html.escape(ab)}</title><style>{CSS}</style></head><body>"
            + "\n".join(parts) + "</body></html>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(page, encoding="utf-8")
    return out / "report.html"


# ---- numbers ---------------------------------------------------------------------------
# Every aggregate the evidence sections of the grading article assert. They were written as
# literals because no key produced them; a number a reader cannot regenerate is not a result.
# `ev` is one row per candidate pattern of every model: the grade tables joined to the
# candidate tables, plus the model id, its organism and its drug class.
def _evidence_numbers(ev: pd.DataFrame, put) -> None:
    layers = ("prevalence", "mda", "cpss", "pyseer")
    for layer in layers:
        ev[layer] = F._bool(ev[layer])
    ev["n_layers_passed"] = ev[list(layers)].sum(axis=1)
    n = len(ev)

    # the layers, one by one and together
    for layer in layers:
        k = int(ev[layer].sum())
        put(f"all.n_pass_{layer}", k, "grades_patterns", layer,
            f"Candidates passing the {layer} layer")
        put(f"all.share_pass_{layer}", k / n, "grades_patterns", layer,
            f"Share (0-1) of the candidates passing the {layer} layer")
    for k in range(len(layers) + 1):
        put(f"all.n_layers_{k}", int((ev["n_layers_passed"] == k).sum()), "grades_patterns",
            "passes", f"Candidates passing exactly {k} statistical layer(s)")
    combos = ev.groupby([ev[l] for l in layers]).size().sort_values(ascending=False)
    for rank, (flags, size) in enumerate(combos.items(), start=1):
        passed = [l for l, f in zip(layers, flags, strict=True) if f]
        if rank > 3 or not passed:
            continue
        put(f"all.n_layer_combo_{'_and_'.join(passed)}", int(size), "grades_patterns", "passes",
            f"Candidates whose passing layers are exactly {' + '.join(passed)}")

    # the two grading modes: which patterns move, and where to
    moved = ev[ev["allele_aware_grade"] != ev["homolog_only_grade"]]
    put("all.n_grade_differs_between_modes", len(moved), "grades_patterns", "homolog_only_grade",
        "Candidates graded differently in the allele-aware and homolog-only modes")
    put("all.share_grade_differs_between_modes", len(moved) / n, "grades_patterns",
        "homolog_only_grade", "Share (0-1) of the candidates whose grade differs between modes")
    for (a, b), size in moved.groupby(["allele_aware_grade", "homolog_only_grade"]).size().items():
        put(f"all.n_mode_change_{a}_to_{b}", int(size), "grades_patterns", "homolog_only_grade",
            f"Candidates graded {a} allele-aware and {b} homolog-only")

    # what the gain ranking alone reports (the top-gain patterns of every model)
    top = ev[ev["gain_rank"].notna()]
    if len(top):
        nt = len(top)
        put("all.n_top_gain_patterns", nt, "candidates", "gain_rank",
            "Patterns in the top total gain of a final model, summed over models")
        states = {"b": "b", "card_hit_without_b": "card_hit_without_b", "no_card_hit": "no_card_hit"}
        for name, state in states.items():
            k = int((top["allele_aware_card_state"] == state).sum())
            put(f"all.top_gain_n_{name}", k, "grades_patterns", "allele_aware_card_state",
                f"Top-gain patterns whose CARD state is {state}")
            put(f"all.top_gain_share_{name}", k / nt, "grades_patterns",
                "allele_aware_card_state", f"Share (0-1) of the top-gain patterns with {state}")
        first = top.loc[top.groupby("model")["gain_rank"].idxmin()]
        put("all.n_models_top_feature_b",
            int((first["allele_aware_card_state"] == "b").sum()), "grades_patterns",
            "allele_aware_card_state", "Models whose single highest-gain pattern has b = 1")
        hit = top[top["allele_aware_card_state"] == "card_hit_without_b"]
        for reason, size in hit["allele_aware_card_reasons"].value_counts().items():
            put(f"all.top_gain_n_reason_{reason}", int(size), "grades_patterns",
                "allele_aware_card_reasons",
                f"Top-gain patterns denied b = 1 because of {reason}")
            put(f"all.top_gain_share_reason_{reason}", int(size) / len(hit), "grades_patterns",
                "allele_aware_card_reasons",
                f"Share (0-1) of those patterns denied b = 1 because of {reason}")
        blank = top[top["allele_aware_card_state"] == "no_card_hit"]
        failed = blank[~blank["pyseer"]]
        put("all.top_gain_no_hit_n_pyseer_fail", len(failed), "grades_patterns", "pyseer",
            "Top-gain patterns without a CARD hit that do not pass the mixed model")
        if len(blank):
            put("all.top_gain_no_hit_share_pyseer_fail", len(failed) / len(blank),
                "grades_patterns", "pyseer",
                "Share (0-1) of the top-gain patterns without a CARD hit failing the mixed model")
        put("all.top_gain_no_hit_pyseer_fail_n_prevalence_pass",
            int(failed["prevalence"].sum()), "grades_patterns", "prevalence",
            "Of those, the ones that still pass the prevalence difference")
        for g in F.GRADES:
            k = int((top["grade"] == g).sum())
            put(f"all.top_gain_share_{g}", k / nt, "grades_patterns", "grade",
                f"Share (0-1) of the top-gain patterns graded {g}")
            put(f"all.n_{g}_in_top_gain", k, "grades_patterns", "grade",
                f"Patterns graded {g} that a model ranks in its top total gain")
            put(f"all.n_{g}_outside_top_gain", int((ev["grade"] == g).sum()) - k,
                "grades_patterns", "grade",
                f"Patterns graded {g} reaching the candidate set only by stability selection")

    # the grade a candidate reaches without a CARD hit, with one, or with b = 1
    cand = ev[ev["grade"] == "candidate"]
    if len(cand):
        for name, state in (("no_card_hit", "no_card_hit"),
                            ("card_hit_without_b", "card_hit_without_b"), ("b", "b")):
            put(f"all.n_candidate_grade_{name}",
                int((cand["allele_aware_card_state"] == state).sum()), "grades_patterns",
                "allele_aware_card_state", f"Candidate-graded patterns whose CARD state is {state}")
        hit = cand[cand["allele_aware_card_state"] == "card_hit_without_b"]
        for reason, size in hit["allele_aware_card_reasons"].value_counts().items():
            put(f"all.n_candidate_grade_reason_{reason}", int(size), "grades_patterns",
                "allele_aware_card_reasons", f"Candidate-graded patterns denied b = 1 by {reason}")

    novel = ev[ev["grade"] == "strong_novel"]
    if len(novel):
        put("all.n_models_with_strong_novel", int(novel["model"].nunique()), "grades_patterns",
            "grade", "Models contributing at least one strong_novel pattern")
        for k in (3, 4):
            put(f"all.n_strong_novel_layers_{k}", int((novel["n_layers_passed"] == k).sum()),
                "grades_patterns", "passes", f"strong_novel patterns passing {k} layers")

    # direction: the sign of the prevalence difference; the grade does not depend on it
    if "direction" in ev:
        for d, name, where in (("R", "r", "resistant"), ("S", "s", "susceptible")):
            put(f"all.n_direction_{name}", int((ev["direction"] == d).sum()), "prevalence_layer",
                "direction", f"Candidates more frequent in the {where} genomes")
            put(f"all.n_strong_novel_direction_{name}",
                int(((ev["grade"] == "strong_novel") & (ev["direction"] == d)).sum()),
                "prevalence_layer", "direction",
                f"strong_novel patterns more frequent in the {where} genomes")

    # H2: the share of every model's stable patterns that reach b = 1
    stable = ev[F._bool(ev["cpss_stable"])] if "cpss_stable" in ev else ev.iloc[:0]
    if len(stable):
        put("all.n_stable_patterns", len(stable), "candidates", "cpss_stable",
            "Patterns the stability selection calls stable, summed over models")
        k = int((stable["allele_aware_card_state"] == "b").sum())
        put("all.n_stable_b", k, "grades_patterns", "allele_aware_card_state",
            "Of those, the ones matching a curated determinant of the modelled drug class")
        put("all.stable_b_share", k / len(stable), "grades_patterns", "allele_aware_card_state",
            "Share (0-1) of the stable patterns with b = 1, pooled over models")
        for mode in ("allele_aware", "homolog_only"):
            share = (stable.assign(hit=stable[f"{mode}_card_state"] == "b")
                     .groupby("model")["hit"].mean())
            put(f"all.stable_b_median_share.{mode}", float(share.median()), "grades_patterns",
                f"{mode}_card_state",
                f"Median over models of the share of stable patterns with b = 1, {mode} mode")
            if mode == "allele_aware":
                put("all.stable_b_share_q1", float(share.quantile(0.25)), "grades_patterns",
                    "allele_aware_card_state", "First quartile of that share over models")
                put("all.stable_b_share_q3", float(share.quantile(0.75)), "grades_patterns",
                    "allele_aware_card_state", "Third quartile of that share over models")
                put("all.n_models_with_stable_patterns", int(share.size), "candidates",
                    "cpss_stable", "Models with at least one stable pattern")
                put("all.n_models_no_stable_b", int((share == 0).sum()), "grades_patterns",
                    "allele_aware_card_state",
                    "Models where no stable pattern matches a curated determinant")

    # grade shares per organism and per drug class, which 3.3 reads off the figure
    for col, label in (("organism", "organism"), ("drug_class", "drug class")):
        if col not in ev:
            continue
        for key, part in ev.groupby(col):
            slug = str(key).replace(" ", "_").replace("/", "_")
            put(f"{slug}.n_candidates_in_group", len(part), "grades_patterns", "pattern_id",
                f"Candidate patterns of this {label}")
            for g in F.GRADES:
                put(f"{slug}.n_{g}", int((part["grade"] == g).sum()), "grades_patterns", "grade",
                    f"Candidate patterns of this {label} graded {g}")
                put(f"{slug}.share_{g}", float((part["grade"] == g).mean()), "grades_patterns",
                    "grade", f"Share (0-1) of this {label}'s candidates graded {g}")


# hypotheses.json (protocol §10) -> keys. A key names what its test measures, not the
# test's number: the protocol's H3 and H7 are the article's H4 and H3, so a number in the
# key would be right for one text and wrong for the other.
# (summary, field, key, table, column, description)
HYPOTHESIS_KEYS = [
    ("H1", "n_models", "stable.n_models", "h1_models", "model_id", "H1: models tested"),
    ("H1", "n_with_stable", "stable.n_with_stable", "h1_models", "has_stable",
     "H1: models with at least one stable pattern"),
    ("H1", "supported", "stable.supported", "h1_models", "has_stable",
     "H1: 1 if every model has a stable pattern, else 0"),
    ("H2", "n_defined", "curated.n_models", "h2_models", "share_b_allele_aware",
     "H2: models with stable patterns, the ones the median is taken over"),
    ("H2", "median", "curated.median_share", "h2_models", "share_b_allele_aware",
     "H2: median over models of the share of stable patterns with b = 1"),
    ("H2", "median_homolog_only", "curated.median_share_homolog_only", "h2_models",
     "share_b_homolog_only", "H2: that median in the homolog-only mode"),
    ("H2", "supported", "curated.supported", "h2_models", "share_b_allele_aware",
     "H2: 1 if the median is at least 0.40, else 0"),
    ("H3", "n_within", "class_overlap.n_within", "h3_pairs", "same_class",
     "Protocol H3: within-class model pairs with a defined overlap"),
    ("H3", "n_cross", "class_overlap.n_cross", "h3_pairs", "same_class",
     "Protocol H3: cross-class model pairs with a defined overlap"),
    ("H3", "median_within", "class_overlap.median_within", "h3_pairs", "overlap",
     "Protocol H3: median overlap coefficient of the within-class pairs"),
    ("H3", "median_cross", "class_overlap.median_cross", "h3_pairs", "overlap",
     "Protocol H3: median overlap coefficient of the cross-class pairs"),
    ("H3", "p", "class_overlap.p", "h3_pairs", "overlap",
     "Protocol H3: one-sided Mann-Whitney U p, within-class above cross-class"),
    ("H3", "supported", "class_overlap.supported", "h3_pairs", "overlap",
     "Protocol H3: 1 if p < 0.05, else 0"),
    ("H6", "n_acquired", "acquired_vs_mutational.n_acquired", "h6_models", "class",
     "H6 (explanatory): acquired phenotypes with an H2 share"),
    ("H6", "n_mutational", "acquired_vs_mutational.n_mutational", "h6_models", "class",
     "H6 (explanatory): mutational phenotypes with an H2 share"),
    ("H6", "n_mixed", "acquired_vs_mutational.n_mixed", "h6_models", "class",
     "H6 (explanatory): mixed phenotypes"),
    ("H6", "p_acquired_gt_mutational", "acquired_vs_mutational.p", "h6_models", "h2_share",
     "H6 (explanatory): one-sided Mann-Whitney U p, acquired above mutational"),
]
# protocol H7 at each threshold of expected prevalence among the resistant genomes
EXPECTED_THRESHOLDS = {"0.1": "thr10", "0.05": "thr5", "0.2": "thr20"}
EXPECTED_KEYS = [
    ("n_expected", "n_expected", "expected determinants"),
    ("n_recovered", "n_recovered", "expected determinants a candidate maps to"),
    ("share_confirmed", "share_confirmed", "share (0-1) of those whose best grade is confirmed"),
    ("supported_primary", "supported_primary", "1 if that share is at least 0.5, else 0"),
    ("secondary_p", "secondary_p",
     "one-sided Fisher p, confirmed more often among candidates of an expected determinant"),
    ("supported_secondary", "supported_secondary", "1 if that p is below 0.05, else 0"),
]
SECONDARY_CELLS = [((0, 0), "expected_confirmed", "candidates of an expected determinant, confirmed"),
                   ((0, 1), "expected_not_confirmed",
                    "candidates of an expected determinant, not confirmed"),
                   ((1, 0), "other_confirmed",
                    "candidates with a CARD hit to another family, confirmed"),
                   ((1, 1), "other_not_confirmed",
                    "candidates with a CARD hit to another family, not confirmed")]


def _hypothesis_numbers(summary: dict, put, pre: str, label: str = "") -> None:
    """The verdicts and statistics of one set of hypothesis summaries under pre."""
    def val(x):
        return int(x) if isinstance(x, bool) else x

    for test, field, key, table, column, text in HYPOTHESIS_KEYS:
        x = val((summary.get(test) or {}).get(field))
        if field.startswith("p"):
            put(f"{pre}.{key}", _four_significant(x), table, column, label + text, digits=None)
        else:
            put(f"{pre}.{key}", x, table, column, label + text)
    for thr, tag in EXPECTED_THRESHOLDS.items():
        s = (summary.get("H7") or {}).get(thr) or {}
        for field, key, text in EXPECTED_KEYS:
            x = val(s.get(field))
            p_value = field == "secondary_p"
            put(f"{pre}.expected.{key}.{tag}", _four_significant(x) if p_value else x,
                "h7_determinants", "best_grade", f"{label}Protocol H7, threshold {thr}: {text}",
                digits=None if p_value else 6 if field == "share_confirmed" else "auto")
        table = s.get("secondary_table")
        if table:
            for (i, j), key, text in SECONDARY_CELLS:
                put(f"{pre}.expected.secondary_{key}.{tag}", int(table[i][j]), "h7_determinants",
                    "best_grade", f"{label}Protocol H7, threshold {thr}: {text}")


def _flagged_models(lp: pd.DataFrame, models: list[str], alpha: float) -> list[str] | None:
    """Models whose label permutation q is at or above α (§8.6); None unless every model
    has its row, since the q values are corrected across all of them."""
    if "q" not in lp:
        return None
    q = dict(zip(lp["organism"] + "__" + lp["antibiotic"], lp["q"], strict=True))
    if any(m not in q for m in models):
        return None
    return sorted(m for m in models if pd.notna(q[m]) and q[m] >= alpha)


def _permutation_sensitivity(ev: pd.DataFrame, flagged: list[str], put) -> None:
    """Protocol §14 item 9: the grade distribution without the flagged models, and the
    layers each flagged model's candidates pass (an illustration, not a test). The grades
    of a flagged model are its {model}.n_{grade} keys. The hypotheses are recomputed in
    hypotheses.json."""
    layers = ("prevalence", "mda", "cpss", "pyseer")
    rest = ev[~ev["model"].isin(flagged)]
    if flagged and len(rest):
        pre = "lp_sensitivity"
        put(f"{pre}.n_models", int(rest["model"].nunique()), "grades_patterns", "pattern_id",
            "Models left once the models flagged by label permutation are excluded")
        put(f"{pre}.n_candidate_patterns", len(rest), "grades_patterns", "pattern_id",
            "Candidate patterns of those models")
        for g in F.GRADES:
            k = int((rest["grade"] == g).sum())
            put(f"{pre}.n_{g}", k, "grades_patterns", "grade",
                f"Candidate patterns graded {g} without the flagged models, primary mode")
            put(f"{pre}.share_{g}", k / len(rest), "grades_patterns", "grade",
                f"Share (0-1) of those candidate patterns graded {g}, primary mode")
        put(f"{pre}.weak_or_none_share", float(rest["grade"].isin(("weak", "none")).mean()),
            "grades_patterns", "grade",
            "Share (0-1) of those candidate patterns graded weak or none, primary mode")
        for layer in layers:
            put(f"{pre}.share_pass_{layer}", float(F._bool(rest[layer]).mean()),
                "grades_patterns", layer,
                f"Share (0-1) of those candidate patterns passing the {layer} layer")
    for mid in flagged:
        part = ev[ev["model"] == mid]
        for layer in layers:
            put(f"{mid}.n_pass_{layer}", int(F._bool(part[layer]).sum()), "grades_patterns",
                layer, f"Candidates of this flagged model passing the {layer} layer")


def _cpss_split_numbers(mid: str, cpss: pd.DataFrame, n_fits: int, cards: list,
                        grades: list[pd.DataFrame], put) -> None:
    """Why a model has no stable pattern (H1): the ARO behind the most patterns the
    stability selection picked, how often its patterns were picked and the grade they got.
    When the signal is spread over near-identical patterns of one determinant, each fit takes
    a few of them and no single one reaches the threshold. Only graded patterns carry a CARD
    annotation, so the count is a lower bound."""
    card = pd.concat([c for c in cards if c is not None and "allele_aware_aros" in c],
                     ignore_index=True).drop_duplicates("pattern_id")
    if card.empty:
        return
    sel = cpss[pd.to_numeric(cpss["n_selected"], errors="coerce").fillna(0) > 0]
    e = sel.merge(card[["pattern_id", "allele_aware_aros"]], on="pattern_id", how="inner")
    e = e.assign(aro=e["allele_aware_aros"].fillna("").astype(str).str.split(";")).explode("aro")
    e = e[e["aro"] != ""]
    if e.empty:
        return
    aro = e.groupby("aro")["pattern_id"].nunique().idxmax()
    pats = e[e["aro"] == aro].drop_duplicates("pattern_id")
    what = f"selected patterns annotated with ARO:{aro}, the ARO behind the most selected patterns"
    put(f"{mid}.cpss_split.n_patterns", len(pats), "cpss_patterns", "n_selected",
        f"Number of {what}")
    put(f"{mid}.cpss_split.selections_per_fit", float(pats["n_selected"].sum() / n_fits),
        "cpss_patterns", "n_selected", f"Of the {what}: selected per fit, mean over the fits")
    put(f"{mid}.cpss_split.max_pi", float(pats["pi"].max()), "cpss_patterns", "pi",
        f"Highest selection frequency among the {what}")
    g = pd.concat([x[["pattern_id", "grade"]] for x in grades if "grade" in x],
                  ignore_index=True).drop_duplicates("pattern_id") if grades else pd.DataFrame()
    if not g.empty:
        put(f"{mid}.cpss_split.n_confirmed",
            int(g[g["pattern_id"].isin(pats["pattern_id"])]["grade"].eq("confirmed").sum()),
            "grades_patterns", "grade", f"Of the {what}: graded confirmed, primary mode")


def _b_aros(units: pd.DataFrame | None) -> set[str]:
    """AROs behind b = 1 of a path's unitigs, primary mode: the determinants it recovered."""
    if units is None or "allele_aware_state" not in units:
        return set()
    b = units[units["allele_aware_state"] == "b"]
    return {a for x in b["allele_aware_aros"].dropna() for a in str(x).split(";") if a}


def card_tier_table(t: Tables, models: list[tuple[str, str]]) -> pd.DataFrame:
    """The candidates with a CARD hit of every model, with the RGI tier of their hits
    (protocol §14 item 10). Empty when no organism has its hit table."""
    return card_tiers.collect(t.get, models)


def _card_tier_numbers(table: pd.DataFrame, put) -> None:
    src = "grades_unitigs"
    tiered = table[table["tier"].notna() & table["group"].notna()]
    put("card_tier.n_patterns", len(tiered), src, "located_genomes",
        "Candidates with a CARD hit and an RGI tier (§14 item 10)")
    put("card_tier.n_patterns_without_tier", int(table["tier"].isna().sum()), src,
        "located_genomes", "Candidates with a CARD hit whose every (genome, ARO) pair is "
        "ambiguous or has no hit")
    put("card_tier.n_models", int(tiered["model"].nunique()), src, "pattern_id",
        "Models with at least one such candidate")
    aros = {a for x in tiered["aros"].dropna() for a in str(x).split(";")}
    put("card_tier.n_aros", len(aros), "rgi_hits", "aro", "AROs behind those candidates")
    for c, what in (("n_perfect", "Perfect"), ("n_strict", "Strict"),
                    ("n_ambiguous", "ambiguous (hits of the ARO in both tiers)"),
                    ("n_no_hit", "without a Perfect or Strict hit of the ARO")):
        put(f"card_tier.n_pairs_{c[2:]}", int(table[c].fillna(0).sum()), "rgi_hits", "cut_off",
            f"(genome, ARO) pairs of the candidates, {what}")
    for g in card_tiers.GROUPS:
        s = tiered[tiered["group"] == g]
        for tier in card_tiers.TIER_ORDER:
            put(f"card_tier.n_{g}_{tier.lower()}", int((s["tier"] == tier).sum()), "rgi_hits",
                "cut_off", f"Candidates of group {g} whose tier is {tier}")
        if len(s):
            put(f"card_tier.share_perfect_{g}", float((s["tier"] == "Perfect").mean()),
                "rgi_hits", "cut_off", f"Share (0-1) of the group {g} whose tier is Perfect")
    for org, s in tiered.groupby("organism"):
        for g in card_tiers.COMPARED:
            sg = s[s["group"] == g]
            put(f"card_tier.{org}.n_{g}", len(sg), "rgi_hits", "cut_off",
                f"Candidates of group {g} with a tier")
            if len(sg):
                put(f"card_tier.{org}.share_perfect_{g}", float((sg["tier"] == "Perfect").mean()),
                    "rgi_hits", "cut_off", f"Share (0-1) of the group {g} whose tier is Perfect")
    c = card_tiers.compare(tiered)
    put("card_tier.compare.n_confirmed", c["n_confirmed"], src, "allele_aware_grade",
        "confirmed candidates with b = 1 and a tier, in the comparison")
    put("card_tier.compare.n_candidate", c["n_candidate_b"], src, "allele_aware_grade",
        "candidate candidates with b = 1 and a tier, in the comparison")
    put("card_tier.compare.n_organisms", c["n_strata"], "rgi_hits", "cut_off",
        "Organisms with both grades and both outcomes, which the test is computed over")
    put("card_tier.compare.odds_ratio", c["odds_ratio"], "rgi_hits", "cut_off",
        "Mantel–Haenszel odds ratio of a Perfect tier, confirmed vs candidate, by organism")
    put("card_tier.compare.ci_low", c["ci_low"], "rgi_hits", "cut_off",
        "Lower 95% bound of that odds ratio (Robins–Breslow–Greenland)")
    put("card_tier.compare.ci_high", c["ci_high"], "rgi_hits", "cut_off",
        "Upper 95% bound of that odds ratio (Robins–Breslow–Greenland)")
    if c["p"] is not None:
        put("card_tier.compare.p", _four_significant(c["p"]), "rgi_hits", "cut_off",
            "Cochran–Mantel–Haenszel p, two-sided, no continuity correction", digits=None)


# The knowledge base (-entry KB) is read for what only it joins: the mechanisms and gene
# families behind the curated agreement and the NCBI context of the patterns. All three
# queries take the primary mode and the patterns of the model route, as v_biomarker does.
_KB_FROM = """
  FROM grade g
  JOIN candidate c ON c.model_id = g.model_id AND c.pattern_id = g.pattern_id
  JOIN model m ON m.model_id = g.model_id"""
_KB_WHERE = "\n WHERE g.primary_mode = 1 AND c.route = 'model'"
KB_GRADED = f"""
SELECT g.model_id, g.pattern_id, g.grade, g.card_state, m.organism_id,
       (SELECT pr.direction FROM prevalence_result pr
         WHERE pr.model_id = g.model_id AND pr.pattern_id = g.pattern_id) AS direction
{_KB_FROM}{_KB_WHERE}"""
# the AROs of the RGI hits on a pattern's b = 1 unitigs (every hit there, as card_layer
# lists them: an ARO beside the one that gave b counts too)
KB_B_AROS = f"""
SELECT DISTINCT g.model_id, g.pattern_id, m.organism_id, a.aro_accession, a.gene_family,
       a.mechanism
{_KB_FROM}
  JOIN pattern_member pm ON pm.model_id = g.model_id AND pm.pattern_id = g.pattern_id
  JOIN card_annotation ca ON ca.model_id = pm.model_id AND ca.unitig_id = pm.unitig_id
                         AND ca.mode = g.mode AND ca.state = 'b'
  JOIN card_hit h ON h.model_id = pm.model_id AND h.unitig_id = pm.unitig_id
  JOIN aro a ON a.aro_accession = h.aro_accession{_KB_WHERE} AND g.card_state = 'b'"""
# a pattern's context is that of its longest member with a context row, as in v_novel
KB_CONTEXT = f"""
SELECT g.model_id, g.pattern_id, x.unitig_id, x.n_hits, x.plasmid_share, x.gene, x.product
{_KB_FROM}
  JOIN unitig_context x ON x.organism_id = m.organism_id AND x.unitig_id = (
       SELECT pm.unitig_id FROM pattern_member pm
         JOIN unitig u ON u.unitig_id = pm.unitig_id
         JOIN unitig_context y ON y.unitig_id = pm.unitig_id AND y.organism_id = m.organism_id
        WHERE pm.model_id = g.model_id AND pm.pattern_id = g.pattern_id
        ORDER BY u.length DESC, u.unitig_id LIMIT 1){_KB_WHERE}"""
_PATTERN = ["model_id", "pattern_id"]
# Names a best nt hit's gene is checked against: the CARD terms RGI hit in these genomes and the
# AMR element symbols AMRFinderPlus called in them (both stored in the knowledge base).
KB_AMR_NAMES = """
SELECT name FROM aro
 UNION SELECT element_symbol FROM external_call WHERE type = 'AMR'"""
# A best hit whose product names one of these is on a mobile genetic element (M1, SM10).
MOBILE_PRODUCT = re.compile(
    r"transposase|integrase|recombinase|resolvase|insertion (?:sequence|element)|\bIS\d+|"
    r"\bTn\d+|conjugal transfer|conjugative transfer|relaxase|mobili[sz]ation protein|integron",
    re.I)


def _name_forms(name: str) -> set[str]:
    """A gene name as it is compared: lower case, with and without the bla prefix and an
    allele number (blaTEM-1, TEM-1 and tem all meet in tem)."""
    n = str(name).strip().lower()
    forms = {n, n.removeprefix("bla")}
    forms |= {re.sub(r"[-_]\d+[a-z]?$", "", f) for f in list(forms)}
    return {f for f in forms if len(f) >= 3}


def amr_name_index(names) -> set[str]:
    out: set[str] = set()
    for n in names:
        out |= _name_forms(n)
    return out


def is_amr_name(gene, index: set[str]) -> bool:
    """True when the gene name of a best hit is a CARD or AMRFinderPlus name in some form."""
    if gene is None or (isinstance(gene, float) and pd.isna(gene)) or not str(gene).strip():
        return False
    return bool(_name_forms(gene) & index)


def _terms(value) -> list[str]:
    """A ';'-joined CARD field as its terms."""
    return [x.strip() for x in str(value).split(";") if x.strip()] if pd.notna(value) else []


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.lower().removeprefix("antibiotic ")).strip("_")


def _n_patterns(df: pd.DataFrame) -> int:
    return len(df[_PATTERN].drop_duplicates())


def _kb_card_numbers(b: pd.DataFrame, put) -> None:
    src = "card_hit"
    put("kb.card.n_patterns_b", _n_patterns(b), src, "aro_accession",
        "Patterns of the model route with b = 1, primary mode")
    put("kb.card.n_aros", int(b["aro_accession"].nunique()), src, "aro_accession",
        "AROs of the RGI hits on the b = 1 unitigs of those patterns")
    mech = b.assign(term=b["mechanism"].map(_terms)).explode("term").dropna(subset=["term"])
    fam = b.assign(term=b["gene_family"].map(_terms)).explode("term").dropna(subset=["term"])
    put("kb.card.n_mechanisms", int(mech["term"].nunique()), "aro", "mechanism",
        "CARD resistance mechanisms of those AROs")
    put("kb.card.n_gene_families", int(fam["term"].nunique()), "aro", "gene_family",
        "CARD gene families of those AROs")
    orgs = fam.groupby("term")["organism_id"].nunique()
    put("kb.card.n_gene_families_multi_organism", int((orgs > 1).sum()), "aro", "gene_family",
        "Gene families of those AROs found in more than one organism")
    for org, s in b.groupby("organism_id"):
        n = _n_patterns(s)
        put(f"kb.card.{org}.n_patterns_b", n, src, "aro_accession",
            "Patterns of the model route with b = 1, primary mode")
        put(f"kb.card.{org}.n_gene_families", int(fam.loc[fam["organism_id"] == org, "term"]
                                                    .nunique()), "aro", "gene_family",
            "CARD gene families of the AROs on those patterns' b = 1 unitigs")
        for term, sm in mech[mech["organism_id"] == org].groupby("term"):
            k = _n_patterns(sm)
            put(f"kb.card.{org}.n_{_slug(term)}", k, "aro", "mechanism",
                f"Of them, patterns with an ARO of the mechanism '{term}'")
            put(f"kb.card.{org}.share_{_slug(term)}", k / n, "aro", "mechanism",
                f"Share (0-1) of this organism's b = 1 patterns with an ARO of the mechanism "
                f"'{term}' (a pattern can have several)")


def _kb_context_numbers(graded: pd.DataFrame, ctx: pd.DataFrame, put) -> None:
    src = "unitig_context"
    put("kb.context.n_patterns", len(graded), "grade", "pattern_id",
        "Graded patterns of the model route, primary mode")
    put("kb.context.n_patterns_searched", _n_patterns(ctx), src, "unitig_id",
        "Of them, patterns with a member searched in NCBI nt")
    c = graded.merge(ctx.drop_duplicates(_PATTERN), on=_PATTERN, how="left")

    def block(pre: str, s: pd.DataFrame, what: str) -> None:
        hit = s[pd.to_numeric(s["n_hits"], errors="coerce") > 0]
        put(f"{pre}.n_patterns", len(s), "grade", "grade", f"Patterns {what}")
        put(f"{pre}.n_with_hit", len(hit), src, "n_hits",
            f"Patterns {what} whose longest searched member has a hit in nt")
        ps = pd.to_numeric(hit["plasmid_share"], errors="coerce").dropna()
        if len(ps):
            put(f"{pre}.median_plasmid_share", float(ps.median()), src, "plasmid_share",
                "Median over those patterns of the share of the hits on plasmid records")
            put(f"{pre}.share_mostly_plasmid", float((ps >= 0.5).mean()), src, "plasmid_share",
                "Share (0-1) of those patterns with at least half of their hits on plasmid "
                "records")
            put(f"{pre}.share_no_plasmid", float((ps == 0).mean()), src, "plasmid_share",
                "Share (0-1) of those patterns with no hit on a plasmid record")
        if len(hit):
            named = hit["gene"].fillna("").astype(str).str.strip().ne("")
            put(f"{pre}.share_with_gene", float(named.mean()), src, "gene",
                "Share (0-1) of those patterns whose best hit names a gene")

    for grade in F.GRADES:
        s = c[c["grade"] == grade]
        block(f"kb.context.{grade}", s, f"graded {grade}")
        for d in ("R", "S"):
            block(f"kb.context.{grade}.{d}", s[s["direction"] == d],
                  f"graded {grade}, direction {d}")


def _kb_context_grade_numbers(graded: pd.DataFrame, ctx: pd.DataFrame, index: set[str],
                              put) -> None:
    """M1 (added 16.11.2026, exploratory): the nt context of the candidates without a CARD
    hit, by grade. Does the grade order them as an independent annotation would? Per pattern,
    the best hit of its longest searched member, as in _kb_context_numbers."""
    src = "unitig_context"
    c = graded.merge(ctx.drop_duplicates(_PATTERN), on=_PATTERN, how="left")
    c = c.assign(amr=c["gene"].map(lambda g: is_amr_name(g, index)),
                 mobile=c["product"].fillna("").astype(str).str.contains(MOBILE_PRODUCT))
    groups = [(f"kb.context.no_card.{g}", c[(c["card_state"] == "no_card_hit") &
                                         (c["grade"] == g)], f"graded {g} without a CARD hit")
              for g in F.GRADES if g != "confirmed"]
    groups.append(("kb.context.confirmed_all", c[c["grade"] == "confirmed"],
                   "graded confirmed (reference)"))
    for pre, s, what in groups:
        hit = s[pd.to_numeric(s["n_hits"], errors="coerce") > 0]
        put(f"{pre}.n_patterns", len(s), "grade", "grade", f"Patterns {what}")
        put(f"{pre}.n_with_hit", len(hit), src, "n_hits",
            f"Patterns {what} whose longest searched member has a hit in nt")
        if not len(hit):
            continue
        ps = pd.to_numeric(hit["plasmid_share"], errors="coerce")
        # A gene or product name exists only when the hit lies in a gene the nt record annotates;
        # the name shares below count the unnamed hits in their denominator.
        named = (hit["gene"].fillna("").astype(str).str.strip().ne("")
                 | hit["product"].fillna("").astype(str).str.strip().ne(""))
        for key, mask, desc in (
                ("share_named", named, "whose best hit lies in a gene the record annotates (gene or product name)"),
                ("share_amr_name", hit["amr"], "whose best hit's gene is a CARD or AMRFinderPlus name"),
                ("share_mobile", hit["mobile"], "whose best hit's product is a mobile-element protein"),
                ("share_amr_name_or_mobile", hit["amr"] | hit["mobile"], "with either of the two"),
                ("share_mostly_plasmid", ps >= 0.5, "with at least half of their hits on plasmid records")):
            put(f"{pre}.{key}", float(mask.mean()), src, "gene" if key in ("share_named", "share_amr_name",
                "share_amr_name_or_mobile") else "product" if key == "share_mobile" else "plasmid_share",
                f"Share (0-1) of those patterns {desc}")


# The three case studies of the knowledge base paper, chosen by a rule written before their
# results were read. Case 1: a confirmed pattern of a model that AMRFinderPlus assesses.
# Case 2 and case 3: a strong_novel pattern towards R, then towards S, whose longest searched
# member has a hit in nt. Each case comes from an organism the earlier cases did not take. The
# order: π, then the total gain in the final model (a pattern without one, from stability
# selection alone, ranks below those with one), then |Δ|. |Δ| was added on 14.11.2026, when
# the gain left case 2 tied; the lowest pyseer p and the highest MDA are reported beside it.
KB_CASE_POOL = """
SELECT b.model_id, b.organism_id, b.pattern_id, b.grade, b.direction, b.n_members, b.pi,
       c.total_gain, b.delta, b.prevalence_q, b.mda, b.mda_q, b.pyseer_p, b.n_layers,
       pr.present_resistant, pr.present_susceptible, pr.prev_resistant, pr.prev_susceptible,
       m.n_genomes, m.n_resistant, m.n_susceptible,
       (SELECT e.assessable FROM external_comparison e
         WHERE e.model_id = b.model_id AND e.tool = 'amrfinderplus') AS amrfinderplus,
       (SELECT mm.value FROM model_metric mm WHERE mm.model_id = b.model_id
           AND mm.arm = 'lineage_aware' AND mm.metric = 'roc_auc') AS roc_auc
  FROM v_biomarker b
  JOIN candidate c ON c.model_id = b.model_id AND c.pattern_id = b.pattern_id
  JOIN prevalence_result pr ON pr.model_id = b.model_id AND pr.pattern_id = b.pattern_id
  JOIN model m ON m.model_id = b.model_id
 WHERE b.route = 'model' AND b.grade IN ('confirmed', 'strong_novel')"""
_CASES = (("confirmed", None), ("strong_novel", "R"), ("strong_novel", "S"))
_RC = str.maketrans("ACGT", "TGCA")


def _carriers(con: sqlite3.Connection, model: str, pattern: int, n: int) -> int:
    """pattern.carriers as an integer whose n bits are the genomes, row 0 the highest bit."""
    blob = con.execute("SELECT carriers FROM pattern WHERE model_id = ? AND pattern_id = ?",
                       (model, int(pattern))).fetchone()[0]
    return int.from_bytes(blob, "big") >> (8 * len(blob) - n)


def _one_base_apart(a: str, b: str) -> bool:
    """Equal length and one mismatch, on either strand (stored sequences are canonical)."""
    def ham1(x: str, y: str) -> bool:
        return len(x) == len(y) and sum(i != j for i, j in zip(x, y, strict=True)) == 1
    return ham1(a, b) or ham1(a.translate(_RC)[::-1], b)


def _pick_case(pool: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """The chosen pattern and the patterns tied with it before |Δ|."""
    s = pool.assign(_gain=pd.to_numeric(pool["total_gain"], errors="coerce").fillna(-math.inf),
                    _abs_delta=pool["delta"].abs())
    s = s.sort_values(["pi", "_gain", "_abs_delta", "model_id", "pattern_id"],
                      ascending=[False, False, False, True, True])
    top = s.iloc[0]
    return top, s[(s["pi"] == top["pi"]) & (s["_gain"] == top["_gain"])]


def _kb_case_numbers(con: sqlite3.Connection, ctx: pd.DataFrame, put) -> None:
    pool = pd.read_sql_query(KB_CASE_POOL, con)
    hit = ctx[pd.to_numeric(ctx["n_hits"], errors="coerce") > 0].drop_duplicates(_PATTERN)
    pool = pool.merge(hit[_PATTERN + ["unitig_id", "n_hits", "plasmid_share"]], on=_PATTERN,
                      how="left")
    taken: list[str] = []
    for k, (grade, direction) in enumerate(_CASES, start=1):
        s = pool[(pool["grade"] == grade) & ~pool["organism_id"].isin(taken)]
        if direction is None:
            s = s[s["amrfinderplus"] == 1]
        else:
            s = s[(s["direction"] == direction) & s["unitig_id"].notna()]
        if s.empty:
            continue
        top, tied = _pick_case(s)
        taken.append(top["organism_id"])
        pre, m, p = f"kb.case{k}", top["model_id"], int(top["pattern_id"])
        what = f"case {k} ({grade}{', direction ' + direction if direction else ''})"
        put(f"{pre}.model_id", m, "model", "model_id", f"Model of {what}", digits=None)
        put(f"{pre}.pattern_id", str(p), "pattern", "pattern_id", f"Pattern of {what}",
            digits=None)
        put(f"{pre}.n_eligible", len(s), "grade", "grade",
            f"Patterns eligible for {what}, after the organisms of the earlier cases")
        put(f"{pre}.n_tied", len(tied), "cpss_result", "pi",
            f"Eligible patterns with the π and the total gain of {what} (|Δ| breaks the tie)")
        if len(tied) > 1:
            same = (tied.loc[tied["pyseer_p"].idxmin(), "pattern_id"] == p
                    and tied.loc[tied["mda"].idxmax(), "pattern_id"] == p
                    and tied.loc[tied["pyseer_p"].idxmin(), "model_id"] == m
                    and tied.loc[tied["mda"].idxmax(), "model_id"] == m)
            put(f"{pre}.alternatives_agree", int(same), "pyseer_result", "lrt_p",
                f"1 when the lowest pyseer p and the highest MDA among the tied patterns also "
                f"choose {what}")
        for col, table in (("n_members", "pattern"), ("present_resistant", "prevalence_result"),
                           ("present_susceptible", "prevalence_result"),
                           ("n_resistant", "model"), ("n_susceptible", "model"),
                           ("n_layers", "grade")):
            put(f"{pre}.{col}", int(top[col]), table, col, f"{col} of {what}")
        for col, label in (("prev_resistant", "resistant"), ("prev_susceptible", "susceptible")):
            put(f"{pre}.{col}", float(top[col]), "prevalence_result", col,
                f"Share (0-1) of the {label} genomes of the model that carry {what}")
        put(f"{pre}.pi", float(top["pi"]), "cpss_result", "pi", f"π of {what}")
        if pd.notna(top["total_gain"]):
            put(f"{pre}.total_gain", float(top["total_gain"]), "candidate", "total_gain",
                f"Total gain of {what} in the final model")
        for col, table in (("prevalence_q", "prevalence_result"), ("mda", "mda_result"),
                           ("mda_q", "mda_result"), ("pyseer_p", "pyseer_result")):
            put(f"{pre}.{col}", _four_significant(float(top[col])), table, col,
                f"{col} of {what}", digits=None)
        if pd.notna(top["roc_auc"]):
            put(f"{pre}.roc_auc", float(top["roc_auc"]), "model_metric", "value",
                f"Lineage-aware ROC-AUC of the model of {what}")
        members = con.execute(
            "SELECT pm.unitig_id, u.length, u.sequence, ca.state FROM pattern_member pm "
            "JOIN unitig u ON u.unitig_id = pm.unitig_id "
            "JOIN grade g ON g.model_id = pm.model_id AND g.pattern_id = pm.pattern_id "
            " AND g.primary_mode = 1 "
            "LEFT JOIN card_annotation ca ON ca.model_id = pm.model_id "
            " AND ca.unitig_id = pm.unitig_id AND ca.mode = g.mode "
            "WHERE pm.model_id = ? AND pm.pattern_id = ?", (m, p)).fetchall()
        put(f"{pre}.n_members_b", sum(r[3] == "b" for r in members), "card_annotation", "state",
            f"Members of {what} with b = 1")
        if pd.notna(top["unitig_id"]):
            put(f"{pre}.unitig_id", top["unitig_id"], "unitig_context", "unitig_id",
                f"Longest searched member of {what}", digits=None)
            put(f"{pre}.unitig_length", int(next(r[1] for r in members
                                                 if r[0] == top["unitig_id"])),
                "unitig", "length", f"Length (nt) of that member of {what}")
            put(f"{pre}.context_n_hits", int(top["n_hits"]), "unitig_context", "n_hits",
                f"nt hits of that member of {what}")
            put(f"{pre}.context_plasmid_share", float(top["plasmid_share"]), "unitig_context",
                "plasmid_share", f"Share (0-1) of those hits on plasmid records, {what}")
        # carriage beside the model's confirmed patterns, and a confirmed pattern one base away
        n = int(top["n_genomes"])
        mine = _carriers(con, m, p, n)
        put(f"{pre}.n_carriers", mine.bit_count(), "pattern", "carriers",
            f"Genomes of the model that carry {what}")
        others = [int(x) for (x,) in con.execute(
            "SELECT g.pattern_id FROM grade g JOIN candidate c ON c.model_id = g.model_id "
            "AND c.pattern_id = g.pattern_id WHERE g.model_id = ? AND g.primary_mode = 1 "
            "AND c.route = 'model' AND g.grade = 'confirmed' AND g.pattern_id <> ?", (m, p))]
        union = 0
        for q in others:
            union |= _carriers(con, m, q, n)
        put(f"{pre}.n_carriers_with_confirmed", (mine & union).bit_count(), "pattern",
            "carriers", f"Of them, genomes that also carry another confirmed pattern of the "
            f"model, {what}")
        seqs = [r[2] for r in members]
        for q in others:
            near = [r[0] for r in con.execute(
                "SELECT u.sequence FROM pattern_member pm JOIN unitig u USING (unitig_id) "
                "WHERE pm.model_id = ? AND pm.pattern_id = ?", (m, q))]
            if any(_one_base_apart(a, b) for a in seqs for b in near):
                theirs = _carriers(con, m, q, n)
                put(f"{pre}.neighbour_pattern_id", str(q), "pattern", "pattern_id",
                    f"Confirmed pattern of the model with a unitig one base from a member of "
                    f"{what}", digits=None)
                put(f"{pre}.neighbour_n_carriers", theirs.bit_count(), "pattern", "carriers",
                    f"Genomes that carry that confirmed pattern, {what}")
                put(f"{pre}.n_carriers_with_neighbour", (mine & theirs).bit_count(), "pattern",
                    "carriers", f"Genomes that carry both {what} and that confirmed pattern")
                break


# ---- the independent external isolates (CABBAGE) -------------------------------------
# Predictors: the final model, the KANIT lookup (one row per rule; "lookup" alone is the
# primary rule chosen on the training genomes) and the catalogue tools. A predictor is scored
# on a pair it can assess (a tool covers the antibiotic) that has at least min_per_class
# resistant and susceptible isolates. Pairs are compared as in the training set: median paired
# difference in balanced accuracy, organism-clustered bootstrap, and a sign test.
# The secondary score (R markers present minus S markers present, up to weak) enters as its
# ROC-AUC only (cabbage.lookup_score.*): it ranks isolates, and its cut at > 0 is not a rule.
CABBAGE_LOOKUP_RULES = ("upto_confirmed", "upto_strong_novel", "upto_candidate", "upto_weak",
                        "b1")
CABBAGE_TOOLS = ("amrfinderplus", "resfinder", "rgi_all", "rgi_without_near_universal")
_CABBAGE_STATS = {"balanced_accuracy": "ba", "sensitivity": "sensitivity",
                  "specificity": "specificity", "very_major_error_rate": "vme",
                  "major_error_rate": "me"}


def _cabbage_json(t: Tables, location: str, file: str, organism=None):
    """A CABBAGE summary file, None when absent or when the configuration has no such path."""
    try:
        return t.json(location, file, organism)
    except KeyError:
        return None


def _cabbage_scores(t: Tables, organisms: list[str], min_per_class: int):
    """One row per pair and predictor (scored pairs only), the lineage-split rows of the model
    and the primary lookup rule, and the primary rule."""
    rows, split = [], []
    sel = _cabbage_json(t, "cabbage_dir", "cabbage_lookup_selection.json") or {}
    primary = sel.get("primary_rule")
    for org in organisms:
        comp = t.get("cabbage_comparison", org)
        if comp is not None:
            for r in comp.itertuples():
                n_r, n = r.n_resistant, r.n
                if r.assessable == 1 and n_r >= min_per_class and n - n_r >= min_per_class:
                    rows.append({"model_id": r.model_id, "organism": org, "predictor": r.tool,
                                 **{k: getattr(r, c) for c, k in _CABBAGE_STATS.items()}})
        look = _cabbage_json(t, "cabbage_organism_dir", "cabbage_lookup_metrics.json", org) or {}
        for ab, p in look.get("pairs", {}).items():
            for rule in CABBAGE_LOOKUP_RULES:
                for part in ("all", "lineage_seen", "lineage_unseen"):
                    x = p.get(rule, {}).get(part, {})
                    if not x.get("assessed"):
                        continue
                    row = {"model_id": f"{org}__{ab}", "organism": org,
                           "predictor": f"lookup_{rule}",
                           **{k: x.get(c) for c, k in _CABBAGE_STATS.items()}}
                    if part == "all":
                        rows.append(row)
                    elif rule == primary:
                        split.append({**row, "predictor": "lookup", "part": part})
        mm = _cabbage_json(t, "cabbage_organism_dir", "cabbage_metrics.json", org) or {}
        for ab, p in mm.get("pairs", {}).items():
            for part in ("all", "lineage_seen", "lineage_unseen"):
                x = p.get(part, {})
                if x.get("assessed"):
                    split.append({"model_id": f"{org}__{ab}", "organism": org,
                                  "predictor": "model", "part": part,
                                  "ba": x.get("balanced_accuracy"), "roc_auc": x.get("roc_auc"),
                                  "internal_roc_auc": p.get("internal_lineage_aware", {})
                                  .get("roc_auc")})
    scores = pd.DataFrame(rows)
    if primary and len(scores):
        scores = pd.concat([scores, scores[scores["predictor"] == f"lookup_{primary}"]
                            .assign(predictor="lookup")], ignore_index=True)
    return scores, pd.DataFrame(split), primary, sel


def _cabbage_flow_numbers(t: Tables, organisms: list[str], sel: dict, put) -> None:
    left = (_cabbage_json(t, "cabbage_dir", "cabbage_selection.json") or {}).get("isolates_left", {})
    n_sel = sum(int(v.get("assembly", 0)) for v in left.values())
    if n_sel:
        put("cabbage.n_isolates_selected", n_sel, "cabbage_selection", "isolates_left",
            "External isolates left after every selection step, with an assembly")
    fetched = passed = used = 0
    for org in organisms:
        prep = _cabbage_json(t, "cabbage_organism_dir", "cabbage_prepare.json", org) or {}
        fetched += int(prep.get("n_fetched", 0))
        passed += int(prep.get("n_qc_passed", 0))
        pred = t.get("cabbage_predictions", org, dtype={"biosample_id": str})
        if pred is not None:
            used += int(pred["biosample_id"].nunique())
    if fetched:
        put("cabbage.n_isolates_fetched", fetched, "cabbage_prepare", "n_fetched",
            "External isolates whose assembly was downloaded")
        put("cabbage.n_isolates_qc_pass", passed, "cabbage_prepare", "n_qc_passed",
            "Of them, isolates that pass the CheckM2 gate of the training genomes")
    if used:
        put("cabbage.n_isolates_predicted", used, "cabbage_predictions", "biosample_id",
            "External isolates with a phenotype for a modelled pair, predicted")
    for rule, ba in sel.get("balanced_accuracy", {}).items():
        put(f"cabbage.lookup.training_ba.{rule}", float(ba), "cabbage_lookup_selection",
            "balanced_accuracy", f"Pooled training balanced accuracy of the lookup rule {rule}")


def _cabbage_compare(scores: pd.DataFrame, a: str, b: str, put, n_boot: int) -> None:
    ba = scores.pivot_table(index="model_id", columns="predictor", values="ba", aggfunc="first")
    if a not in ba or b not in ba:
        return
    both = ba[[a, b]].dropna()
    if both.empty:
        return
    d = both[a] - both[b]
    s = concordance.sign_test(d)
    pd_ = concordance.paired_difference(d, [i.split("__")[0] for i in d.index], n_boot=n_boot)
    pre, src = f"cabbage.compare.{a}__{b}", "cabbage_comparison"
    put(f"{pre}.n_pairs", len(both), src, "balanced_accuracy", f"Pairs both {a} and {b} score")
    put(f"{pre}.median_a", float(both[a].median()), src, "balanced_accuracy",
        f"Median balanced accuracy of {a} on those pairs")
    put(f"{pre}.median_b", float(both[b].median()), src, "balanced_accuracy",
        f"Median balanced accuracy of {b} on those pairs")
    put(f"{pre}.median_diff", pd_["median"], src, "balanced_accuracy",
        f"Median paired difference in balanced accuracy, {a} minus {b}")
    put(f"{pre}.ci_low", pd_["ci_low"], src, "balanced_accuracy",
        "Lower 95% bound of that difference, organism-clustered bootstrap")
    put(f"{pre}.ci_high", pd_["ci_high"], src, "balanced_accuracy",
        "Upper 95% bound of that difference, organism-clustered bootstrap")
    put(f"{pre}.n_organisms", pd_["n_clusters"], src, "model_id", "Organisms behind those pairs")
    put(f"{pre}.n_a_better", s["n_positive"], src, "balanced_accuracy", f"Pairs where {a} wins")
    put(f"{pre}.n_b_better", s["n_negative"], src, "balanced_accuracy", f"Pairs where {b} wins")
    put(f"{pre}.n_tied", s["n_tied"], src, "balanced_accuracy", "Pairs where they tie")
    if s["p_value"] is not None:
        put(f"{pre}.p", _four_significant(s["p_value"]), src, "balanced_accuracy",
            f"Two-sided sign test, {a} vs {b}, ties dropped", digits=None)


def _cabbage_score_numbers(t: Tables, organisms: list[str], put) -> None:
    auc = []
    for org in organisms:
        look = _cabbage_json(t, "cabbage_organism_dir", "cabbage_lookup_metrics.json", org) or {}
        for p in look.get("pairs", {}).values():
            x = p.get("score", {}).get("all", {})
            if x.get("assessed") and x.get("roc_auc") is not None:
                auc.append(float(x["roc_auc"]))
    if not auc:
        return
    s, src = pd.Series(auc), "cabbage_lookup_metrics"
    put("cabbage.lookup_score.n_pairs", len(s), src, "roc_auc",
        "Pairs the secondary lookup score (R minus S markers present, up to weak) is scored on")
    put("cabbage.lookup_score.median_roc_auc", float(s.median()), src, "roc_auc",
        "Median external ROC-AUC of the secondary lookup score")
    put("cabbage.lookup_score.min_roc_auc", float(s.min()), src, "roc_auc", "Lowest, secondary score")
    put("cabbage.lookup_score.max_roc_auc", float(s.max()), src, "roc_auc", "Highest, secondary score")


def _cabbage_numbers(t: Tables, organisms: list[str], put) -> None:
    min_per_class = int(t.config.get("cabbage", {}).get("min_per_class", 10))
    n_boot = int(t.config["cv"]["n_bootstrap"])
    scores, split, primary, sel = _cabbage_scores(t, organisms, min_per_class)
    _cabbage_flow_numbers(t, organisms, sel, put)
    _cabbage_score_numbers(t, organisms, put)
    if scores.empty:
        return
    if primary:
        put("cabbage.lookup.primary_rule", primary, "cabbage_lookup_selection", "primary_rule",
            "Lookup rule chosen on the training genomes", digits=None)
    src = "cabbage_comparison"
    model_pairs = scores.loc[scores["predictor"] == "model", "model_id"]
    put("cabbage.n_pairs_scored", int(model_pairs.nunique()), src, "model_id",
        f"Pairs with at least {min_per_class} resistant and susceptible external isolates")
    put("cabbage.n_organisms_scored", int(model_pairs.str.split("__").str[0].nunique()), src,
        "model_id", "Organisms with such a pair")
    for p, s in scores.groupby("predictor"):
        pre = f"cabbage.{p}"
        put(f"{pre}.n_pairs", len(s), src, "assessable", f"Pairs {p} scores")
        for k in ("ba", "sensitivity", "specificity", "vme", "me"):
            v = pd.to_numeric(s[k], errors="coerce").dropna()
            if len(v):
                put(f"{pre}.median_{k}", float(v.median()), src, k, f"Median {k} of {p}")
        v = pd.to_numeric(s["ba"], errors="coerce").dropna()
        if len(v):
            put(f"{pre}.min_ba", float(v.min()), src, "balanced_accuracy", f"Lowest, {p}")
            put(f"{pre}.max_ba", float(v.max()), src, "balanced_accuracy", f"Highest, {p}")
            if p.startswith("rgi"):
                put(f"{pre}.n_ba_half", int((v == 0.5).sum()), src, "balanced_accuracy",
                    f"Pairs where {p} scores 0.5 (one class called for every isolate)")
    for tool in CABBAGE_TOOLS:
        _cabbage_compare(scores, "lookup", tool, put, n_boot)
        _cabbage_compare(scores, "model", tool, put, n_boot)
    _cabbage_compare(scores, "lookup", "lookup_b1", put, n_boot)
    _cabbage_compare(scores, "model", "lookup", put, n_boot)
    # the pairs every predictor scores
    wide = scores.pivot_table(index="model_id", columns="predictor", values="ba", aggfunc="first")
    cols = [c for c in ("model", "lookup", "lookup_b1", *CABBAGE_TOOLS) if c in wide]
    common = wide[cols].dropna()
    put("cabbage.common.n_pairs", len(common), src, "model_id",
        "Pairs every predictor (model, lookup, lookup b1, four tools) scores")
    for c in cols:
        if len(common):
            put(f"cabbage.common.{c}.median_ba", float(common[c].median()), src,
                "balanced_accuracy", f"Median balanced accuracy of {c} on those pairs")
    if split.empty:
        return
    # the model's ROC-AUC beside the internal lineage-aware one
    m = split[(split["predictor"] == "model") & (split["part"] == "all")]
    auc = pd.to_numeric(m["roc_auc"], errors="coerce")
    if auc.notna().any():
        put("cabbage.model.median_roc_auc", float(auc.median()), "cabbage_metrics", "roc_auc",
            "Median external ROC-AUC of the models")
        put("cabbage.model.min_roc_auc", float(auc.min()), "cabbage_metrics", "roc_auc", "Lowest")
        put("cabbage.model.max_roc_auc", float(auc.max()), "cabbage_metrics", "roc_auc", "Highest")
        by_org = m.assign(auc=auc).groupby("organism")["auc"].median()
        put("cabbage.model.roc_auc_median_by_organism_min", float(by_org.min()),
            "cabbage_metrics", "roc_auc", "Lowest organism median of the external ROC-AUC")
        put("cabbage.model.roc_auc_median_by_organism_max", float(by_org.max()),
            "cabbage_metrics", "roc_auc", "Highest organism median of the external ROC-AUC")
        inner = pd.to_numeric(m["internal_roc_auc"], errors="coerce")
        ok = auc.notna() & inner.notna()
        if ok.any():
            gap = (auc - inner)[ok]
            put("cabbage.model.median_internal_roc_auc", float(inner[ok].median()),
                "cabbage_metrics", "roc_auc", "Median internal lineage-aware ROC-AUC, same pairs")
            put("cabbage.model.median_auc_external_minus_internal", float(gap.median()),
                "cabbage_metrics", "roc_auc", "Median of external minus internal ROC-AUC")
            put("cabbage.model.n_external_above_internal", int((gap > 0).sum()),
                "cabbage_metrics", "roc_auc", "Pairs whose external ROC-AUC is the higher")
            for org, g in m.assign(gap=auc - inner)[ok].groupby("organism"):
                put(f"cabbage.{org}.model_median_auc_external_minus_internal",
                    float(g["gap"].median()), "cabbage_metrics", "roc_auc",
                    "Median of external minus internal ROC-AUC, this organism")
    # seen and unseen lineages, on the pairs scored in both
    for p in ("model", "lookup"):
        s = split[split["predictor"] == p]
        w = s.pivot_table(index="model_id", columns="part", values="ba", aggfunc="first")
        if not {"lineage_seen", "lineage_unseen"} <= set(w.columns):
            continue
        w = w[["lineage_seen", "lineage_unseen"]].dropna()
        if w.empty:
            continue
        d = w["lineage_unseen"] - w["lineage_seen"]
        pd_ = concordance.paired_difference(d, [i.split("__")[0] for i in d.index],
                                            n_boot=n_boot)
        pre = f"cabbage.lineage.{p}"
        put(f"{pre}.n_pairs", len(w), "cabbage_metrics", "lineage_seen",
            "Pairs scored on both seen and unseen lineages")
        put(f"{pre}.median_ba_seen", float(w["lineage_seen"].median()), "cabbage_metrics",
            "balanced_accuracy", "Median balanced accuracy, isolates of lineages the model saw")
        put(f"{pre}.median_ba_unseen", float(w["lineage_unseen"].median()), "cabbage_metrics",
            "balanced_accuracy", "Median balanced accuracy, isolates of unseen lineages")
        put(f"{pre}.median_diff", pd_["median"], "cabbage_metrics", "balanced_accuracy",
            "Median paired difference, unseen minus seen")
        put(f"{pre}.ci_low", pd_["ci_low"], "cabbage_metrics", "balanced_accuracy",
            "Lower 95% bound, organism-clustered bootstrap")
        put(f"{pre}.ci_high", pd_["ci_high"], "cabbage_metrics", "balanced_accuracy",
            "Upper 95% bound, organism-clustered bootstrap")


# ---- Makale 3: arms B and C under a feature budget ---------------------------------------
# Arm A is the main run. Arms B (arm A's unitig patterns) and C (the organism's raw canonical
# 31-mers, not collapsed) keep the K features with the largest χ² inside every training set
# and are graded as arm A is (config/m3_B.yml, config/m3_C.yml). An arm's outputs sit where its
# overlay moves them, below this configuration's own prefix. Only the arm's own outputs are
# read, so the overlay's exclusions (the inputs it shares with arm A) are not carried over.
# Performance: lineage-aware ROC-AUC, pair by pair as the CABBAGE predictors. Redundancy:
# distinct presence vectors among the final model's K features, one vector being one signal
# (arm C repeats a locus's signal once per k-mer). Grades: the arms against each other, and
# against arm A only inside arm A's own χ² prefilter, where the budget's criterion would have
# kept them too (decision of 15.11).
M3_ARMS = ("B", "C")
M3_LAYERS = ("prevalence", "mda", "cpss", "pyseer")
# copies of a candidate's presence vector among the K features
M3_COPIES = (("one", 1, 1), ("2_10", 2, 10), ("over_10", 11, None))


def _m3_arm_tables(t: Tables) -> dict[str, Tables]:
    """The tables of every arm whose overlay exists, under this configuration's prefix."""
    if not isinstance(t, Tables):
        return {}
    arms = {}
    base = t.config.get("paths_prefix")
    for arm in M3_ARMS:
        p = PROJECT_ROOT / "config" / f"m3_{arm}.yml"
        prefix = (yaml.safe_load(p.read_text()) or {}).get("paths_prefix") if p.exists() else None
        if prefix:
            arms[arm] = Tables({**t.config, "paths_prefix_exclude": [],
                                "paths_prefix": f"{base.rstrip('/')}/{prefix}" if base else prefix})
    return arms


def _m3_vectors(t, org: str, ab: str) -> pd.DataFrame | None:
    """Per feature of the final model (row = pattern_id): its presence vector's number and
    copies among the K features. None without the arm's model matrix."""
    try:
        d = contract.output_path("matrix_dir", "", t.config, org, ab)
    except KeyError:
        return None
    if not (d / "patterns.bin").exists():
        return None
    shape = json.loads((d / "matrix_summary.json").read_text())["patterns_shape"]
    bits = np.fromfile(d / "patterns.bin", dtype=np.uint8).reshape(shape)
    _, vector, copies = np.unique(bits, axis=0, return_inverse=True, return_counts=True)
    vector = vector.ravel()
    return pd.DataFrame({"vector": vector, "copies": copies[vector]})


M3_METRICS = {"roc_auc": ("", "lineage-aware ROC-AUC"),
              "balanced_accuracy": (".ba", "lineage-aware balanced accuracy"),
              "pr_auc": (".pr_auc", "lineage-aware PR-AUC")}


def _m3_compare(scores: pd.DataFrame, a: str, b: str, put, n_boot: int,
                metric: str = "roc_auc") -> None:
    s_ = scores[scores["metric"] == metric].pivot(index="model", columns="arm", values="value")
    if a not in s_ or b not in s_:
        return
    both = s_[[a, b]].dropna()
    if both.empty:
        return
    d = both[a] - both[b]
    s = concordance.sign_test(d)
    pd_ = concordance.paired_difference(d, [i.split("__")[0] for i in d.index], n_boot=n_boot)
    sub, what = M3_METRICS[metric]
    pre, src = f"m3.compare.{a}__{b}{sub}", "cv_metrics"
    put(f"{pre}.n_pairs", len(both), src, metric, f"Models both arm {a} and arm {b} evaluate")
    put(f"{pre}.median_diff", pd_["median"], src, metric,
        f"Median paired difference in {what}, arm {a} minus arm {b}")
    put(f"{pre}.ci_low", pd_["ci_low"], src, metric,
        "Lower 95% bound of that difference, organism-clustered bootstrap")
    put(f"{pre}.ci_high", pd_["ci_high"], src, metric,
        "Upper 95% bound of that difference, organism-clustered bootstrap")
    put(f"{pre}.n_organisms", pd_["n_clusters"], src, "model_id", "Organisms behind those models")
    put(f"{pre}.n_a_better", s["n_positive"], src, metric, f"Models where arm {a} is higher")
    put(f"{pre}.n_b_better", s["n_negative"], src, metric, f"Models where arm {b} is higher")
    put(f"{pre}.n_tied", s["n_tied"], src, metric, "Models where the arms tie")
    if s["p_value"] is not None:
        put(f"{pre}.p", _four_significant(s["p_value"]), src, metric,
            f"Two-sided sign test of {what}, arm {a} vs arm {b}, ties dropped", digits=None)


def _m3_grade_numbers(c: pd.DataFrame, pre: str, what: str, put) -> None:
    """Grades, CARD state and layers of one arm's candidates (one row per candidate)."""
    n = len(c)
    put(f"{pre}.n_candidates", n, "grades_patterns", "pattern_id",
        f"Candidates of {what}, summed over models")
    if not n:
        return
    for g in F.GRADES:
        k = int((c["grade"] == g).sum())
        put(f"{pre}.n_{g}", k, "grades_patterns", "grade", f"Candidates of {what} graded {g}")
        put(f"{pre}.share_{g}", k / n, "grades_patterns", "grade",
            f"Share (0-1) of the candidates of {what} graded {g}")
    b = c[c["allele_aware_card_state"] == "b"]
    put(f"{pre}.share_card_b", len(b) / n, "grades_patterns", "allele_aware_card_state",
        f"Share (0-1) of the candidates of {what} with b = 1")
    for layer in M3_LAYERS:
        put(f"{pre}.share_pass_{layer}", float(c[layer].astype(bool).mean()), "grades_patterns",
            layer, f"Share (0-1) of the candidates of {what} passing the {layer} layer")
        if len(b):
            put(f"{pre}.b.share_pass_{layer}", float(b[layer].astype(bool).mean()),
                "grades_patterns", layer,
                f"Share (0-1) of the b = 1 candidates of {what} passing the {layer} layer")
    aros = b.assign(aro=b["aros"].astype(str).str.split(";")).explode("aro")
    per_aro = aros[aros["aro"].str.len() > 0].groupby(["model", "aro"]).size()
    if len(per_aro):
        put(f"{pre}.candidates_per_b_aro_median", float(per_aro.median()), "card_patterns",
            "allele_aware_aros",
            f"Candidates of {what} carrying one b = 1 determinant, median over models and AROs")
        put(f"{pre}.candidates_per_b_aro_max", int(per_aro.max()), "card_patterns",
            "allele_aware_aros", "Largest of those counts")


def _prefixed(t, *parts: str) -> Path:
    """A project path below this configuration's prefix (outputs with no location key)."""
    prefix = (getattr(t, "config", None) or {}).get("paths_prefix")
    return PROJECT_ROOT.joinpath(*([prefix] if prefix else []), *parts)


# the steps of an arm's run, by what they cost. Arm A is the main run: its unitig store and
# model matrices are its store, and only the steps the arms also run are compared (not the
# label permutation, the association set, RGI, the external tools or genome QC).
M3_COST_STEPS = {"store": ("M3_KMER_STORE", "STORE", "MODEL_MATRIX"), "budget": ("M3_BUDGET",),
                 "cv": ("CV_PACKING", "CV_UNITS", "CV_FINAL", "CV_METRICS"),
                 "evidence": ("CPSS_PREFILTER", "CPSS_CHUNK", "CPSS_SELECT", "PREVALENCE", "MDA",
                              "CARD_LAYER", "PYSEER_PREP", "PYSEER_LMM", "PYSEER_POST", "GRADING")}


def _m3_cost_numbers(t, arms: dict, put) -> None:
    """Each arm's cost from the resource report of its last completed run (arm A: the main
    run's). A run resumed from an earlier one lists the cached tasks too, with their own jobs,
    so the last run holds all; failed attempts of earlier runs are not counted."""
    runs = _prefixed(t, "runs", "nextflow")
    if not runs.is_dir():
        return
    last: dict[str, Path] = {}
    for d in sorted(runs.iterdir()):
        man = d / "run_manifest.json"
        if not (man.exists() and (d / "resources_summary.json").exists()):
            continue
        m = json.loads(man.read_text())
        overlay = str((m.get("params") or {}).get("config_overlay") or "")
        ok = (m.get("completed") or {}).get("success")
        entry = (m.get("run") or {}).get("entry")
        if ok and entry == "main" and not overlay:
            last["A"] = d
        for arm in arms:
            if ok and entry == "M3" and overlay.endswith(f"m3_{arm}.yml"):
                last[arm] = d
    for arm, d in last.items():
        s = json.loads((d / "resources_summary.json").read_text())
        pre, src = f"m3.{arm}.cost", "resources_summary"
        put(f"{pre}.core_hours", float(s["total"]["core_hours"]), src, "core_hours",
            f"Core-hours allocated to arm {arm}'s whole run (SLURM accounting)")
        put(f"{pre}.cpu_hours", float(s["total"]["cpu_hours"]), src, "cpu_hours",
            f"CPU-hours used by arm {arm}'s run (SLURM accounting)")
        put(f"{pre}.n_tasks", int(s["total"]["tasks"]), src, "tasks", f"Tasks of arm {arm}'s run")
        shared = {"core_hours": 0.0, "cpu_hours": 0.0}
        for step, procs in M3_COST_STEPS.items():
            own = {p: v for p, v in s["by_process"].items() if p.split(":")[-1] in procs}
            if not own:
                continue
            for k in shared:
                shared[k] += sum(v.get(k) or 0.0 for v in own.values())
            put(f"{pre}.{step}.cpu_hours", float(sum(v.get("cpu_hours") or 0.0
                                                     for v in own.values())), src,
                "cpu_hours", f"CPU-hours of arm {arm}'s {step} steps")
            put(f"{pre}.{step}.core_hours", float(sum(v.get("core_hours") or 0.0
                                                      for v in own.values())), src,
                "core_hours", f"Core-hours allocated to arm {arm}'s {step} steps")
            put(f"{pre}.{step}.max_rss_gib", float(max(v.get("max_rss_gib") or 0.0
                                                       for v in own.values())),
                src, "max_rss_gib", f"Largest SLURM MaxRSS (GiB) of arm {arm}'s {step} steps")
            put(f"{pre}.{step}.max_elapsed_h", float(max(v.get("max_elapsed_h") or 0.0
                                                         for v in own.values())),
                src, "max_elapsed_h", f"Longest {step} task of arm {arm} (hours)")
        for k, what in (("core_hours", "Core-hours allocated"), ("cpu_hours", "CPU-hours used")):
            put(f"{pre}.shared_steps.{k}", float(shared[k]), src, k,
                f"{what} by arm {arm}'s store, budget, cross-validation and evidence steps "
                "(the steps every arm runs)")


def _m3_measurement_numbers(t, put) -> None:
    """The measurements made before the arms ran (scripts/m3_0a, m3_0b, m3_0c)."""
    tables = _prefixed(t, "results", "tables")
    homo: dict[str, float] = {}
    design: set[tuple[int, int]] = set()
    for p in sorted(tables.glob("m3_0a_summary_*.json")):
        s = json.loads(p.read_text())
        org = s["organism"]
        homo[org] = s["homogeneous_frac"]
        put(f"m3_0a.{org}.homogeneous_share", float(s["homogeneous_frac"]), "m3_0a_summary",
            "homogeneous_frac", "Share (0-1) of the tested unitigs whose 31-mers all sit in the "
            "same sampled genomes")
        put(f"m3_0a.{org}.max_genomes_apart", int(s["max_pairwise_hamming"]["max"]),
            "m3_0a_summary", "max_pairwise_hamming",
            "Most genomes in which two 31-mers of one unitig differ")
        design.add((int(s["n_unitigs_tested"]), int(s["n_genomes_sampled"])))
    if len(design) == 1:            # the same sample size in every organism
        n_units, n_genomes = design.pop()
        put("m3_0a.n_unitigs_tested", n_units, "m3_0a_summary", "n_unitigs_tested",
            "Unitigs tested per organism")
        put("m3_0a.n_genomes_sampled", n_genomes, "m3_0a_summary", "n_genomes_sampled",
            "Genomes sampled per organism")
    if homo:
        put("m3_0a.homogeneous_share_min", float(min(homo.values())), "m3_0a_summary",
            "homogeneous_frac", "Share (0-1) of homogeneous unitigs, lowest organism")
        put("m3_0a.homogeneous_share_max", float(max(homo.values())), "m3_0a_summary",
            "homogeneous_frac", "Share (0-1) of homogeneous unitigs, highest organism")
    b = tables / "m3_0b_kmer_feature_count.csv"
    if b.exists():
        f = pd.read_csv(b)
        for col, name, unit in (("n_kmer_features", "kmer_features", "k-mer features"),
                                ("kmer_per_pattern", "kmer_per_pattern", "31-mers per pattern"),
                                ("kmer_csr_gib", "kmer_matrix_gib",
                                 "GiB of the k-mer matrix at full width (8 bytes per entry)"),
                                ("unitig_csr_gib", "unitig_matrix_gib",
                                 "GiB of the unitig matrix (8 bytes per entry)")):
            for stat in ("median", "min", "max"):
                put(f"m3_0b.{name}_{stat}", float(getattr(f[col], stat)()),
                    "m3_0b_kmer_feature_count", col, f"{unit}, {stat} over models")
        put("m3_0b.n_models", len(f), "m3_0b_kmer_feature_count", "antibiotic", "Models measured")
        put("m3_0b.n_models_kmer_matrix_over_100gib", int((f["kmer_csr_gib"] > 100).sum()),
            "m3_0b_kmer_feature_count", "kmer_csr_gib",
            "Models whose full k-mer matrix alone exceeds 100 GiB")
    c = tables / "m3_0c_summary.json"
    if c.exists():
        r = json.loads(c.read_text())["collapse_ratio"]
        for stat in ("median", "q1", "q3", "min", "max"):
            put(f"m3_0c.collapse_ratio_{stat}", float(r[stat]), "m3_0c_summary", "collapse_ratio",
                f"Unitigs per pattern of a model, {stat} over models")


def _m3_collect(t: Tables, arms: dict, organisms: list[str]) -> dict | None:
    """Everything the arms' keys and figures read, per model (None without the panel or an arm):
    scores (model, arm, metric, value), chi2 (arm -> model -> χ² of the K-th feature),
    candidates (A_prefilter, B, C -> one row per candidate; copies and vector where the arm's
    matrix is at hand), redundancy (arm -> one row per model), short (arm -> models whose final
    model gains on fewer than top_gain features) and lengths (arm -> candidate lengths)."""
    dec = t.get("panel_decisions")
    if dec is None or not arms:
        return None
    pairs = dec[(dec["decision"] == panel.INCLUDED) & dec["organism"].isin(organisms)]
    top_gain = int((t.config.get("candidates") or {}).get("top_gain") or 50)
    scores: list[dict] = []
    chi2: dict[str, dict[str, float]] = {}
    cands: dict[str, list[pd.DataFrame]] = {}
    redundancy: dict[str, list[dict]] = {}
    short: dict[str, int] = {}
    lengths: dict[str, list] = {}
    for org, ab in zip(pairs["organism"], pairs["antibiotic"], strict=True):
        mid = f"{org}__{ab}"
        for arm, tx in {"A": t, **arms}.items():
            m = tx.json("cv_dir", "metrics.json", org, ab)
            if m is None:
                continue
            la = m["arms"]["lineage_aware"]
            scores += [{"model": mid, "arm": arm, "metric": k, "value": la.get(k)}
                       for k in M3_METRICS]
            g = tx.get("grades_patterns", org, ab)
            pf = tx.get("cpss_prefilter", org, ab)
            if g is None or pf is None:
                continue
            card = tx.get("card_patterns", org, ab, dtype={"allele_aware_aros": str})
            aros = (card.set_index("pattern_id")["allele_aware_aros"] if card is not None
                    else pd.Series(dtype=str))
            g = g.assign(model=mid, aros=g["pattern_id"].map(aros).fillna(""))
            if arm == "A":
                cands.setdefault("A_prefilter", []).append(g[g["pattern_id"].isin(pf["pattern_id"])])
                continue
            chi2.setdefault(arm, {})[mid] = float(pf["chi2"].min())
            cand = tx.get("candidates", org, ab)
            if cand is not None and cand["gain_rank"].notna().sum() < top_gain:
                short[arm] = short.get(arm, 0) + 1
            units = tx.get("card_unitigs", org, ab)
            if units is not None and "length" in units:
                lengths.setdefault(arm, []).extend(units["length"].dropna().tolist())
            v = _m3_vectors(tx, org, ab)
            if v is not None:
                k = len(v)
                n_vec = int(v["vector"].nunique())
                redundancy.setdefault(arm, []).append(
                    {"model": mid, "distinct": n_vec, "per_vector": k / n_vec,
                     "largest_share": float(v["copies"].max()) / k})
                g = g.join(v, on="pattern_id")
            cands.setdefault(arm, []).append(g)
    return {"scores": pd.DataFrame(scores, columns=["model", "arm", "metric", "value"]).dropna(),
            "chi2": chi2, "short": short, "lengths": lengths, "top_gain": top_gain,
            "candidates": {k: pd.concat(v, ignore_index=True) for k, v in cands.items()},
            "redundancy": {k: pd.DataFrame(v) for k, v in redundancy.items()}}


def _m3_numbers(t: Tables, arms: dict, organisms: list[str], put) -> None:
    got = _m3_collect(t, arms, organisms)
    if got is None:
        return
    sc, chi2, short, lengths = got["scores"], got["chi2"], got["short"], got["lengths"]
    top_gain = got["top_gain"]
    n_boot = int(t.config["cv"]["n_bootstrap"])
    for (arm, metric), s in sc.groupby(["arm", "metric"]):
        name = {"roc_auc": "roc_auc", "balanced_accuracy": "ba", "pr_auc": "pr_auc"}[metric]
        if metric == "roc_auc":
            put(f"m3.{arm}.n_models", len(s), "cv_metrics", metric, f"Models arm {arm} evaluates")
        put(f"m3.{arm}.{name}_median", float(s["value"].median()), "cv_metrics", metric,
            f"Median {M3_METRICS[metric][1]} of arm {arm}")
    for x, y in (("C", "B"), ("B", "A"), ("C", "A")):
        for metric in M3_METRICS:
            _m3_compare(sc, x, y, put, n_boot, metric)

    for arm, per in chi2.items():
        put(f"m3.{arm}.chi2_at_budget_median", float(pd.Series(per).median()), "cpss_prefilter",
            "chi2", f"χ² of the K-th feature of arm {arm}'s final model, median over models")
        put(f"m3.{arm}.n_models_fewer_than_top_gain", short.get(arm, 0), "candidates",
            "gain_rank", f"Models of arm {arm} whose final model gains on fewer than "
            f"{top_gain} features")
    if "B" in chi2 and "C" in chi2:
        both = pd.DataFrame(chi2)[["B", "C"]].dropna()
        put("m3.compare.C__B.n_chi2_at_budget_higher", int((both["C"] > both["B"]).sum()),
            "cpss_prefilter", "chi2", "Models where arm C's K-th χ² is above arm B's")

    for arm, r in got["redundancy"].items():
        pre = f"m3.{arm}"
        put(f"{pre}.n_models_with_matrix", len(r), "model_patterns", "pattern_id",
            f"Models of arm {arm} whose final model matrix is at hand")
        put(f"{pre}.distinct_vectors_median", float(r["distinct"].median()), "model_patterns",
            "pattern_id", f"Distinct presence vectors among the K features of arm {arm}, "
            "median over models")
        put(f"{pre}.distinct_vectors_min", int(r["distinct"].min()), "model_patterns",
            "pattern_id", "Fewest of those over models")
        put(f"{pre}.distinct_vectors_max", int(r["distinct"].max()), "model_patterns",
            "pattern_id", "Most of those over models")
        put(f"{pre}.features_per_vector_median", float(r["per_vector"].median()),
            "model_patterns", "pattern_id",
            f"Features per distinct presence vector of arm {arm}, median over models")
        put(f"{pre}.largest_vector_share_median", float(r["largest_share"].median()),
            "model_patterns", "pattern_id",
            f"Share (0-1) of the K features of arm {arm} that carry its most-copied vector, "
            "median over models")
        put(f"{pre}.largest_vector_share_max", float(r["largest_share"].max()),
            "model_patterns", "pattern_id", "Share (0-1) of the K features that carry the "
            "most-copied vector, largest over models")

    for arm, ls in lengths.items():
        put(f"m3.{arm}.candidate_length_median", float(pd.Series(ls).median()), "card_unitigs",
            "length", f"Length (bp) of the candidates' sequences in arm {arm}, median")
    _m3_cost_numbers(t, arms, put)
    _m3_measurement_numbers(t, put)

    labels = {"A_prefilter": "arm A inside its χ² prefilter", "B": "arm B", "C": "arm C"}
    for arm, c in got["candidates"].items():
        _m3_grade_numbers(c, f"m3.{arm}", labels[arm], put)
        if "copies" not in c or c["copies"].isna().any():
            continue
        # one signal per presence vector: a candidate's copies carry the same signal, and the
        # signal takes the best grade any of them reached
        best = (c.assign(rank=c["grade"].map(GRADE_ORDER))
                .groupby(["model", "vector"])["rank"].min().map(dict(enumerate(F.GRADES))))
        put(f"m3.{arm}.n_candidate_signals", len(best), "grades_patterns", "pattern_id",
            f"Distinct presence vectors among the candidates of arm {arm}, summed over models")
        for g in F.GRADES:
            put(f"m3.{arm}.signal.share_{g}", float((best == g).mean()), "grades_patterns",
                "grade", f"Share (0-1) of those signals whose best candidate is graded {g}")
        for name, lo, hi in M3_COPIES:
            sel = c[(c["copies"] >= lo) & ((c["copies"] <= hi) if hi else True)]
            put(f"m3.{arm}.copies_{name}.n_candidates", len(sel), "model_patterns", "pattern_id",
                f"Candidates of arm {arm} whose vector has {name.replace('_', '-')} copies "
                "among the K features")
            if len(sel):
                for layer in ("mda", "cpss"):
                    put(f"m3.{arm}.copies_{name}.share_pass_{layer}",
                        float(sel[layer].astype(bool).mean()), "grades_patterns", layer,
                        f"Share (0-1) of those candidates passing the {layer} layer")


def _kb_numbers(db: Path, put) -> None:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rel = con.execute("SELECT kb_version, schema_version FROM release").fetchone()
        if rel is not None:
            put("kb.version", rel[0], "release", "kb_version", "Version of the knowledge base",
                digits=None)
            put("kb.schema_version", rel[1], "release", "schema_version",
                "Version of its schema", digits=None)
        kinds = dict(con.execute("SELECT type, count(*) FROM sqlite_master "
                                 "WHERE name NOT LIKE 'sqlite_%' GROUP BY type").fetchall())
        put("kb.n_tables", int(kinds.get("table", 0)), "sqlite_master", "type",
            "Tables of the knowledge base")
        put("kb.n_views", int(kinds.get("view", 0)), "sqlite_master", "type",
            "Views of the knowledge base")
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                   "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
            n = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            put(f"kb.n_rows.{name}", int(n), name, "*", f"Rows of the table {name}")
        graded = pd.read_sql_query(KB_GRADED, con)
        b = pd.read_sql_query(KB_B_AROS, con)
        ctx = pd.read_sql_query(KB_CONTEXT, con)
        names = [r[0] for r in con.execute(KB_AMR_NAMES).fetchall()]
        _kb_case_numbers(con, ctx, put)
    finally:
        con.close()
    if len(b):
        _kb_card_numbers(b, put)
    if len(graded):
        _kb_context_numbers(graded, ctx, put)
        _kb_context_grade_numbers(graded, ctx, amr_name_index(names), put)


def numbers(t: Tables, organisms: list[str]) -> pd.DataFrame:
    rows: list[dict] = []

    def put(key, value, table, column, description, digits="auto"):
        # A share rounded to four decimals cannot be rendered as a percentage with one
        # decimal: 0.074471 becomes 0.0745, which reads as 7.5 % instead of 7.4 %.
        # digits=None keeps the value as it is: a p-value, already rounded to four
        # significant digits, would read 0 at four decimals.
        if digits == "auto":
            digits = 6 if description.startswith("Share (0-1)") else 4
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return
        # exact keeps a float before rounding: an article that rounds the value again (to
        # three decimals, say) would otherwise round twice, and 0.08949 stored as 0.0895
        # would read 0.090
        exact = repr(float(getattr(value, "raw", value))) if isinstance(value, float) else ""
        if isinstance(value, float):
            value = round(value, digits) if digits is not None else float(value)
        rows.append({"key": key, "value": value, "source_table": table, "column": column,
                     "description": description, "exact": exact})

    min_support = int((t.config.get("unitig") or {}).get("min_support") or 10)
    dec = t.get("panel_decisions")
    if dec is not None:
        put("panel.n_pairs", len(dec), "panel_decisions", "decision", "Organism–antibiotic pairs assessed")
        for d in ("included", "excluded", "blocked"):
            put(f"panel.n_{d}", int((dec["decision"] == d).sum()), "panel_decisions", "decision",
                f"Pairs {d}")
        # 3.1 splits the excluded pairs by why they fell out, and quotes the minority class of
        # the pairs that stayed; both were literals because no key carried them.
        exc = dec[dec["decision"] == "excluded"]
        if len(exc) and "reason" in exc:
            short = exc["reason"].str.startswith("minority")
            put("panel.n_excluded_small_minority", int(short.sum()), "panel_decisions", "reason",
                "Pairs excluded because the minority class is below the threshold")
            put("panel.n_excluded_other_reason", int((~short).sum()), "panel_decisions", "reason",
                "Pairs excluded for any other reason, e.g. a label naming no single drug")
        # The unitig/k-mer article weighs a relative frequency floor against min_support. A
        # floor of q is stricter exactly where q * n_eligible > min_support, so the number of
        # pairs it would bind is what that argument turns on.
        inc_n = pd.to_numeric(
            dec.loc[dec["decision"] == panel.INCLUDED, "n_eligible"], errors="coerce").dropna() \
            if "n_eligible" in dec else pd.Series(dtype=float)
        if len(inc_n):
            for name, q in (("1pct", 0.01), ("5pct", 0.05)):
                put(f"panel.n_pairs_relative_floor_stricter_{name}",
                    int((inc_n * q > min_support).sum()), "panel_decisions", "n_eligible",
                    f"Included pairs where a {q:.0%} relative floor binds tighter than "
                    f"min_support = {min_support}")
            put("panel.n_eligible_median", float(inc_n.median()), "panel_decisions", "n_eligible",
                "Median genomes of the included pairs after QC and lineage assignment")
            put("panel.n_eligible_min", int(inc_n.min()), "panel_decisions", "n_eligible",
                "Fewest genomes of any included pair")
            put("panel.n_eligible_max", int(inc_n.max()), "panel_decisions", "n_eligible",
                "Most genomes of any included pair")
        inc_min = pd.to_numeric(
            dec.loc[dec["decision"] == panel.INCLUDED, "minority"], errors="coerce").dropna()
        if len(inc_min):
            put("panel.median_minority_included", float(inc_min.median()), "panel_decisions",
                "minority", "Median minority class of the included pairs")
            put("panel.min_minority_included", int(inc_min.min()), "panel_decisions", "minority",
                "Smallest minority class of the included pairs")
            put("panel.max_minority_included", int(inc_min.max()), "panel_decisions", "minority",
                "Largest minority class of the included pairs")
    comparisons: list[pd.DataFrame] = []
    genomes_qc_pass = 0
    flow = {"phenotyped": 0, "downloaded": 0, "with_lineage": 0, "eligible": 0}   # the genome funnel
    for org in organisms:
        snap = t.json("metadata_file", "snapshot.json", org)
        if snap is not None:
            put(f"{org}.n_genomes_downloaded", int(snap["n_genomes"]), "snapshot", "n_genomes",
                "Phenotyped genomes whose assembly was downloaded")
            flow["downloaded"] += int(snap["n_genomes"])
        long = t.get("phenotypes_long", org)
        if long is not None:
            flow["phenotyped"] += int(long["genome_id"].nunique())
            put(f"{org}.n_genomes_phenotyped", int(long["genome_id"].nunique()), "phenotypes_long",
                "genome_id", "Genomes with a phenotype")
            put(f"{org}.n_antibiotics", int(long["antibiotic"].nunique()), "phenotypes_long",
                "antibiotic", "Antibiotics with a phenotype")
        qc = t.get("genome_qc", org)
        if qc is not None:
            n_pass = int(F._bool(qc["pass_overall"]).sum())
            put(f"{org}.n_genomes_qc_pass", n_pass, "genome_qc",
                "pass_overall", "Genomes that pass QC")
            genomes_qc_pass += n_pass
        clusters = t.get("lineage_clusters", org)
        if clusters is not None:
            n_lin = int(clusters["Genome ID"].nunique())
            put(f"{org}.n_genomes_with_lineage", n_lin, "lineage_clusters", "Genome ID",
                "Genomes assigned to a lineage (PopPUNK, after its own QC)")
            flow["with_lineage"] += n_lin
            if qc is not None:
                n_el = len(panel.eligible_genomes(qc, clusters))
                put(f"{org}.n_genomes_eligible", n_el, "genome_qc", "pass_overall",
                    "Genomes that pass QC and have a lineage")
                flow["eligible"] += n_el
        if dec is not None:
            mine = dec[dec["organism"] == org]
            if len(mine):
                put(f"{org}.n_pairs_assessed", len(mine), "panel_decisions", "decision",
                    "Pairs of this organism the panel rule assessed")
                put(f"{org}.n_pairs_included", int((mine["decision"] == panel.INCLUDED).sum()),
                    "panel_decisions", "decision", "Pairs of this organism that entered the panel")
                if "minority" in mine:
                    put(f"{org}.max_minority", int(mine["minority"].max()), "panel_decisions",
                        "minority", "Largest minority class among this organism's pairs")
                for r in mine.itertuples(index=False):
                    for col, what in (("n_eligible", "Genomes after QC and lineage assignment"),
                                      ("n_tested", "Genomes with a phenotype for this pair")):
                        v = pd.to_numeric(getattr(r, col, None), errors="coerce")
                        if pd.notna(v):
                            put(f"{r.organism}__{r.antibiotic}.{col}", int(v), "panel_decisions",
                                col, what)
        cl = t.get("lineage_clusters", org)
        if cl is not None:
            put(f"{org}.n_lineages", int(cl["Cluster"].nunique()), "lineage_clusters", "Cluster",
                "PopPUNK lineages")
        if dec is not None:
            # The lineage-aware folds never split a lineage, so a model whose largest lineage
            # holds more than a fold's share of its genomes tests that lineage as one fold.
            pairs = dec[(dec["organism"] == org) & (dec["decision"] == panel.INCLUDED)]
            shares = [d["largest_lineage_share"] for ab in pairs["antibiotic"]
                      if (d := t.json("cv_dir", "cv_design.json", org, ab)) is not None]
            if shares:
                put(f"{org}.model_largest_lineage_share_median",
                    round(float(pd.Series(shares).median()), 6),
                    "cv_design", "largest_lineage_share",
                    "Share (0-1) of a model's genomes in its largest lineage, median over the "
                    "organism's models")
                put(f"{org}.model_largest_lineage_share_max", round(float(max(shares)), 6),
                    "cv_design", "largest_lineage_share",
                    "Share (0-1) of a model's genomes in its largest lineage, largest over the "
                    "organism's models")
        comp = t.get("external_comparison", org)
        if comp is not None:
            comparisons.append(comp)
    included = (dec[(dec["decision"] == panel.INCLUDED) & dec["organism"].isin(organisms)]
                if dec is not None else pd.DataFrame(columns=["organism", "antibiotic"]))
    if dec is not None:
        put("panel.n_organisms_assessed", int(dec["organism"].nunique()),
            "panel_decisions", "organism", "Organisms the panel rule assessed")
        put("panel.n_organisms_with_models", int(included["organism"].nunique()),
            "panel_decisions", "organism", "Organisms with at least one included pair")
        put("panel.n_antibiotics_included", int(included["antibiotic"].nunique()),
            "panel_decisions", "antibiotic", "Antibiotics of the included pairs")
        if "drug_class" in included:
            put("panel.n_drug_classes_included", int(included["drug_class"].nunique()),
                "panel_decisions", "drug_class", "Drug classes of the included pairs")
    genomes_in_models: set[str] = set()
    evidence: list[pd.DataFrame] = []
    drug_class = (dict(zip(included["organism"] + "__" + included["antibiotic"],
                           included["drug_class"], strict=True))
                  if "drug_class" in included else {})
    n_candidate_patterns = 0
    n_graded_primary = 0
    n_weak_or_none = 0
    grade_counts: dict[str, int] = {g: 0 for g in F.GRADES}
    # 3.7: the determinants each path recovers, counted per model (an ARO in two models
    # counts twice), and the models where the association path adds one.
    overlap = {"candidate_only": 0, "both": 0, "association_only": 0}
    n_models_association_only = n_models_overlap = 0
    lambda_background: dict[str, float] = {}
    for org, ab in zip(included["organism"], included["antibiotic"], strict=True):
        mid = f"{org}__{ab}"
        lam = (t.json("pyseer_dir", "pyseer_summary.json", org, ab) or {}).get("lambda_background")
        if lam is not None:
            lambda_background[mid] = float(lam)
            put(f"{mid}.lambda_background", float(lam), "pyseer_summary", "lambda_background",
                "Genomic inflation of the LMM on the background sample of the model's unitigs")
        mg = t.get("model_genomes", org, ab)
        if mg is not None:
            put(f"{mid}.n_genomes", len(mg), "model_genomes", "Genome ID", "Genomes of the model")
            genomes_in_models.update(mg["Genome ID"].astype(str))
            put(f"{mid}.n_resistant", int((mg["label"] == 1).sum()), "model_genomes", "label",
                "Resistant genomes")
        rm = t.get("repeat_metrics", org, ab)
        if rm is not None:
            for arm in ("lineage_aware", "lineage_blind"):
                r = rm[rm["arm"] == arm]
                for col in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
                    if col in r and len(r):
                        put(f"{mid}.{col}.{arm}", float(r[col].mean()), "repeat_metrics", col,
                            f"Mean over repeats, {arm.replace('_', '-')} arm")
        grades = t.get("grades_patterns", org, ab)
        if grades is not None:
            put(f"{mid}.n_candidates", len(grades), "grades_patterns", "pattern_id",
                "Candidate patterns")
            n_candidate_patterns += len(grades)
            if "grade" in grades:
                n_graded_primary += len(grades)
                n_weak_or_none += int(grades["grade"].isin(("weak", "none")).sum())
                vc_primary = grades["grade"].value_counts()
                for g in F.GRADES:
                    grade_counts[g] += int(vc_primary.get(g, 0))
            for mode in ("allele_aware", "homolog_only"):
                vc = grades[f"{mode}_grade"].value_counts()
                for g in F.GRADES:
                    put(f"{mid}.n_{g}.{mode}", int(vc.get(g, 0)), "grades_patterns",
                        f"{mode}_grade", f"Patterns graded {g}, {mode.replace('_', '-')} mode")
        cpss = t.get("cpss_patterns", org, ab)
        if cpss is not None and "pi" in cpss and len(cpss):
            pi = pd.to_numeric(cpss["pi"], errors="coerce")
            put(f"{mid}.max_cpss_pi", float(pi.max()), "cpss_patterns", "pi",
                "Highest selection frequency the stability selection reached")
            put(f"{mid}.n_patterns_cpss_pi_positive", int((pi > 0).sum()), "cpss_patterns", "pi",
                "Patterns the stability selection picked at least once")
            summ = t.json("cpss_dir", "cpss_summary.json", org, ab) or {}
            if summ.get("n_fits") and "n_selected" in cpss:
                n_fits = int(summ["n_fits"])
                sel = pd.to_numeric(cpss["n_selected"], errors="coerce").fillna(0)
                put(f"{mid}.cpss_selections_per_fit", float(sel.sum() / n_fits), "cpss_patterns",
                    "n_selected", "Patterns each stability-selection fit selects, mean over the fits")
                if summ.get("n_trees") is not None:
                    put(f"{mid}.cpss_n_trees", int(summ["n_trees"]), "cpss_summary", "n_trees",
                        "Trees of each stability-selection fit")
                if float(pi.max()) < float(t.config["cpss"]["pi_threshold"]):
                    aro = {"dtype": {"allele_aware_aros": str}}
                    _cpss_split_numbers(
                        mid, cpss, n_fits,
                        [t.get(x, org, ab, **aro) for x in ("card_patterns", "association_card_patterns")],
                        [g for g in (grades, t.get("association_grades_patterns", org, ab)) if g is not None],
                        put)
        cands = t.get("candidates", org, ab)
        if grades is not None and cands is not None and "grade" in grades:
            e = grades.merge(cands, on="pattern_id", how="left")
            prev = t.get("prevalence_layer", org, ab)
            if prev is not None and "direction" in prev:
                e = e.merge(prev[["pattern_id", "direction"]], on="pattern_id", how="left")
            evidence.append(e.assign(model=mid, organism=org,
                                     drug_class=drug_class.get(mid, "unmapped")))
        assoc = t.get("association_grades_patterns", org, ab)
        if assoc is not None:
            put(f"{mid}.association.n_patterns", len(assoc), "association_grades_patterns",
                "pattern_id", "Patterns nominated by association (secondary analysis)")
            vc = assoc["grade"].value_counts()
            for g in F.GRADES:
                put(f"{mid}.association.n_{g}", int(vc.get(g, 0)), "association_grades_patterns",
                    "grade", f"Patterns nominated by association graded {g}, primary mode")
            aros = {"dtype": {"allele_aware_aros": str}}
            cand_b = _b_aros(t.get("card_unitigs", org, ab, **aros))
            assoc_b = _b_aros(t.get("association_card_unitigs", org, ab, **aros))
            overlap["candidate_only"] += len(cand_b - assoc_b)
            overlap["both"] += len(cand_b & assoc_b)
            overlap["association_only"] += len(assoc_b - cand_b)
            n_models_association_only += bool(assoc_b - cand_b)
            n_models_overlap += 1
    if n_models_overlap:
        for k, what in (("candidate_only", "only among the candidates"),
                        ("both", "on both paths"),
                        ("association_only", "only on the association path")):
            put(f"assoc_overlap.n_{k}", overlap[k], "card_unitigs", "allele_aware_aros",
                f"(model, ARO) pairs giving b = 1 {what}, primary mode, summed over models")
        put("assoc_overlap.n_models_association_only", n_models_association_only,
            "association_card_unitigs", "allele_aware_aros",
            "Models where the association path recovers an ARO the candidates miss")
    if evidence:
        ev = pd.concat(evidence, ignore_index=True)
        _evidence_numbers(ev, put)
        lp = t.get("label_permutation_models")
        flagged = (_flagged_models(lp, sorted(ev["model"].unique()),
                                   float(t.config["label_permutation"]["alpha"]))
                   if lp is not None else None)
        if flagged is not None:
            _permutation_sensitivity(ev, flagged, put)
        tier_table = card_tier_table(t, sorted({(m.split("__")[0], m.split("__", 1)[1])
                                                for m in ev["model"]}))
        if len(tier_table):
            _card_tier_numbers(tier_table, put)
    # 3.1 quotes the spread of model performance and of the gap between the two arms. Both are
    # medians over models, so they belong here rather than in any one model's keys.
    auc: dict[str, dict[str, float]] = {"lineage_aware": {}, "lineage_blind": {}}
    for row in rows:
        for arm in auc:
            if row["key"].endswith(f".roc_auc.{arm}"):
                auc[arm][row["key"].rsplit(".", 2)[0]] = row["value"]
    if auc["lineage_aware"]:
        a_vals = pd.Series(auc["lineage_aware"])
        put("models.roc_auc_median.lineage_aware", float(a_vals.median()), "repeat_metrics",
            "roc_auc", "Median over models of the lineage-aware ROC-AUC")
        put("models.roc_auc_min.lineage_aware", float(a_vals.min()), "repeat_metrics", "roc_auc",
            "Lowest lineage-aware ROC-AUC of any model")
        put("models.roc_auc_max.lineage_aware", float(a_vals.max()), "repeat_metrics", "roc_auc",
            "Highest lineage-aware ROC-AUC of any model")
        shared = sorted(set(auc["lineage_aware"]) & set(auc["lineage_blind"]))
        if shared:
            gap = pd.Series({m: auc["lineage_blind"][m] - auc["lineage_aware"][m]
                             for m in shared})
            put("models.arm_gap_median", float(gap.median()), "repeat_metrics", "roc_auc",
                "Median over models of the random-split minus lineage-aware ROC-AUC")
            put("models.arm_gap_max", float(gap.max()), "repeat_metrics", "roc_auc",
                "Largest such gap")
            put("models.arm_gap_min", float(gap.min()), "repeat_metrics", "roc_auc",
                "Smallest such gap; above zero means no model is an exception")
            put("models.n_arm_gap_positive", int((gap > 0).sum()), "repeat_metrics", "roc_auc",
                "Models where the random split gives the higher ROC-AUC")
            by_organism = {}
            for org in organisms:
                mine = gap[[m for m in gap.index if m.startswith(f"{org}__")]]
                if len(mine):
                    by_organism[org] = float(mine.median())
                    put(f"{org}.arm_gap_median", float(mine.median()), "repeat_metrics",
                        "roc_auc", "Median arm gap over this organism's models")
                    put(f"{org}.arm_gap_max", float(mine.max()), "repeat_metrics", "roc_auc",
                        "Largest arm gap among this organism's models")
                own = pd.Series({m: v for m, v in auc["lineage_aware"].items()
                                 if m.startswith(f"{org}__")})
                if len(own):
                    put(f"{org}.roc_auc_median.lineage_aware", float(own.median()),
                        "repeat_metrics", "roc_auc",
                        "Median lineage-aware ROC-AUC of this organism's models")
            if by_organism:
                put("models.arm_gap_median_by_organism_min", min(by_organism.values()),
                    "repeat_metrics", "roc_auc",
                    "Smallest of the per-organism median arm gaps")
                put("models.arm_gap_median_by_organism_max", max(by_organism.values()),
                    "repeat_metrics", "roc_auc", "Largest of the per-organism median arm gaps")
    # 3.5: a candidate failing the LMM layer is not evidence of no association when the layer is
    # conservative, which a background lambda below 1 suggests.
    if lambda_background:
        lam_s = pd.Series(lambda_background)
        put("models.lambda_background_median", float(lam_s.median()), "pyseer_summary",
            "lambda_background", "Median over models of the background genomic inflation")
        put("models.lambda_background_min", float(lam_s.min()), "pyseer_summary",
            "lambda_background", "Lowest background genomic inflation of any model")
        put("models.lambda_background_max", float(lam_s.max()), "pyseer_summary",
            "lambda_background", "Highest background genomic inflation of any model")
        put("models.n_lambda_background", len(lam_s), "pyseer_summary", "lambda_background",
            "Models with a background genomic inflation")
        put("models.n_lambda_background_below_1", int((lam_s < 1).sum()), "pyseer_summary",
            "lambda_background", "Models whose background genomic inflation is below 1")
    if genomes_in_models:
        put("all.n_genomes_in_models", len(genomes_in_models), "model_genomes", "Genome ID",
            "Distinct genomes entering at least one model")
    if n_candidate_patterns:
        put("all.n_candidate_patterns", n_candidate_patterns, "grades_patterns", "pattern_id",
            "Candidate patterns, summed over models (pattern numbers are model-specific)")
    if n_graded_primary:
        put("all.weak_or_none_share", n_weak_or_none / n_graded_primary, "grades_patterns",
            "grade", "Share (0-1) of the candidate patterns graded weak or none, primary mode")
        for g in F.GRADES:
            put(f"all.n_{g}", grade_counts[g], "grades_patterns", "grade",
                f"Candidate patterns graded {g}, summed over models, primary mode")
            put(f"all.share_{g}", grade_counts[g] / n_graded_primary, "grades_patterns", "grade",
                f"Share (0-1) of the candidate patterns graded {g}, primary mode")
    if genomes_qc_pass:
        put("all.n_genomes_qc_pass", genomes_qc_pass, "genome_qc", "pass_overall",
            "Genomes that pass QC, summed over organisms")
    flow_source = {"phenotyped": ("phenotypes_long", "genome_id", "Genomes with a phenotype"),
                   "downloaded": ("snapshot", "n_genomes",
                                  "Phenotyped genomes whose assembly was downloaded"),
                   "with_lineage": ("lineage_clusters", "Genome ID",
                                    "Genomes assigned to a lineage (PopPUNK, after its own QC)"),
                   "eligible": ("genome_qc", "pass_overall", "Genomes that pass QC and have a lineage")}
    for name, n in flow.items():
        if n:
            src, col, desc = flow_source[name]
            put(f"all.n_genomes_{name}", n, src, col, f"{desc}, summed over organisms")
    if comparisons:
        comp_all = pd.concat(comparisons, ignore_index=True)
        ba = comp_all.pivot_table(index="model_id", columns="tool",
                                  values="balanced_accuracy", aggfunc="first")
        ok = comp_all.pivot_table(index="model_id", columns="tool",
                                  values="assessable", aggfunc="first")
        if "model" in ba and "model" in ok:
            for tool in sorted(c for c in ba.columns if c != "model"):
                both = (ok["model"] == 1) & (ok[tool] == 1)
                d = (ba.loc[both, "model"] - ba.loc[both, tool]).dropna()
                s = concordance.sign_test(d)
                pre = f"compare.{tool}"
                put(f"{pre}_n_assessable", int(both.sum()), "external_comparison", "assessable",
                    f"Pairs both the model and {tool} can assess")
                put(f"{pre}_n_model_better", s["n_positive"], "external_comparison",
                    "balanced_accuracy", f"Pairs where the model beats {tool}")
                put(f"{pre}_n_tool_better", s["n_negative"], "external_comparison",
                    "balanced_accuracy", f"Pairs where {tool} beats the model")
                put(f"{pre}_n_tied", s["n_tied"], "external_comparison", "balanced_accuracy",
                    f"Pairs where the model and {tool} score the same (dropped by the sign test)")
                put(f"{pre}_p", _four_significant(s["p_value"]), "external_comparison",
                    "balanced_accuracy",
                    f"Two-sided sign test of balanced accuracy, model vs {tool}, ties dropped",
                    digits=None)
                # the same test with the ties counted for one side: 3.5 reports how far the
                # result turns on a convention, so both alternatives need a key of their own
                put(f"{pre}_p_tie_to_model", _four_significant(s["p_tie_to_ours"]),
                    "external_comparison", "balanced_accuracy",
                    "That sign test with a tie counted for the model, not dropped", digits=None)
                put(f"{pre}_p_tie_to_tool", _four_significant(s["p_tie_to_theirs"]),
                    "external_comparison", "balanced_accuracy",
                    f"That sign test with a tie counted for {tool}, not dropped", digits=None)
                put(f"{pre}_median_ba_model", float(ba.loc[both, "model"].median()),
                    "external_comparison", "balanced_accuracy",
                    f"Median balanced accuracy of the model, pairs shared with {tool}")
                put(f"{pre}_median_ba_tool", float(ba.loc[both, tool].median()),
                    "external_comparison", "balanced_accuracy",
                    f"Median balanced accuracy of {tool} on those pairs")
                pd_ = concordance.paired_difference(
                    d, [i.split("__")[0] for i in d.index],
                    n_boot=int(t.config["cv"]["n_bootstrap"]))
                put(f"{pre}_median_diff", pd_["median"], "external_comparison",
                    "balanced_accuracy",
                    f"Median paired difference in balanced accuracy, model minus {tool}")
                put(f"{pre}_diff_ci_low", pd_["ci_low"], "external_comparison",
                    "balanced_accuracy",
                    "Lower 95% bound of that difference, organism-clustered bootstrap (§6.3)")
                put(f"{pre}_diff_ci_high", pd_["ci_high"], "external_comparison",
                    "balanced_accuracy",
                    "Upper 95% bound of that difference, organism-clustered bootstrap (§6.3)")
                put(f"{pre}_n_organisms", pd_["n_clusters"], "external_comparison", "model_id",
                    f"Organisms contributing a pair shared with {tool}")
                put(f"{pre}_p_signed_rank",
                    _four_significant(concordance.signed_rank_p(d)), "external_comparison",
                    "balanced_accuracy",
                    f"Signed-rank p, model vs {tool}; sensitivity only, pairs not independent",
                    digits=None)
                rows_t = comp_all[(comp_all["tool"] == tool) & (comp_all["assessable"] == 1)]
                if {"sensitivity", "specificity"} <= set(rows_t.columns):
                    put(f"{pre}_n_calls_all_resistant",
                        int(((rows_t["sensitivity"] == 1) & (rows_t["specificity"] == 0)).sum()),
                        "external_comparison", "sensitivity",
                        f"Models where {tool} calls every genome resistant")
                gram = rows_t["model_id"].str.split("__").str[0].map(_gram_stain)
                for stain in ("positive", "negative"):
                    sel = rows_t[gram == stain]
                    if len(sel):
                        put(f"{pre}_median_ba_tool_gram_{stain}",
                            float(sel["balanced_accuracy"].median()), "external_comparison",
                            "balanced_accuracy",
                            f"Median balanced accuracy of {tool}, Gram-{stain} organisms")
    lp = t.get("label_permutation_models")
    if lp is not None and "q" in lp:
        put("label_permutation.n_significant", int((lp["q"] < float(
            t.config["label_permutation"]["alpha"])).sum()), "label_permutation_models", "q",
            "Models whose label permutation q is below α")
        put("label_permutation.n_flagged", int((lp["q"] >= float(
            t.config["label_permutation"]["alpha"])).sum()), "label_permutation_models", "q",
            "Models whose q is at or above α: flagged, kept in every analysis (§8.6)")
    hyp = t.json("cross_model_dir", "hypotheses/hypotheses.json")
    if hyp is not None:
        _hypothesis_numbers(hyp, put, "hyp")
        put("hyp.n_organisms_without_reference", len(hyp.get("reference_missing") or []),
            "h7_determinants", "model_id",
            "Organisms without AMRFinderPlus calls, left out of H7 and H6")
        sens = hyp.get("sensitivity_label_permutation") or {}
        if sens.get("status") == "computed":
            put("lp_sensitivity.hyp.n_models_excluded", len(sens["excluded_models"]),
                "h1_models", "model_id", "Models the hypotheses are recomputed without (§14 item 9)")
            _hypothesis_numbers(sens, put, "lp_sensitivity.hyp", "Without the flagged models. ")
    try:
        kb: Path | None = contract.output_path("kb_dir", "kanit.sqlite", t.config)
    except KeyError:            # a configuration without kb_dir
        kb = None
    if kb is not None and kb.exists():
        _kb_numbers(kb, put)
    _cabbage_numbers(t, organisms, put)
    _m3_numbers(t, _m3_arm_tables(t), organisms, put)
    return pd.DataFrame(rows, columns=["key", "value", "source_table", "column", "description",
                                       "exact"])


def main():
    ap = argparse.ArgumentParser(description="Figures, model reports and the numbers file.")
    ap.add_argument("--entry", choices=["main", "DOWNLOAD", "CONTEXT", "KB", "CABBAGE"], required=True)
    ap.add_argument("--organisms", required=True)
    args = ap.parse_args()
    config = load_config()
    organisms = args.organisms.split(",")
    t = Tables(config)
    out = resolve_path("reports_dir", config=config)
    n_figs = len(global_figures(t, out))
    for org in organisms:
        n_figs += len(organism_figures(t, org, out / org))
    n_reports = 0
    dec = t.get("panel_decisions")
    if args.entry == "main" and dec is not None:
        inc = dec[(dec["decision"] == panel.INCLUDED) & dec["organism"].isin(organisms)]
        for org, ab in zip(inc["organism"], inc["antibiotic"], strict=True):
            mdir = resolve_path("model_reports_dir", organism=org, antibiotic=ab, config=config)
            figs = model_figures(t, org, ab, mdir)
            n_figs += len(figs)
            model_report(t, org, ab, figs, mdir)
            n_reports += 1
    nums = numbers(t, organisms)
    out.mkdir(parents=True, exist_ok=True)
    nums.to_csv(out / "tez_sayilari.csv", index=False)
    (out / "reports_summary.json").write_text(json.dumps({
        "entry": args.entry, "figures": n_figs, "model_reports": n_reports,
        "numbers": len(nums), "skipped": t.skipped}, indent=2) + "\n")
    print(f"REPORTS — {args.entry}: {n_figs} figures, {n_reports} model reports, "
          f"{len(nums)} numbers")


if __name__ == "__main__":
    main()
