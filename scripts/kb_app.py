#!/usr/bin/env python3
"""
KANIT explorer: a local, interactive interface to the knowledge base (kanit.sqlite).

Choose an organism–antibiotic model to see its performance, its evidence-graded
biomarkers, the evidence chain of each biomarker, its genomic context in NCBI and its
sequences. Search by gene or product name, unitig identifier or DNA sequence. When the
CABBAGE results are available, the comparison on independent isolates is shown as well.

Runs locally (not part of the workflow):
    pip install streamlit pandas altair
    streamlit run scripts/kb_app.py -- --db <kanit.sqlite> [--results <results directory>]
Without --db, the KANIT_DB environment variable is read, and without that
results/kb/kanit.sqlite. --results is the directory holding the CABBAGE files
(results/<organism>/cabbage/); without it the first `results` directory above the
database is tried. The database is opened read-only.
"""

import argparse
import glob
import json
import os
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KANIT", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--db", default=os.environ.get("KANIT_DB", str(PROJECT_ROOT / "results" / "kb" / "kanit.sqlite")))
_ap.add_argument("--results", default=os.environ.get("KANIT_RESULTS"))
ARGS, _ = _ap.parse_known_args()

ORG_NAME = {"enterococcus_faecium": "E. faecium", "staphylococcus_aureus": "S. aureus",
            "kpneumoniae": "K. pneumoniae", "acinetobacter_baumannii": "A. baumannii",
            "pseudomonas_aeruginosa": "P. aeruginosa", "enterobacter_cloacae": "E. cloacae complex",
            "ecoli": "E. coli"}
ORG_ORDER = list(ORG_NAME)
GRADES = ["confirmed", "strong_novel", "candidate", "weak", "none"]
GRADE_COLOUR = {"confirmed": "#1c5cab", "strong_novel": "#eb6834", "candidate": "#3987e5",
                "weak": "#86b6ef", "none": "#bdbdbd"}
GRADE_MEANING = {
    "confirmed": "matches a curated determinant (b = 1) and at least two statistical layers support it",
    "strong_novel": "no CARD hit; passes stability selection, the mixed model and at least one more layer",
    "candidate": "the evidence is incomplete: concordant but at most one layer, or layers but no concordance "
                 "(with the reason), or no CARD hit but the layers do not make the strong_novel combination",
    "weak": "no concordance and exactly one layer passes",
    "none": "no layer passes; no evidence beyond the model's importance ranking"}
TOOL_NAME = {"amrfinderplus": "AMRFinderPlus", "resfinder": "ResFinder", "rgi_all": "RGI (all hits)",
             "rgi_without_near_universal": "RGI (without near-universal genes)", "model": "KANIT model"}
SEARCH_NAME = {"upto_confirmed": "KANIT search: confirmed only",
               "upto_strong_novel": "KANIT search: up to strong_novel (primary)",
               "upto_candidate": "KANIT search: up to candidate", "upto_weak": "KANIT search: up to weak",
               "b1": "KANIT search: b = 1 only"}


def ab_name(ab: str) -> str:
    return ab.replace("_", " ")


def grade_rank(g: str) -> int:
    return GRADES.index(g) if g in GRADES else len(GRADES)


# --- data -------------------------------------------------------------------------------------
@st.cache_resource
def connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def query(path: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, connect(path), params=params)


def results_directory(db: Path) -> Path | None:
    if ARGS.results:
        return Path(ARGS.results)
    for p in db.resolve().parents:
        if p.name == "results":
            return p
    return None


@st.cache_data(show_spinner=False)
def cabbage(results: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Results on independent isolates: the tools and the model (comparison table), and the KANIT search."""
    comparison = [pd.read_csv(f) for f in glob.glob(f"{results}/*/cabbage/cabbage_comparison.csv")]
    rows = []
    for f in glob.glob(f"{results}/*/cabbage/cabbage_lookup_metrics.json"):
        j = json.load(open(f))
        for ab, v in j["pairs"].items():
            for rule, name in SEARCH_NAME.items():
                a = v.get(rule, {}).get("all", {})
                if a.get("assessed"):
                    rows.append(dict(model_id=f"{j['organism']}__{ab}", predictor=name, n=a["n"],
                                     n_resistant=a["n_resistant"], sensitivity=a["sensitivity"],
                                     specificity=a["specificity"], balanced_accuracy=a["balanced_accuracy"]))
    return (pd.concat(comparison, ignore_index=True) if comparison else pd.DataFrame()), pd.DataFrame(rows)


# --- sidebar ----------------------------------------------------------------------------------
st.sidebar.title("KANIT")
db = Path(st.sidebar.text_input("Database (kanit.sqlite)", ARGS.db))
if not db.exists():
    st.error(f"Database not found: {db}")
    st.stop()
DB = str(db)
rel = query(DB, "SELECT * FROM release").iloc[0]
st.sidebar.caption(f"Version {rel.kb_version} · CARD {rel.card_version} · {str(rel.created_at)[:10]}")

models = query(DB, "SELECT * FROM v_model")
orgs = sorted(models.organism_id.unique(), key=lambda o: ORG_ORDER.index(o) if o in ORG_ORDER else 99)
org = st.sidebar.selectbox("Organism", orgs, format_func=lambda o: ORG_NAME.get(o, o),
                           index=orgs.index("enterococcus_faecium") if "enterococcus_faecium" in orgs else 0)
antibiotics = sorted(models[models.organism_id == org].antibiotic_id)
ab = st.sidebar.selectbox("Antibiotic", antibiotics, format_func=ab_name)
MODEL = f"{org}__{ab}"
st.sidebar.markdown("**Grades**")
for g in GRADES:
    st.sidebar.markdown(f"<span style='display:inline-block;width:0.7em;height:0.7em;margin-right:0.3em;"
                        f"background:{GRADE_COLOUR[g]}'></span>`{g}`: {GRADE_MEANING[g]}",
                        unsafe_allow_html=True)
st.sidebar.caption("A grade states an association, not causation. Direction R: more frequent in resistant "
                   "genomes; S: more frequent in susceptible genomes.")

st.title("KANIT: Knowledgebase for Antimicrobial Resistance from machine-learned uNitigs with "
         "Integrated evidence Tiering")
st.caption("An evidence-graded knowledge base of machine-learned AMR biomarkers in the ESKAPEE pathogens")
# One page instead of tabs: tables in hidden tabs measure their width wrongly.
PAGES = ["Overview", "Model", "Biomarker", "Search", "Independent isolates (CABBAGE)"]
PAGE = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()

# --- 1. overview ------------------------------------------------------------------------------
if PAGE == "Overview":
    c = st.columns(5)
    c[0].metric("Models", len(models))
    c[1].metric("Organisms", models.organism_id.nunique())
    c[2].metric("Candidate biomarkers", int(models.n_candidates.sum()))
    c[3].metric("confirmed", int(models.n_confirmed.sum()))
    c[4].metric("strong_novel", int(models.n_strong_novel.sum()))

    st.subheader("Which organism has a model for which antibiotic")
    st.caption("Cell: lineage-aware ROC-AUC. Empty cell: no model for that pair.")
    heat = models.assign(organism=models.organism_id.map(ORG_NAME), antibiotic=models.antibiotic_id.map(ab_name),
                         auc=models.roc_auc_lineage_aware.round(2))
    org_order = [ORG_NAME[o] for o in ORG_ORDER if o in set(models.organism_id)]
    base = alt.Chart(heat).encode(
        x=alt.X("antibiotic:N", title=None, sort="ascending",
              axis=alt.Axis(labelAngle=-60, labelOverlap=False, labelLimit=160)),
        y=alt.Y("organism:N", title=None, sort=org_order))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("auc:Q", title="ROC-AUC", scale=alt.Scale(scheme="blues", domain=[0.5, 1.0])),
        tooltip=["organism", "antibiotic", alt.Tooltip("auc:Q", title="ROC-AUC (lineage-aware)"),
                 alt.Tooltip("n_genomes:Q", title="genomes"), alt.Tooltip("n_resistant:Q", title="resistant"),
                 alt.Tooltip("n_confirmed:Q", title="confirmed"), alt.Tooltip("n_strong_novel:Q", title="strong_novel")])
    labels = base.mark_text(fontSize=10).encode(
        text=alt.Text("auc:Q", format=".2f"),
        color=alt.condition("datum.auc > 0.85", alt.value("white"), alt.value("black")))
    st.altair_chart((cells + labels).properties(height=330), use_container_width=True)

    st.subheader("Per organism")
    per_org = (models.groupby("organism_id")
               .agg(models=("model_id", "count"), median_genomes=("n_genomes", "median"),
                    median_roc_auc=("roc_auc_lineage_aware", "median"), candidates=("n_candidates", "sum"),
                    confirmed=("n_confirmed", "sum"), strong_novel=("n_strong_novel", "sum"))
               .reindex([o for o in ORG_ORDER if o in set(models.organism_id)]))
    per_org.index = [ORG_NAME.get(o, o) for o in per_org.index]
    st.dataframe(per_org.style.format({"median_genomes": "{:.0f}", "median_roc_auc": "{:.2f}"}),
                 use_container_width=True)

# --- 2. model ---------------------------------------------------------------------------------
# NCBI context: the best hit of the pattern's longest searched unitig (the rule of v_novel, but
# for every grade, not only strong_novel). Gene and product names are filled when the hit's
# region is annotated as a CDS or gene; otherwise the record title and plasmid share still show.
biomarkers = query(DB, """WITH c AS (
                    SELECT pm.pattern_id, x.gene, x.product, x.best_title, x.plasmid_share,
                           row_number() OVER (PARTITION BY pm.pattern_id ORDER BY u.length DESC, u.unitig_id) AS rn
                      FROM pattern_member pm JOIN unitig u ON u.unitig_id = pm.unitig_id
                      JOIN unitig_context x ON x.unitig_id = pm.unitig_id AND x.organism_id = ?
                     WHERE pm.model_id = ?)
                  SELECT b.*, c.gene AS context_gene, c.product AS context_product,
                         c.best_title AS context_title, c.plasmid_share AS context_plasmid_share
                    FROM v_biomarker b LEFT JOIN c ON c.pattern_id = b.pattern_id AND c.rn = 1
                   WHERE b.model_id = ? AND b.route = 'model'""", (org, MODEL, MODEL))
biomarkers["_rank"] = biomarkers.grade.map(grade_rank)
biomarkers = biomarkers.sort_values(["_rank", "pi"], ascending=[True, False])

if PAGE == "Model":
    m = models[models.model_id == MODEL].iloc[0]
    st.header(f"{ORG_NAME.get(org, org)} – {ab_name(ab)}")
    c = st.columns(7)
    c[0].metric("Genomes", int(m.n_genomes))
    c[1].metric("Resistant", int(m.n_resistant))
    c[2].metric("Susceptible", int(m.n_susceptible))
    c[3].metric("Lineages", int(m.n_lineages))
    c[4].metric("ROC-AUC, lineage-aware", f"{m.roc_auc_lineage_aware:.2f}",
                help=f"95% interval {m.roc_auc_lineage_aware_low:.2f}–{m.roc_auc_lineage_aware_high:.2f}")
    c[5].metric("ROC-AUC, random split", f"{m.roc_auc_lineage_blind:.2f}",
                delta=f"{m.roc_auc_lineage_blind - m.roc_auc_lineage_aware:+.2f}", delta_color="off",
                help="A split that ignores lineages; the difference is how much lineages inflate the performance.")
    c[6].metric("Permutation q", f"{m.label_permutation_q:.3g}",
                help="Label permutation: is the model better than chance? Yes if q < 0.05.")
    if isinstance(m.flag, str) and m.flag:
        st.warning(f"Flag: {m.flag}")

    st.subheader("Biomarkers by grade and direction")
    counts = biomarkers.groupby(["grade", "direction"]).size().reset_index(name="n")
    if len(counts):
        counts["order"] = counts.grade.map(grade_rank)
        counts["dir"] = counts.direction.map({"R": "R (more frequent in resistant)",
                                              "S": "S (more frequent in susceptible)"})
        chart = alt.Chart(counts).mark_bar(stroke="white", strokeWidth=1).encode(
            x=alt.X("n:Q", title="biomarkers", stack="zero"),
            y=alt.Y("dir:N", title=None),
            color=alt.Color("grade:N", sort=GRADES, title="grade",
                            scale=alt.Scale(domain=GRADES, range=[GRADE_COLOUR[g] for g in GRADES]),
                            legend=alt.Legend(orient="bottom", direction="horizontal")),
            order=alt.Order("order:Q"), tooltip=["grade", "direction", "n"]).properties(height=alt.Step(38))
        st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)
    pick_grades = c1.multiselect("Grade", GRADES, default=GRADES)
    pick_dirs = c2.multiselect("Direction", ["R", "S"], default=["R", "S"])
    table = biomarkers[biomarkers.grade.isin(pick_grades) & biomarkers.direction.isin(pick_dirs)]
    shown = table[["pattern_id", "grade", "direction", "n_members", "layers_passed", "delta", "pi",
                   "pyseer_p", "card_genes", "card_state_allele_aware", "context_gene", "context_product",
                   "context_plasmid_share", "context_title", "gain_rank"]].rename(columns={
        "pattern_id": "pattern", "n_members": "unitigs", "layers_passed": "layers passed",
        "delta": "Δ (R−S)", "pi": "π", "pyseer_p": "pyseer p", "card_genes": "CARD genes",
        "card_state_allele_aware": "CARD state", "context_gene": "NCBI gene", "context_product": "NCBI product",
        "context_plasmid_share": "plasmid share", "context_title": "closest NCBI record", "gain_rank": "gain rank"})
    st.caption("Select a row and open the 'Biomarker' section to see that biomarker's evidence chain. "
               "NCBI gene and product names are filled only when the hit's region is annotated as a gene; "
               "the closest record and the plasmid share are shown for every candidate. The NCBI context is "
               "not evidence; it is shown alongside.")
    event = st.dataframe(shown, hide_index=True, use_container_width=True, on_select="rerun",
                         selection_mode="single-row",
                         column_config={"Δ (R−S)": st.column_config.NumberColumn(format="%.2f"),
                                        "π": st.column_config.NumberColumn(format="%.2f"),
                                        "pyseer p": st.column_config.NumberColumn(format="%.1e"),
                                        "plasmid share": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                         format="%.2f")})
    if event.selection.rows:
        st.session_state["pattern"] = int(table.iloc[event.selection.rows[0]].pattern_id)

    training = query(DB, "SELECT * FROM external_comparison WHERE model_id = ?", (MODEL,))
    if len(training):
        st.subheader("Catalogue-based tools on the training genomes (out of fold)")
        e = training[training.assessable == 1].copy()
        e["tool"] = e.tool.map(TOOL_NAME).fillna(e.tool)
        st.dataframe(e[["tool", "n", "n_resistant", "sensitivity", "specificity", "balanced_accuracy"]]
                     .rename(columns={"n_resistant": "resistant", "balanced_accuracy": "balanced accuracy"})
                     .style.format(precision=2), hide_index=True, use_container_width=True)

# --- 3. biomarker -----------------------------------------------------------------------------
if PAGE == "Biomarker":
    if biomarkers.empty:
        st.info("This model has no biomarkers.")
    else:
        options = list(biomarkers.pattern_id.astype(int))
        default = st.session_state.get("pattern")
        idx = options.index(default) if default in options else 0
        label = {int(r.pattern_id): f"{int(r.pattern_id)} · {r.grade} · {r.direction}"
                                    + (f" · {r.card_genes}" if r.card_genes else "")
                                    + (f" · {r.context_gene}" if isinstance(r.context_gene, str) and r.context_gene else "")
                 for r in biomarkers.itertuples()}
        pid = st.selectbox("Pattern", options, index=idx, format_func=lambda p: label[p])
        r = biomarkers[biomarkers.pattern_id == pid].iloc[0]
        st.markdown(f"### Grade: <span style='color:{GRADE_COLOUR.get(r.grade, '#000')}'>`{r.grade}`</span>"
                    f" · direction {r.direction}", unsafe_allow_html=True)
        st.caption(GRADE_MEANING.get(r.grade, ""))

        chain = query(DB, """
            SELECT 'differential prevalence (Δ = R − S)' AS layer, printf('%.3f', delta) AS value,
                   printf('q = %.1e', q) AS significance, passes FROM prevalence_result WHERE model_id = ? AND pattern_id = ?
            UNION ALL SELECT 'permutation importance (drop in ROC-AUC)', printf('%.4f', mda), printf('q = %.1e', q), passes
                   FROM mda_result WHERE model_id = ? AND pattern_id = ?
            UNION ALL SELECT 'stability selection (π)', printf('%.3f', pi), 'threshold π ≥ 0.6', passes
                   FROM cpss_result WHERE model_id = ? AND pattern_id = ?
            UNION ALL SELECT 'linear mixed model (β)', printf('%.3f', beta), printf('p = %.1e', lrt_p), passes
                   FROM pyseer_result WHERE model_id = ? AND pattern_id = ?""", (MODEL, int(pid)) * 4)
        chain["result"] = chain.passes.map({1: "passed", 0: "not passed"}).fillna("not tested")
        c1, c2 = st.columns([3, 2])
        c1.markdown("**Statistical evidence layers**")
        c1.table(chain[["layer", "value", "significance", "result"]].set_index("layer"))
        c2.markdown("**Concordance with curated determinants (CARD)**")
        c2.markdown(f"- State: `{r.card_state_allele_aware}`\n"
                    f"- CARD genes: {r.card_genes or '—'}\n"
                    f"- Reason: `{r.card_reasons_allele_aware or '—'}`\n"
                    f"- Route into the candidate set: {r.candidate_source}"
                    + (f" (gain rank {int(r.gain_rank)})" if pd.notna(r.gain_rank) else ""))

        unitigs = query(DB, """SELECT u.unitig_id, u.length, u.sequence, c.gene, c.product, c.best_title,
                                      c.best_identity, c.best_coverage, c.plasmid_share
                                 FROM pattern_member pm JOIN unitig u USING (unitig_id)
                                 LEFT JOIN unitig_context c ON c.unitig_id = u.unitig_id AND c.organism_id = ?
                                WHERE pm.model_id = ? AND pm.pattern_id = ? ORDER BY u.length DESC""",
                        (org, MODEL, int(pid)))
        context = unitigs.dropna(subset=["best_title"])
        st.markdown("**Closest sequence in NCBI (longest unitig)**")
        if len(context):
            b0 = context.iloc[0]
            st.markdown(f"- Gene: `{b0['gene'] or '—'}` · Product: {b0['product'] or '—'}\n"
                        f"- Record: {b0['best_title']}\n"
                        f"- Identity {b0['best_identity']:.1f}% · coverage {100 * b0['best_coverage']:.0f}% · "
                        f"plasmid share of the NCBI hits {b0['plasmid_share']:.2f}")
        else:
            st.caption("The NCBI context of this pattern was not searched.")
        st.markdown(f"**Unitigs ({len(unitigs)})**")
        st.table(unitigs[["unitig_id", "length", "gene", "product"]].fillna("").rename(
            columns={"gene": "NCBI gene", "product": "NCBI product"}).set_index("unitig_id"))
        if len(unitigs):
            st.markdown("**Sequence of the longest unitig**")
            st.code(unitigs.iloc[0]["sequence"], language=None, wrap_lines=True)
            elsewhere = query(DB, f"""SELECT model_id, grade, card_state, unitig_id FROM v_unitig
                                      WHERE unitig_id IN ({','.join('?' * len(unitigs))}) AND model_id <> ?
                                        AND primary_mode = 1""", (*unitigs.unitig_id, MODEL))
            if len(elsewhere):
                st.markdown("**The same unitigs in other models**")
                st.table(elsewhere.drop_duplicates(["model_id", "grade"]).set_index("model_id"))

# --- 4. search --------------------------------------------------------------------------------
if PAGE == "Search":
    kind = st.radio("Search by", ["Gene or product name", "Unitig identifier", "DNA sequence"], horizontal=True)
    q = st.text_input("Query", placeholder={"Gene or product name": "e.g. KPC, mecA, gyrA, transposase",
                                            "Unitig identifier": "e.g. KUc0f88eb591ee5a2551f2",
                                            "DNA sequence": "at least 15 bases"}[kind]).strip()
    if q:
        if kind == "Gene or product name":
            hits = query(DB, """SELECT DISTINCT b.model_id, b.pattern_id, b.grade, b.direction, b.card_genes,
                                       n.context_gene, n.context_product
                                  FROM v_biomarker b LEFT JOIN v_novel n
                                    ON n.model_id = b.model_id AND n.pattern_id = b.pattern_id
                                 WHERE b.route = 'model' AND (b.card_genes LIKE ? OR n.context_gene LIKE ?
                                       OR n.context_product LIKE ?)""", (f"%{q}%",) * 3)
        elif kind == "Unitig identifier":
            hits = query(DB, """SELECT model_id, pattern_id, grade, card_state, length FROM v_unitig
                                 WHERE unitig_id = ? AND primary_mode = 1""", (q,))
        else:
            s = q.upper().replace(" ", "")
            if len(s) < 15:
                st.warning("Enter at least 15 bases.")
                hits = pd.DataFrame()
            else:
                revcomp = s[::-1].translate(str.maketrans("ACGT", "TGCA"))
                hits = query(DB, """SELECT model_id, pattern_id, grade, card_state, unitig_id, length FROM v_unitig
                                     WHERE primary_mode = 1 AND (sequence LIKE ? OR sequence LIKE ?)""",
                             (f"%{s}%", f"%{revcomp}%"))
        if len(hits):
            hits = hits.copy()
            hits["_rank"] = hits.grade.map(grade_rank)
            st.write(f"{len(hits)} results in {hits.model_id.nunique()} models")
            st.dataframe(hits.sort_values(["_rank", "model_id"]).drop(columns="_rank"), hide_index=True,
                         use_container_width=True)
        else:
            st.info("No results.")

# --- 5. independent isolates ------------------------------------------------------------------
if PAGE == "Independent isolates (CABBAGE)":
    rd = results_directory(db)
    comparison, search = cabbage(str(rd)) if rd else (pd.DataFrame(), pd.DataFrame())
    if comparison.empty and search.empty:
        st.info("No CABBAGE results found (results/<organism>/cabbage/). Give the directory with --results.")
    else:
        st.caption("CABBAGE: isolates independent of the training data. A pair is assessed with at least 10 "
                   "resistant and 10 susceptible isolates. A tool that makes no call for the antibiotic has no row.")
        k = comparison[(comparison.model_id == MODEL) & (comparison.assessable == 1)].copy()
        k["predictor"] = k.tool.map(TOOL_NAME).fillna(k.tool)
        a = search[search.model_id == MODEL]
        both = pd.concat([a, k[["predictor", "n", "n_resistant", "sensitivity", "specificity",
                                "balanced_accuracy"]]], ignore_index=True)
        st.subheader(f"{ORG_NAME.get(org, org)} – {ab_name(ab)}")
        if both.empty:
            st.info("This pair was not assessed on independent isolates.")
        else:
            n0 = both.iloc[0]
            enough = n0.n_resistant >= 10 and n0.n - n0.n_resistant >= 10
            st.write(f"{int(n0.n)} isolates, {int(n0.n_resistant)} resistant"
                     + ("" if enough else " — fewer than 10 resistant and 10 susceptible isolates"))
            chart = (alt.Chart(both).mark_bar().encode(
                x=alt.X("balanced_accuracy:Q", title="balanced accuracy", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("predictor:N", title=None, sort="-x", axis=alt.Axis(labelLimit=360)),
                color=alt.condition(alt.FieldOneOfPredicate("predictor", list(SEARCH_NAME.values()) + ["KANIT model"]),
                                    alt.value("#1a1a1a"), alt.value("#9e9e9e")),
                tooltip=["predictor", alt.Tooltip("balanced_accuracy:Q", format=".2f"),
                         alt.Tooltip("sensitivity:Q", format=".2f"), alt.Tooltip("specificity:Q", format=".2f")])
                + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[3, 3], color="#7a7a7a").encode(x="x:Q"))
            st.altair_chart(chart.properties(height=28 * len(both)), use_container_width=True)
            st.table(both[["predictor", "sensitivity", "specificity", "balanced_accuracy"]].rename(
                columns={"balanced_accuracy": "balanced accuracy"}).set_index("predictor").style.format(precision=2))

        st.subheader("All pairs: the KANIT search against the best catalogue-based tool")
        ks = search[search.predictor == SEARCH_NAME["upto_strong_novel"]].set_index("model_id")
        ks = ks[(ks.n_resistant >= 10) & (ks.n - ks.n_resistant >= 10)]
        tools = (comparison[comparison.tool.isin(["amrfinderplus", "resfinder"]) & (comparison.assessable == 1)]
                 .pivot_table(index="model_id", columns="tool", values="balanced_accuracy"))
        pairs = ks[["balanced_accuracy", "sensitivity", "specificity"]].join(tools, how="left")
        pairs["best tool"] = pairs[["amrfinderplus", "resfinder"]].max(axis=1)
        pairs["difference"] = pairs.balanced_accuracy - pairs["best tool"]
        pairs.index = [f"{ORG_NAME.get(i.split('__')[0], i)} – {ab_name(i.split('__', 1)[1])}" for i in pairs.index]
        st.caption("Difference = KANIT search (up to strong_novel) − the higher of AMRFinderPlus and ResFinder. "
                   "Click a column header to sort.")
        st.dataframe(pairs.rename(columns={"balanced_accuracy": "KANIT BA", "sensitivity": "KANIT sensitivity",
                                           "specificity": "KANIT specificity", "amrfinderplus": "AMRFinderPlus BA",
                                           "resfinder": "ResFinder BA", "best tool": "best tool BA"})
                     .sort_values("difference").style.format(precision=2, na_rep="—")
                     .background_gradient(subset=["difference"], cmap="RdBu", vmin=-0.4, vmax=0.4),
                     use_container_width=True, height=420)
