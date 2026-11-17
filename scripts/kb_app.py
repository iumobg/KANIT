#!/usr/bin/env python3
"""
KANIT gezgini: bilgi tabanının (kanit.sqlite) yerel, etkileşimli arayüzü.

Bir organizma–antibiyotik modeli seçilir; modelin başarımı, kanıt dereceli biyobelirteçleri, her
biyobelirtecin kanıt zinciri, NCBI'daki genomik bağlamı ve dizileri görülür. Gen adı, unitig kimliği ya
da DNA dizisiyle arama yapılır. CABBAGE sonuçları varsa bağımsız izolatlardaki karşılaştırma da gösterilir.

Yerelde çalışır (HPC hattının parçası değil):
    pip install streamlit pandas altair
    streamlit run scripts/kb_app.py -- --db <kanit.sqlite> [--results <results dizini>]
--db verilmezse KANIT_DB ortam değişkeni, o da yoksa results/kb/kanit.sqlite okunur. --results,
CABBAGE dosyalarının (results/<organizma>/cabbage/) bulunduğu dizin; verilmezse veritabanının üstündeki
ilk `results` dizini denenir. Veritabanı salt okunur açılır.
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

ORG_AD = {"enterococcus_faecium": "E. faecium", "staphylococcus_aureus": "S. aureus",
          "kpneumoniae": "K. pneumoniae", "acinetobacter_baumannii": "A. baumannii",
          "pseudomonas_aeruginosa": "P. aeruginosa", "enterobacter_cloacae": "E. cloacae kompleksi",
          "ecoli": "E. coli"}
ORG_SIRA = list(ORG_AD)
DERECELER = ["confirmed", "strong_novel", "candidate", "weak", "none"]
DERECE_RENK = {"confirmed": "#1c5cab", "strong_novel": "#eb6834", "candidate": "#3987e5",
               "weak": "#86b6ef", "none": "#bdbdbd"}
DERECE_ANLAM = {
    "confirmed": "küratörlü bir determinanta uyuyor (b = 1) ve en az iki istatistiksel katman destekliyor",
    "strong_novel": "hiçbir CARD eşleşmesi yok; kararlılık seçimi, karma model ve en az bir katman daha geçiyor",
    "candidate": "kanıt var ama tamamlanmıyor: uyum var ama en çok bir katman, ya da katmanlar var ama uyum "
                 "yok (gerekçesiyle) ya da kararlılık ile karma model birlikte geçmiyor",
    "weak": "uyum yok ve tam olarak bir katman geçiyor",
    "none": "hiçbir katman geçmiyor; modelin önem sıralamasından başka kanıt yok"}
ARAC_AD = {"amrfinderplus": "AMRFinderPlus", "resfinder": "ResFinder", "rgi_all": "RGI (bütün eşleşmeler)",
           "rgi_without_near_universal": "RGI (neredeyse evrensel genler çıkarılınca)", "model": "KANIT modeli"}
ARAMA_AD = {"upto_confirmed": "KANIT araması: yalnız confirmed",
            "upto_strong_novel": "KANIT araması: strong_novel'a kadar (birincil)",
            "upto_candidate": "KANIT araması: candidate'a kadar", "upto_weak": "KANIT araması: weak'e kadar",
            "b1": "KANIT araması: yalnız b = 1"}


def ab_ad(ab: str) -> str:
    return ab.replace("_", " ")


def derece_sira(g: str) -> int:
    return DERECELER.index(g) if g in DERECELER else len(DERECELER)


# --- veri -----------------------------------------------------------------------------------
@st.cache_resource
def baglan(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def sorgu(path: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, baglan(path), params=params)


def results_dizini(db: Path) -> Path | None:
    if ARGS.results:
        return Path(ARGS.results)
    for p in db.resolve().parents:
        if p.name == "results":
            return p
    return None


@st.cache_data(show_spinner=False)
def cabbage(results: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bağımsız izolatlardaki sonuçlar: araçlar ve model (karşılaştırma tablosu), KANIT araması."""
    kars = [pd.read_csv(f) for f in glob.glob(f"{results}/*/cabbage/cabbage_comparison.csv")]
    rows = []
    for f in glob.glob(f"{results}/*/cabbage/cabbage_lookup_metrics.json"):
        j = json.load(open(f))
        for ab, v in j["pairs"].items():
            for kural, ad in ARAMA_AD.items():
                a = v.get(kural, {}).get("all", {})
                if a.get("assessed"):
                    rows.append(dict(model_id=f"{j['organism']}__{ab}", ongorucu=ad, n=a["n"],
                                     n_resistant=a["n_resistant"], sensitivity=a["sensitivity"],
                                     specificity=a["specificity"], balanced_accuracy=a["balanced_accuracy"]))
    return (pd.concat(kars, ignore_index=True) if kars else pd.DataFrame()), pd.DataFrame(rows)


# --- kenar çubuğu ----------------------------------------------------------------------------
st.sidebar.title("KANIT")
db = Path(st.sidebar.text_input("Veritabanı (kanit.sqlite)", ARGS.db))
if not db.exists():
    st.error(f"Veritabanı bulunamadı: {db}")
    st.stop()
DB = str(db)
rel = sorgu(DB, "SELECT * FROM release").iloc[0]
st.sidebar.caption(f"Sürüm {rel.kb_version} · CARD {rel.card_version} · {str(rel.created_at)[:10]}")

modeller = sorgu(DB, "SELECT * FROM v_model")
modeller["_o"] = modeller.organism_id.map({o: i for i, o in enumerate(ORG_SIRA)})
orgs = sorted(modeller.organism_id.unique(), key=lambda o: ORG_SIRA.index(o) if o in ORG_SIRA else 99)
org = st.sidebar.selectbox("Organizma", orgs, format_func=lambda o: ORG_AD.get(o, o),
                           index=orgs.index("enterococcus_faecium") if "enterococcus_faecium" in orgs else 0)
abler = sorted(modeller[modeller.organism_id == org].antibiotic_id)
ab = st.sidebar.selectbox("Antibiyotik", abler, format_func=ab_ad)
MODEL = f"{org}__{ab}"
st.sidebar.markdown("**Dereceler**")
for g in DERECELER:
    st.sidebar.markdown(f"<span style='display:inline-block;width:0.7em;height:0.7em;margin-right:0.3em;"
                        f"background:{DERECE_RENK[g]}'></span>`{g}`: {DERECE_ANLAM[g]}",
                        unsafe_allow_html=True)
st.sidebar.caption("Derece bir ilişki söyler, nedensellik söylemez. Yön R: dirençlilerde daha sık; "
                   "S: duyarlılarda daha sık.")

st.title("KANIT: kanıt dereceli AMR biyobelirteç bilgi tabanı")
# Sekme yerine tek sayfa: gizli sekmelerdeki tablolar genişliğini yanlış ölçüyor.
SAYFALAR = ["Genel bakış", "Model", "Biyobelirteç", "Ara", "Bağımsız izolatlar (CABBAGE)"]
SAYFA = st.radio("Bölüm", SAYFALAR, horizontal=True, label_visibility="collapsed", key="sayfa")
st.divider()

# --- 1. genel bakış --------------------------------------------------------------------------
if SAYFA == "Genel bakış":
    c = st.columns(5)
    c[0].metric("Model", len(modeller))
    c[1].metric("Organizma", modeller.organism_id.nunique())
    c[2].metric("Aday biyobelirteç", int(modeller.n_candidates.sum()))
    c[3].metric("confirmed", int(modeller.n_confirmed.sum()))
    c[4].metric("strong_novel", int(modeller.n_strong_novel.sum()))

    st.subheader("Hangi organizmada hangi antibiyotiğin modeli var")
    st.caption("Hücre: soy-farkında ROC-AUC. Boş hücre: o çift için model yok.")
    isi = modeller.assign(organizma=modeller.organism_id.map(ORG_AD), antibiyotik=modeller.antibiotic_id.map(ab_ad),
                          auc=modeller.roc_auc_lineage_aware.round(2))
    org_sira = [ORG_AD[o] for o in ORG_SIRA if o in set(modeller.organism_id)]
    taban = alt.Chart(isi).encode(
        x=alt.X("antibiyotik:N", title=None, sort="ascending",
              axis=alt.Axis(labelAngle=-60, labelOverlap=False, labelLimit=160)),
        y=alt.Y("organizma:N", title=None, sort=org_sira))
    isi_h = taban.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("auc:Q", title="ROC-AUC", scale=alt.Scale(scheme="blues", domain=[0.5, 1.0])),
        tooltip=["organizma", "antibiyotik", alt.Tooltip("auc:Q", title="ROC-AUC (soy-farkında)"),
                 alt.Tooltip("n_genomes:Q", title="genom"), alt.Tooltip("n_resistant:Q", title="dirençli"),
                 alt.Tooltip("n_confirmed:Q", title="confirmed"), alt.Tooltip("n_strong_novel:Q", title="strong_novel")])
    yazi = taban.mark_text(fontSize=10).encode(
        text=alt.Text("auc:Q", format=".2f"),
        color=alt.condition("datum.auc > 0.85", alt.value("white"), alt.value("black")))
    st.altair_chart((isi_h + yazi).properties(height=330), use_container_width=True)

    st.subheader("Organizma başına")
    oz = (modeller.groupby("organism_id")
          .agg(model=("model_id", "count"), genom_medyan=("n_genomes", "median"),
               roc_auc_medyan=("roc_auc_lineage_aware", "median"), aday=("n_candidates", "sum"),
               confirmed=("n_confirmed", "sum"), strong_novel=("n_strong_novel", "sum"))
          .reindex([o for o in ORG_SIRA if o in set(modeller.organism_id)]))
    oz.index = [ORG_AD.get(o, o) for o in oz.index]
    st.dataframe(oz.style.format({"genom_medyan": "{:.0f}", "roc_auc_medyan": "{:.2f}"}), use_container_width=True)

# --- 2. model --------------------------------------------------------------------------------
bb = sorgu(DB, """SELECT b.*, n.context_gene, n.context_product, n.context_plasmid_share
                    FROM v_biomarker b LEFT JOIN v_novel n
                      ON n.model_id = b.model_id AND n.pattern_id = b.pattern_id
                   WHERE b.model_id = ? AND b.route = 'model'""", (MODEL,))
bb["_d"] = bb.grade.map(derece_sira)
bb = bb.sort_values(["_d", "pi"], ascending=[True, False])

if SAYFA == "Model":
    m = modeller[modeller.model_id == MODEL].iloc[0]
    st.header(f"{ORG_AD.get(org, org)} – {ab_ad(ab)}")
    c = st.columns(7)
    c[0].metric("Genom", int(m.n_genomes))
    c[1].metric("Dirençli", int(m.n_resistant))
    c[2].metric("Duyarlı", int(m.n_susceptible))
    c[3].metric("Soy", int(m.n_lineages))
    c[4].metric("ROC-AUC, soy-farkında", f"{m.roc_auc_lineage_aware:.2f}",
                help=f"%95 aralık {m.roc_auc_lineage_aware_low:.2f}–{m.roc_auc_lineage_aware_high:.2f}")
    c[5].metric("ROC-AUC, rastgele", f"{m.roc_auc_lineage_blind:.2f}",
                delta=f"{m.roc_auc_lineage_blind - m.roc_auc_lineage_aware:+.2f}", delta_color="off",
                help="Soyu hesaba katmayan bölme; fark, soyun başarımı ne kadar şişirdiği.")
    c[6].metric("Permütasyon q", f"{m.label_permutation_q:.3g}",
                help="Etiket permütasyonu: model şanstan iyi mi? q < 0,05 ise evet.")
    if isinstance(m.flag, str) and m.flag:
        st.warning(f"İşaret: {m.flag}")

    st.subheader("Biyobelirteçler, derece ve yöne göre")
    say = bb.groupby(["grade", "direction"]).size().reset_index(name="n")
    if len(say):
        say["sira"] = say.grade.map(derece_sira)
        say["yön"] = say.direction.map({"R": "R (dirençlilerde sık)", "S": "S (duyarlılarda sık)"})
        grafik = alt.Chart(say).mark_bar(stroke="white", strokeWidth=1).encode(
            x=alt.X("n:Q", title="biyobelirteç", stack="zero"),
            y=alt.Y("yön:N", title=None),
            color=alt.Color("grade:N", sort=DERECELER, title="derece",
                            scale=alt.Scale(domain=DERECELER, range=[DERECE_RENK[g] for g in DERECELER]),
                            legend=alt.Legend(orient="bottom", direction="horizontal")),
            order=alt.Order("sira:Q"), tooltip=["grade", "direction", "n"]).properties(height=alt.Step(38))
        st.altair_chart(grafik, use_container_width=True)

    c1, c2 = st.columns(2)
    sec_d = c1.multiselect("Derece", DERECELER, default=DERECELER)
    sec_y = c2.multiselect("Yön", ["R", "S"], default=["R", "S"])
    tablo = bb[bb.grade.isin(sec_d) & bb.direction.isin(sec_y)]
    goster = tablo[["pattern_id", "grade", "direction", "n_members", "layers_passed", "delta", "pi",
                    "pyseer_p", "card_genes", "card_state_allele_aware", "context_gene", "context_product",
                    "context_plasmid_share", "gain_rank"]].rename(columns={
        "pattern_id": "örüntü", "grade": "derece", "direction": "yön", "n_members": "unitig",
        "layers_passed": "geçen katmanlar", "delta": "Δ (R−S)", "pi": "π", "pyseer_p": "pyseer p",
        "card_genes": "CARD genleri", "card_state_allele_aware": "CARD durumu", "context_gene": "NCBI gen",
        "context_product": "NCBI ürün", "context_plasmid_share": "plazmid payı", "gain_rank": "kazanç sırası"})
    st.caption("Bir satır seçip 'Biyobelirteç' bölümüne geçince o biyobelirtecin kanıt zinciri açılır.")
    olay = st.dataframe(goster, hide_index=True, use_container_width=True, on_select="rerun",
                        selection_mode="single-row",
                        column_config={"Δ (R−S)": st.column_config.NumberColumn(format="%.2f"),
                                       "π": st.column_config.NumberColumn(format="%.2f"),
                                       "pyseer p": st.column_config.NumberColumn(format="%.1e"),
                                       "plazmid payı": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                        format="%.2f")})
    if olay.selection.rows:
        st.session_state["oruntu"] = int(tablo.iloc[olay.selection.rows[0]].pattern_id)

    egitim = sorgu(DB, "SELECT * FROM external_comparison WHERE model_id = ?", (MODEL,))
    if len(egitim):
        st.subheader("Eğitim genomlarında katalog araçları (katlar dışı)")
        e = egitim[egitim.assessable == 1].copy()
        e["araç"] = e.tool.map(ARAC_AD).fillna(e.tool)
        st.dataframe(e[["araç", "n", "n_resistant", "sensitivity", "specificity", "balanced_accuracy"]]
                     .rename(columns={"n_resistant": "dirençli", "sensitivity": "duyarlılık",
                                      "specificity": "özgüllük", "balanced_accuracy": "dengeli doğruluk"})
                     .style.format(precision=2), hide_index=True, use_container_width=True)

# --- 3. biyobelirteç -------------------------------------------------------------------------
if SAYFA == "Biyobelirteç":
    if bb.empty:
        st.info("Bu modelde biyobelirteç yok.")
    else:
        secenek = list(bb.pattern_id.astype(int))
        varsayilan = st.session_state.get("oruntu")
        idx = secenek.index(varsayilan) if varsayilan in secenek else 0
        etiket = {int(r.pattern_id): f"{int(r.pattern_id)} · {r.grade} · {r.direction}"
                                     + (f" · {r.card_genes}" if r.card_genes else "")
                                     + (f" · {r.context_gene}" if isinstance(r.context_gene, str) and r.context_gene else "")
                  for r in bb.itertuples()}
        pid = st.selectbox("Örüntü", secenek, index=idx, format_func=lambda p: etiket[p])
        r = bb[bb.pattern_id == pid].iloc[0]
        st.markdown(f"### Derece: <span style='color:{DERECE_RENK.get(r.grade, '#000')}'>`{r.grade}`</span>"
                    f" · yön {r.direction}", unsafe_allow_html=True)
        st.caption(DERECE_ANLAM.get(r.grade, ""))

        zincir = sorgu(DB, """
            SELECT 'yaygınlık farkı (Δ = R − S)' AS katman, printf('%.3f', delta) AS değer,
                   printf('q = %.1e', q) AS anlamlılık, passes FROM prevalence_result WHERE model_id = ? AND pattern_id = ?
            UNION ALL SELECT 'permütasyon önemi (ROC-AUC düşüşü)', printf('%.4f', mda), printf('q = %.1e', q), passes
                   FROM mda_result WHERE model_id = ? AND pattern_id = ?
            UNION ALL SELECT 'kararlılık seçimi (π)', printf('%.3f', pi), 'eşik π ≥ 0,6', passes
                   FROM cpss_result WHERE model_id = ? AND pattern_id = ?
            UNION ALL SELECT 'doğrusal karma model (β)', printf('%.3f', beta), printf('p = %.1e', lrt_p), passes
                   FROM pyseer_result WHERE model_id = ? AND pattern_id = ?""", (MODEL, int(pid)) * 4)
        zincir["sonuç"] = zincir.passes.map({1: "geçti", 0: "geçmedi"}).fillna("sınanmadı")
        c1, c2 = st.columns([3, 2])
        c1.markdown("**İstatistiksel kanıt katmanları**")
        c1.table(zincir[["katman", "değer", "anlamlılık", "sonuç"]].set_index("katman"))
        c2.markdown("**Küratörlü determinantlarla uyum (CARD)**")
        c2.markdown(f"- Durum: `{r.card_state_allele_aware}`\n"
                    f"- CARD genleri: {r.card_genes or '—'}\n"
                    f"- Gerekçe: `{r.card_reasons_allele_aware or '—'}`\n"
                    f"- Aday kümesine giriş yolu: {r.candidate_source}"
                    + (f" (kazanç sırası {int(r.gain_rank)})" if pd.notna(r.gain_rank) else ""))

        uy = sorgu(DB, """SELECT u.unitig_id, u.length, u.sequence, c.gene, c.product, c.best_title,
                                 c.best_identity, c.best_coverage, c.plasmid_share
                            FROM pattern_member pm JOIN unitig u USING (unitig_id)
                            LEFT JOIN unitig_context c ON c.unitig_id = u.unitig_id AND c.organism_id = ?
                           WHERE pm.model_id = ? AND pm.pattern_id = ? ORDER BY u.length DESC""",
                   (org, MODEL, int(pid)))
        bag = uy.dropna(subset=["best_title"])
        st.markdown("**NCBI'daki en yakın dizi (en uzun unitig)**")
        if len(bag):
            b0 = bag.iloc[0]
            st.markdown(f"- Gen: `{b0['gene'] or '—'}` · Ürün: {b0['product'] or '—'}\n"
                        f"- Kayıt: {b0['best_title']}\n"
                        f"- Özdeşlik %{b0['best_identity']:.1f} · örtüşme %{100 * b0['best_coverage']:.0f} · "
                        f"NCBI eşleşmelerinin plazmid payı {b0['plasmid_share']:.2f}")
        else:
            st.caption("Bu örüntü için NCBI bağlamı sorgulanmamış.")
        st.markdown(f"**Unitig'ler ({len(uy)})**")
        st.table(uy[["unitig_id", "length", "gene", "product"]].fillna("").rename(
            columns={"length": "uzunluk", "gene": "NCBI gen", "product": "NCBI ürün"}).set_index("unitig_id"))
        if len(uy):
            st.markdown("**En uzun unitig'in dizisi**")
            st.code(uy.iloc[0]["sequence"], language=None, wrap_lines=True)
            baska = sorgu(DB, f"""SELECT model_id, grade, card_state, unitig_id FROM v_unitig
                                  WHERE unitig_id IN ({','.join('?' * len(uy))}) AND model_id <> ?
                                    AND primary_mode = 1""", (*uy.unitig_id, MODEL))
            if len(baska):
                st.markdown("**Aynı unitig'ler başka modellerde de var**")
                st.table(baska.drop_duplicates(["model_id", "grade"]).set_index("model_id"))

# --- 4. arama --------------------------------------------------------------------------------
if SAYFA == "Ara":
    tur = st.radio("Neye göre", ["Gen ya da ürün adı", "Unitig kimliği", "DNA dizisi"], horizontal=True)
    q = st.text_input("Aranan", placeholder={"Gen ya da ürün adı": "ör. KPC, mecA, gyrA, transposase",
                                              "Unitig kimliği": "ör. KUc0f88eb591ee5a2551f2",
                                              "DNA dizisi": "en az 15 baz"}[tur]).strip()
    if q:
        if tur == "Gen ya da ürün adı":
            sonuc = sorgu(DB, """SELECT DISTINCT b.model_id, b.pattern_id, b.grade, b.direction, b.card_genes,
                                        n.context_gene, n.context_product
                                   FROM v_biomarker b LEFT JOIN v_novel n
                                     ON n.model_id = b.model_id AND n.pattern_id = b.pattern_id
                                  WHERE b.route = 'model' AND (b.card_genes LIKE ? OR n.context_gene LIKE ?
                                        OR n.context_product LIKE ?)""", (f"%{q}%",) * 3)
        elif tur == "Unitig kimliği":
            sonuc = sorgu(DB, """SELECT model_id, pattern_id, grade, card_state, length FROM v_unitig
                                  WHERE unitig_id = ? AND primary_mode = 1""", (q,))
        else:
            s = q.upper().replace(" ", "")
            if len(s) < 15:
                st.warning("En az 15 baz girin.")
                sonuc = pd.DataFrame()
            else:
                ters = s[::-1].translate(str.maketrans("ACGT", "TGCA"))
                sonuc = sorgu(DB, """SELECT model_id, pattern_id, grade, card_state, unitig_id, length FROM v_unitig
                                      WHERE primary_mode = 1 AND (sequence LIKE ? OR sequence LIKE ?)""",
                              (f"%{s}%", f"%{ters}%"))
        if len(sonuc):
            sonuc = sonuc.copy()
            sonuc["_d"] = sonuc.grade.map(derece_sira)
            st.write(f"{len(sonuc)} sonuç, {sonuc.model_id.nunique()} modelde")
            st.dataframe(sonuc.sort_values(["_d", "model_id"]).drop(columns="_d"), hide_index=True,
                         use_container_width=True)
        else:
            st.info("Sonuç yok.")

# --- 5. bağımsız izolatlar -------------------------------------------------------------------
if SAYFA == "Bağımsız izolatlar (CABBAGE)":
    rd = results_dizini(db)
    kars, arama = cabbage(str(rd)) if rd else (pd.DataFrame(), pd.DataFrame())
    if kars.empty and arama.empty:
        st.info("CABBAGE sonuçları bulunamadı (results/<organizma>/cabbage/). --results ile dizini verin.")
    else:
        st.caption("CABBAGE: eğitim verisinden bağımsız izolatlar. Bir çift en az 10 dirençli ve 10 duyarlı "
                   "izolatla değerlendirilir. Bir araç o antibiyotik için çağrı yapamıyorsa satırı yoktur.")
        k = kars[(kars.model_id == MODEL) & (kars.assessable == 1)].copy()
        k["ongorucu"] = k.tool.map(ARAC_AD).fillna(k.tool)
        a = arama[arama.model_id == MODEL]
        ikisi = pd.concat([a, k[["ongorucu", "n", "n_resistant", "sensitivity", "specificity",
                                 "balanced_accuracy"]]], ignore_index=True)
        st.subheader(f"{ORG_AD.get(org, org)} – {ab_ad(ab)}")
        if ikisi.empty:
            st.info("Bu çift bağımsız izolatlarda değerlendirilmedi.")
        else:
            n0 = ikisi.iloc[0]
            yeter = n0.n_resistant >= 10 and n0.n - n0.n_resistant >= 10
            st.write(f"{int(n0.n)} izolat, {int(n0.n_resistant)} dirençli"
                     + ("" if yeter else " — 10 dirençli ve 10 duyarlı koşulu sağlanmıyor"))
            grafik = (alt.Chart(ikisi).mark_bar().encode(
                x=alt.X("balanced_accuracy:Q", title="dengeli doğruluk", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("ongorucu:N", title=None, sort="-x", axis=alt.Axis(labelLimit=360)),
                color=alt.condition(alt.FieldOneOfPredicate("ongorucu", list(ARAMA_AD.values()) + ["KANIT modeli"]),
                                    alt.value("#1a1a1a"), alt.value("#9e9e9e")),
                tooltip=["ongorucu", alt.Tooltip("balanced_accuracy:Q", format=".2f"),
                         alt.Tooltip("sensitivity:Q", format=".2f"), alt.Tooltip("specificity:Q", format=".2f")])
                + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[3, 3], color="#7a7a7a").encode(x="x:Q"))
            st.altair_chart(grafik.properties(height=28 * len(ikisi)), use_container_width=True)
            st.table(ikisi[["ongorucu", "sensitivity", "specificity", "balanced_accuracy"]].rename(
                columns={"ongorucu": "öngörücü", "sensitivity": "duyarlılık", "specificity": "özgüllük",
                         "balanced_accuracy": "dengeli doğruluk"}).set_index("öngörücü").style.format(precision=2))

        st.subheader("Bütün çiftler: KANIT araması ile en iyi katalog aracı")
        ak = arama[arama.ongorucu == ARAMA_AD["upto_strong_novel"]].set_index("model_id")
        ak = ak[(ak.n_resistant >= 10) & (ak.n - ak.n_resistant >= 10)]
        arac = (kars[kars.tool.isin(["amrfinderplus", "resfinder"]) & (kars.assessable == 1)]
                .pivot_table(index="model_id", columns="tool", values="balanced_accuracy"))
        tum = ak[["balanced_accuracy", "sensitivity", "specificity"]].join(arac, how="left")
        tum["en iyi araç"] = tum[["amrfinderplus", "resfinder"]].max(axis=1)
        tum["fark"] = tum.balanced_accuracy - tum["en iyi araç"]
        tum.index = [f"{ORG_AD.get(i.split('__')[0], i)} – {ab_ad(i.split('__', 1)[1])}" for i in tum.index]
        st.caption("Fark = KANIT araması (strong_novel'a kadar) − AMRFinderPlus ya da ResFinder'ın yükseği. "
                   "Sütun başlığına tıklayıp sıralayabilirsiniz.")
        st.dataframe(tum.rename(columns={"balanced_accuracy": "KANIT DD", "sensitivity": "KANIT duyarlılık",
                                         "specificity": "KANIT özgüllük", "amrfinderplus": "AMRFinderPlus DD",
                                         "resfinder": "ResFinder DD", "en iyi araç": "en iyi araç DD"})
                     .sort_values("fark").style.format(precision=2, na_rep="—")
                     .background_gradient(subset=["fark"], cmap="RdBu", vmin=-0.4, vmax=0.4),
                     use_container_width=True, height=420)
