#!/usr/bin/env python3
"""External validation on CABBAGE: the eligible isolates (scripts/lib/cabbage.py)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import cabbage, registry  # noqa: E402

pytestmark = pytest.mark.unit


def _row(bs, species, ab, std="CLSI", pheno="resistant", ecoff="no", asm=None, sra=None,
         db="NCBI_antibiogram"):
    return {"BioSample_ID": bs, "SRA_accession": sra, "assembly_ID": asm, "species": species,
            "database": db, "antibiotic_name": ab, "ast_standard": std,
            "resistance_phenotype": pheno, "used_ECOFF": ecoff}


EC, KP = "Escherichia coli", "Klebsiella pneumoniae"
ROWS = [
    _row("S1", EC, "ciprofloxacin", asm="GCA_000000001.1"),
    _row("S1", EC, "ciprofloxacin", std="EUCAST", asm="GCA_000000001.1"),
    _row("S2", EC, "ciprofloxacin", pheno="susceptible", asm="GCA_000000002.1"),   # tie
    _row("S2", EC, "ciprofloxacin", asm="GCA_000000002.1"),
    _row("S2", EC, "ampicillin", pheno="susceptible", asm="GCA_000000002.1"),
    _row("S3", "Enterobacter hormaechei", "trimethoprim-sulfamethoxazole", pheno="susceptible",
         asm="GCA_000000003.1"),
    _row("S3", "Enterobacter hormaechei", "trimethoprim-sulfamethoxazole", pheno="susceptible",
         asm="GCA_000000003.2"),
    _row("S4", "Enterobacter kobei", "ciprofloxacin", asm="GCA_000000004.1"),     # species
    _row("S5", EC, "ciprofloxacin", std="NCCLS", asm="GCA_000000005.1"),         # standard
    _row("S6", EC, "ciprofloxacin", pheno="intermediate", asm="GCA_000000006.1"),
    _row("S7", EC, "ciprofloxacin", ecoff="yes", asm="GCA_000000007.1"),
    _row("S8", EC, "ciprofloxacin", asm="GCA_000000008.1"),
    _row("S8", EC, "ampicillin", std="NARMS", db="PATRIC;NARMS", asm="GCA_000000008.1"),
    _row("S9", EC, "ciprofloxacin", asm="GCA_000000009.1"),                      # BioSample
    _row("S10", EC, "ciprofloxacin", asm="GCA_000000010.1"),                     # assembly
    _row("S11", EC, "ciprofloxacin"),                                            # no assembly
    _row("S12", EC, "ciprofloxacin", asm="GCA_000000012.1"),
    _row("S12", EC, "ciprofloxacin", asm="GCA_000000013.1"),                     # two assemblies
    _row("S13", KP, "amoxicillin-clavulanic acid", asm="GCA_000000014.1"),
    _row("S13", KP, "cefpodoxime-clavulanic acid", pheno="susceptible", asm="GCA_000000014.1"),
]
IDS = {"biosample": {"S9"}, "sra": set(), "assembly": {"GCA_000000010"}}


def test_names_and_accessions():
    assert cabbage.antibiotic("trimethoprim-sulfamethoxazole") == "trimethoprim_sulfamethoxazole"
    assert cabbage.antibiotic("cefpodoxime-clavulanic acid") == "cefpodoxime_clavulanic_acid"
    assert cabbage.antibiotic("Imipenem") == "imipenem" and cabbage.antibiotic(" ") is None
    assert cabbage.accessions(["SRR1, SRR2", None, "srr3"]) == {"SRR1", "SRR2", "SRR3"}
    assert cabbage.unversioned("GCA_000001405.15") == "GCA_000001405"
    g = pd.DataFrame({"biosample_accession": ["SAMN1", ""], "sra_accession": ["SRR1,SRR2", ""],
                      "assembly_accession": ["GCA_000000001.2", ""]})
    assert cabbage.bvbrc_ids(g) == {"biosample": {"SAMN1"}, "sra": {"SRR1", "SRR2"},
                                    "assembly": {"GCA_000000001"}}
    idx = cabbage.species_index(registry.load_organisms())
    assert idx["Enterobacter bugandensis"] == "enterobacter_cloacae" and "Enterobacter kobei" not in idx


def test_select_follows_every_rule():
    isolates, phenotypes, counts = cabbage.select(pd.DataFrame(ROWS), registry.load_organisms(), IDS)
    assert isolates["biosample_id"].tolist() == ["S1", "S2", "S3", "S13"]
    assert dict(zip(isolates["biosample_id"], isolates["assembly_id"], strict=True))["S3"] == \
        "GCA_000000003.2"                                      # the latest version
    p = {(r.biosample_id, r.antibiotic): r for r in phenotypes.itertuples()}
    assert p[("S1", "ciprofloxacin")].label == 1 and p[("S1", "ciprofloxacin")].n_records == 2
    assert ("S2", "ciprofloxacin") not in p and p[("S2", "ampicillin")].label == 0
    assert p[("S3", "trimethoprim_sulfamethoxazole")].label == 0
    assert {a for b, a in p if b == "S13"} == {"amoxicillin_clavulanic_acid",
                                               "cefpodoxime_clavulanic_acid"}
    ec = counts["ecoli"]
    assert [ec[s] for s in cabbage.STEPS] == [10, 9, 8, 7, 7, 6, 4, 2]
    assert ec["cells_tied"] == 1 and ec["cells"] == 2
    assert counts["enterobacter_cloacae"]["assembly"] == 1 and counts["kpneumoniae"]["cells"] == 2
    from lib import contract
    tables = contract.load()["tables"]
    for tid, frame in (("cabbage_isolates", isolates), ("cabbage_phenotypes", phenotypes),
                       ("cabbage_pairs", cabbage.pair_counts(phenotypes))):
        assert contract.validate_frame(frame.astype(str), tables[tid]) == [], tid
    t = cabbage.pair_counts(phenotypes).set_index(["organism", "antibiotic"])
    assert t.loc[("ecoli", "ciprofloxacin"), "n_resistant"] == 1
    assert t.loc[("ecoli", "ampicillin"), "n_susceptible"] == 1


def test_select_needs_cabbage_columns():
    with pytest.raises(ValueError, match="columns missing"):
        cabbage.select(pd.DataFrame(ROWS).drop(columns=["used_ECOFF"]),
                       registry.load_organisms(), IDS)


# ---- NCBI Datasets client and the fetch step ------------------------------------------

def _zip(acc: str, fasta: bytes) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"ncbi_dataset/data/{acc}/{acc}_ASM1_genomic.fna", fasta)
        z.writestr("ncbi_dataset/data/assembly_data_report.jsonl", "{}")
    return buf.getvalue()


class FakeNcbi:
    """fetch(url) of the Datasets client: genome reports, lineages and packages."""

    def __init__(self, genomes, taxa):
        self.genomes, self.taxa, self.calls = genomes, taxa, []

    def __call__(self, url: str) -> bytes:
        import json
        import urllib.parse
        self.calls.append(url)
        path = urllib.parse.unquote(url.split("/v2/", 1)[1])
        if path.startswith("genome/accession/") and "/dataset_report" in path:
            accs = path.split("/")[2].split(",")
            return json.dumps({"reports": [
                {"accession": a, "organism": {"tax_id": g["tax"], "organism_name": g["name"]},
                 "assembly_stats": {"number_of_contigs": g["contigs"],
                                    "total_sequence_length": str(g["length"])}}
                for a in accs if (g := self.genomes.get(a))]}).encode()
        if path.startswith("taxonomy/taxon/"):
            ids = [int(t) for t in path.split("/")[2].split(",")]
            return json.dumps({"reports": [{"taxonomy": {"tax_id": t, "parents": self.taxa[t]}}
                                           for t in ids if t in self.taxa]}).encode()
        acc = path.split("/")[2]
        return _zip(acc, self.genomes[acc]["fasta"])


def test_datasets_client(monkeypatch):
    import urllib.error

    from lib import ncbi_datasets as nd
    fake = FakeNcbi({"GCA_1.1": {"tax": 562, "name": "E. coli", "contigs": 2, "length": 8,
                                 "fasta": b">a\nACGT\n>b\nACGT\n"},
                     "GCA_2.1": {"tax": 562, "name": "E. coli", "contigs": 1, "length": 4,
                                 "fasta": b">a\nACGT\n"}}, {562: [1, 561]})
    monkeypatch.setattr(nd, "BATCH", 1)
    slept = []
    api = nd.Datasets(fake, sleep=slept.append)
    assert set(api.genome_reports(["GCA_1.1", "GCA_2.1", "GCA_9.1"])) == {"GCA_1.1", "GCA_2.1"}
    assert sum("dataset_report" in c for c in fake.calls) == 3          # one per batch
    assert api.lineages([562]) == {562: {562, 561, 1}}
    assert api.fasta("GCA_1.1") == b">a\nACGT\n>b\nACGT\n"
    rep = api.genome_reports(["GCA_1.1"])["GCA_1.1"]
    assert nd.report_numbers(rep) == ("2", "8", 562, "E. coli")
    tries = iter([urllib.error.HTTPError("u", 429, "busy", {}, None), b'{"reports": []}'])

    def flaky(url):
        r = next(tries)
        if isinstance(r, Exception):
            raise r
        return r
    assert nd.Datasets(flaky, sleep=slept.append, wait=1).lineages([1]) == {}
    assert 1 in slept                                                   # waited once


def test_fetch_checks_taxon_and_assembly(tmp_path):
    import importlib.util

    from lib import ncbi_datasets as nd
    from lib.config import load_config
    spec = importlib.util.spec_from_file_location("amrtest_19", PROJECT_ROOT / "scripts" / "19_cabbage.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    config = load_config()
    config["paths_organism"] = {**config["paths_organism"], "cabbage_dir": str(tmp_path / "out"),
                                "cabbage_genomes_dir": str(tmp_path / "g" / "{organism}")}
    (tmp_path / "out").mkdir()
    pd.DataFrame({"organism": ["ecoli"] * 4, "biosample_id": ["S1", "S2", "S3", "S4"],
                  "assembly_id": ["GCA_1.1", "GCA_2.1", "GCA_3.1", "GCA_4.1"],
                  "sra_accession": "", "sources": "NCBI_antibiogram"}).to_csv(
        tmp_path / "out" / "cabbage_isolates.csv", index=False)
    two = b">a\nACGT\n>b\nACGT\n"
    fake = FakeNcbi({"GCA_1.1": {"tax": 562, "name": "Escherichia coli", "contigs": 2, "length": 8, "fasta": two},
                     "GCA_2.1": {"tax": 1313, "name": "Streptococcus pneumoniae", "contigs": 2, "length": 8, "fasta": two},
                     "GCA_3.1": {"tax": 562, "name": "Escherichia coli", "contigs": 3, "length": 8, "fasta": two}},
                    {562: [1, 561], 1313: [1, 1301]})
    api = nd.Datasets(fake, sleep=lambda s: None)
    r = m.fetch(config, workers=2, api=api).set_index("biosample_id")
    from lib import contract
    assert contract.validate_csv(tmp_path / "out" / "cabbage_download_report.csv",
                                 contract.load()["tables"]["cabbage_download_report"]) == []
    assert r.loc["S1", "status"] == "passed" and r.loc["S1", "bytes"] == len(two)
    assert r.loc["S2", "status"] == "excluded" and "1313" in r.loc["S2", "problem"]
    assert r.loc["S3", "status"] == "failed" and "the record has 3" in r.loc["S3", "problem"]
    assert r.loc["S4", "status"] == "failed" and r.loc["S4", "problem"] == "no NCBI genome report"
    assert (tmp_path / "g" / "ecoli" / "S1.fna").read_bytes() == two
    n_downloads = sum("/download" in c for c in fake.calls)
    m.fetch(config, workers=2, api=api)                                # S1 is not fetched again
    assert sum("/download" in c for c in fake.calls) == n_downloads + 1   # only S3 again

    # a fetch stopped before writing its report (the login node's CPU limit) resumes from the
    # partial report: S1's row is there, so S1 is not fetched again
    out = tmp_path / "out"
    rep = pd.read_csv(out / "cabbage_download_report.csv", dtype=str, keep_default_na=False)
    rep[rep["biosample_id"] == "S1"].to_csv(out / "cabbage_download_report.partial.csv", index=False)
    (out / "cabbage_download_report.csv").unlink()
    n_downloads = sum("/download" in c for c in fake.calls)
    m.fetch(config, workers=2, api=api)
    assert sum("/download" in c for c in fake.calls) == n_downloads + 1   # only S3 again
    assert not (out / "cabbage_download_report.partial.csv").exists()     # removed once complete


# ---- the final models on the external isolates (lib/cabbage_predict.py) --------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import numpy as np
    import xgboost as xgb
    from lib import matrix_store
    tmp = tmp_path_factory.mktemp("cabbage_model")
    rng = np.random.default_rng(1)
    n = 120
    ids = [f"g{i:03d}" for i in range(n)]
    y = (rng.random(n) < 0.5).astype(int)
    rows = [("ACGTSIGA", y), ("ACGTSIGB", y)]                    # one pattern, two members
    rows += [(f"ACGTNOISE{j}", (rng.random(n) < 0.5).astype(int)) for j in range(20)]
    with open(tmp / "u.rtab", "w") as f:
        f.write("Unitig_sequence\t" + "\t".join(ids) + "\n")
        for seq, bits in rows:
            f.write(seq + "\t" + "\t".join(map(str, bits)) + "\n")
    matrix_store.build_store(tmp / "u.rtab", tmp / "store", min_support=5)
    genomes = pd.DataFrame({"Genome ID": ids, "label": y, "lineage": np.arange(n) // 6})
    matrix_store.build_model_matrix(tmp / "store", genomes, tmp / "model", min_support=5)
    mm, store = matrix_store.ModelMatrix(tmp / "model"), matrix_store.Store(tmp / "store")
    x = mm.rows(np.arange(n)).astype(np.float32)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2, "eta": 0.5, "seed": 0},
                        xgb.DMatrix(x, label=y), 10)
    return mm, store, booster


def test_patterns_presence_and_prediction(trained):
    from lib import cabbage_predict as cp
    mm, store, booster = trained
    used = cp.used_patterns(booster)
    q = cp.query_table(mm, store, used)
    sig = set(q.loc[q["sequence"].str.startswith("ACGTSIG"), "pattern_id"])
    assert len(sig) == 1 and q["sequence"].str.startswith("ACGTSIG").sum() == 2
    rtab = pd.DataFrame({"e1": [1, 1], "e2": [1, 0], "e3": [0, 0]},
                        index=["ACGTSIGA", "ACGTSIGB"]).astype("uint8")
    pres = cp.pattern_presence(q, rtab)
    (pid,) = sig
    assert pres.loc[pid].tolist() == [1, 1, 0]                 # half of the members is enough
    assert pres.drop(index=pid).to_numpy().sum() == 0          # members the Rtab lacks: absent
    p = cp.predict(booster, mm.n_patterns, pres)
    assert p["e1"] == p["e2"] and p["e1"] > 0.5 > p["e3"]
    with pytest.raises(ValueError, match="features"):
        cp.predict(booster, mm.n_patterns + 1, pres)


def test_read_rtab(tmp_path):
    from lib import cabbage_predict as cp
    (tmp_path / "q.rtab").write_text("Unitig_sequence\tS1\tS2\nACGT\t1\t0\nTTGA\t0\t0\n")
    r = cp.read_rtab(tmp_path / "q.rtab")
    assert r.loc["ACGT"].tolist() == [1, 0] and list(r.columns) == ["S1", "S2"]


def test_external_metrics_and_bootstrap():
    import numpy as np
    from lib import cabbage_predict as cp
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.1])
    m = cp.metrics(y, p)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["very_major_error_rate"] == pytest.approx(1 / 3) and m["pr_auc_baseline"] == 0.5
    assert m["roc_auc"] == pytest.approx(8 / 9)
    w = cp.metrics(y, p, w=[2, 1, 1, 1, 1, 1])                   # weights act as copies
    yy, pp = np.r_[y, 1], np.r_[p, 0.9]
    assert w["roc_auc"] == pytest.approx(cp.metrics(yy, pp)["roc_auc"])
    lineage = ["a", "a", "b", "b", "c", "c"]
    b = cp.bootstrap(y, p, lineage, n_boot=200)
    assert b["n_lineages"] == 3 and b["roc_auc"]["low"] <= m["roc_auc"] <= b["roc_auc"]["high"]
    small = cp.assess(y, p, lineage, min_per_class=10)
    assert small == {"n": 6, "n_resistant": 3, "n_susceptible": 3, "assessed": False}
    big = cp.assess(y, p, lineage, min_per_class=3, n_boot=50)
    assert big["assessed"] and "intervals" in big and big["balanced_accuracy"] == pytest.approx(2 / 3)


def test_prepare_and_predict(trained, tmp_path, monkeypatch):
    import importlib.util
    import json

    from lib import cabbage_predict as cp
    from lib.config import load_config
    mm, store, booster = trained
    spec = importlib.util.spec_from_file_location("amrtest_19b", PROJECT_ROOT / "scripts" / "19_cabbage.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    cv = tmp_path / "cv"
    (cv / "final").mkdir(parents=True)
    booster.save_model(str(cv / "final" / "model.ubj"))
    (cv / "cv_design.json").write_text(json.dumps({"evaluable": True}))
    (cv / "metrics.json").write_text(json.dumps(
        {"arms": {"lineage_aware": {"roc_auc": 0.9, "ci": {"low": 0.8, "high": 0.95}}}}))
    (tmp_path / "panel").mkdir()
    pd.DataFrame({"organism": ["ecoli"], "antibiotic": ["ciprofloxacin"],
                  "decision": ["included"]}).to_csv(tmp_path / "panel" / "panel_decisions.csv",
                                                    index=False)
    config = load_config()
    config["paths_organism"] = {
        **config["paths_organism"], "panel_dir": str(tmp_path / "panel"), "cv_dir": str(cv),
        "matrix_dir": str(mm.dir), "unitig_store_dir": str(store.dir),
        "cabbage_dir": str(tmp_path / "cab"), "cabbage_organism_dir": str(tmp_path / "cab" / "{organism}"),
        "cabbage_genomes_dir": str(tmp_path / "g" / "{organism}")}
    out = tmp_path / "cab" / "ecoli"
    (out / "checkm2").mkdir(parents=True)
    ext = [f"E{i}" for i in range(24)]                  # E0–E11 resistant, E12–E23 susceptible
    pd.DataFrame({"organism": "ecoli", "biosample_id": ext + ["X"], "assembly_id": "GCA_1.1",
                  "status": ["passed"] * 24 + ["failed"]}).to_csv(
        tmp_path / "cab" / "cabbage_download_report.csv", index=False)
    pd.DataFrame({"Name": ext, "Completeness": 99.0,
                  "Contamination": [1.0] * 23 + [9.0]}).to_csv(out / "checkm2" / "quality_report.tsv",
                                                               sep="\t", index=False)
    pd.DataFrame({"organism": "ecoli", "biosample_id": ext, "antibiotic": "ciprofloxacin",
                  "label": [1] * 12 + [0] * 12, "n_records": 1, "how": "single"}).to_csv(
        tmp_path / "cab" / "cabbage_phenotypes.csv", index=False)
    s = m.prepare("ecoli", config)
    assert s["n_qc_passed"] == 23 and s["models"] == ["ciprofloxacin"]
    assert len((out / "refs.txt").read_text().splitlines()) == 23
    assert {"ACGTSIGA", "ACGTSIGB"} <= set((out / "unitigs.txt").read_text().split())
    # an organism without a final model has no unitig store; prepare records no models
    (tmp_path / "cab" / "kpneumoniae" / "checkm2").mkdir(parents=True)
    pd.DataFrame({"Name": ["K0"], "Completeness": 99.0, "Contamination": 1.0}).to_csv(
        tmp_path / "cab" / "kpneumoniae" / "checkm2" / "quality_report.tsv", sep="\t", index=False)
    no_store = {**config, "paths_organism": {**config["paths_organism"],
                                             "unitig_store_dir": str(tmp_path / "missing")}}
    k = m.prepare("kpneumoniae", no_store)
    assert k["models"] == [] and k["n_unitigs"] == 0
    assert json.loads((tmp_path / "cab" / "kpneumoniae" / "cabbage_prepare.json").read_text())["models"] == []
    # unitig-caller --simple as it really behaves (1.3.2): it skips the first line of --unitigs
    # as a header, ignores --rtab, writes <out>.pyseer and leaves out a unitig found in no
    # genome (E12–E23 carry neither signal unitig); it also records what it searched
    fake = tmp_path / "unitig-caller"
    fake.write_text(f"#!{sys.executable}\nimport sys\na = sys.argv\n"
                    "q = open(a[a.index('--unitigs') + 1]).read().split('\\n')[1:]\n"
                    "open(a[a.index('--out') + 1] + '.searched', 'w').write('\\n'.join(u for u in q if u))\n"
                    "open(a[a.index('--out') + 1] + '.pyseer', 'w').write(''.join(\n"
                    "    u + ' | ' + ' '.join(f'E{i}:1' for i in range(12)) + '\\n'\n"
                    "    for u in ('ACGTSIGA', 'ACGTSIGB') if u in q))\n")
    fake.chmod(0o755)
    monkeypatch.setenv("AMR_UNITIG_CALLER_BIN", str(fake))
    rtab = cp.read_rtab(m.call("ecoli", config, threads=1))
    assert sorted(rtab.columns) == sorted(ext[:23])
    assert len(rtab) == len((out / "unitigs.txt").read_text().split())
    # every unitig is searched, the first one too
    assert (out / "calls.searched").read_text().split() == (out / "unitigs.txt").read_text().split()
    assert rtab.loc["ACGTSIGA"].sum() == 12 and rtab.drop(["ACGTSIGA", "ACGTSIGB"]).sum().sum() == 0
    (out / "poppunk").mkdir()
    pd.DataFrame({"Taxon": ext[:22], "Cluster": [str(i % 4) if i % 2 else "999" for i in range(22)]}
                 ).to_csv(out / "poppunk" / "poppunk_clusters.csv", index=False)
    r = m.predict("ecoli", config)["pairs"]["ciprofloxacin"]
    assert r["n_not_called"] == 1 and r["n_not_assigned"] == 1           # E23 QC, E22 lineage
    a = r["all"]
    assert a["assessed"] and a["n_resistant"] == 12 and a["n_susceptible"] == 10
    assert a["roc_auc"] == 1.0 and a["balanced_accuracy"] == 1.0
    assert r["lineage_seen"]["n"] == 11 and r["lineage_unseen"]["n"] == 11
    assert r["internal_lineage_aware"]["roc_auc"] == 0.9
    pred = pd.read_csv(out / "cabbage_predictions.csv")
    assert len(pred) == 22 and set(pred["lineage_seen"]) == {True, False}
    # the genotype-based predictors on the same isolates (compare)
    (out / "external").mkdir()
    res = [f"E{i}" for i in range(12)]
    pd.DataFrame({"genome_id": res[:9], "element_symbol": "qnrS1", "type": "AMR", "subtype": "AMR",
                  "scope": "core", "class": "QUINOLONE", "subclass": "QUINOLONE"}).to_csv(
        out / "external" / "amrfinder_calls.csv", index=False)
    (out / "external" / "amrfinder_catalog.tsv").write_text("class\tsubclass\nQUINOLONE\tQUINOLONE\n")
    pd.DataFrame({"genome_id": ext[:22], "antibiotic": "ciprofloxacin",
                  "resistant": [1] * 12 + [1] + [0] * 9}).to_csv(
        out / "external" / "resfinder_calls.csv", index=False)
    t = m.compare("ecoli", config).set_index("tool")
    assert t.loc["amrfinderplus", "sensitivity"] == pytest.approx(9 / 12)
    assert t.loc["resfinder", "specificity"] == pytest.approx(9 / 10)
    assert "rgi" not in t.index
    assert t.loc["model", "balanced_accuracy"] == 1.0
    assert t.loc["amrfinderplus", "sensitivity_low"] <= 9 / 12 <= t.loc["amrfinderplus", "sensitivity_high"]
    # the outputs follow the output contract
    from lib import contract
    tables = contract.load()["tables"]
    for tid, f in (("cabbage_qc", "cabbage_qc.csv"), ("cabbage_predictions", "cabbage_predictions.csv"),
                   ("cabbage_comparison", "cabbage_comparison.csv")):
        assert contract.validate_csv(out / f, tables[tid]) == [], tid
