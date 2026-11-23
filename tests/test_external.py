#!/usr/bin/env python3
"""Comparison with genotype-based prediction (lib.external, 16_external.py)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import external as ex  # noqa: E402
from lib import matrix_store  # noqa: E402

pytestmark = pytest.mark.unit

AFP_V4 = ("Protein id\tContig id\tStart\tStop\tStrand\tElement symbol\tElement name\tScope\t"
          "Type\tSubtype\tClass\tSubclass\tMethod\n")


def _afp(path, rows):
    path.write_text(AFP_V4 + "".join(
        f"NA\tc1\t1\t9\t+\t{sym}\tname\t{scope}\tAMR\t{sub}\t{cls}\t{subcls}\tEXACTX\n"
        for sym, scope, sub, cls, subcls in rows))


def test_readers(tmp_path):
    _afp(tmp_path / "a.tsv", [("blaTEM-1", "core", "AMR", "BETA-LACTAM", "BETA-LACTAM"),
                              ("gyrA_S83L", "core", "POINT", "QUINOLONE", "QUINOLONE")])
    d = ex.read_amrfinder(tmp_path / "a.tsv", "g1")
    assert list(d.columns) == ex.CALL_COLUMNS and d["element_symbol"].tolist() == [
        "blaTEM-1", "gyrA_S83L"] and (d["genome_id"] == "g1").all()
    (tmp_path / "v3.tsv").write_text("Gene symbol\tScope\tElement type\tElement subtype\tClass\t"
                                     "Subclass\nqnrS1\tcore\tAMR\tAMR\tQUINOLONE\tQUINOLONE\n")
    assert ex.read_amrfinder(tmp_path / "v3.tsv", "g2")["element_symbol"].tolist() == ["qnrS1"]
    (tmp_path / "cat.tsv").write_text("allele\tclass\tsubclass\nx\tBETA-LACTAM\tCEPHALOSPORIN\n"
                                      "y\tQUINOLONE\tQUINOLONE/FLUOROQUINOLONE\n")
    assert ex.catalog_tokens(tmp_path / "cat.tsv") == {"BETA-LACTAM", "CEPHALOSPORIN",
                                                       "QUINOLONE", "FLUOROQUINOLONE"}
    (tmp_path / "pheno.txt").write_text(
        "# ResFinder\n# Antimicrobial\tClass\tWGS-predicted phenotype\tMatch\tGenetic background\n"
        "amoxicillin+clavulanic acid\tbeta-lactam\tResistant\t3\tblaTEM-1\n"
        "ciprofloxacin\tquinolone\tNo resistance\t0\t\n"
        "ampicillin\tbeta-lactam\tResistant\t3\tblaTEM-1\n")
    assert ex.read_resfinder(tmp_path / "pheno.txt", ["amoxicillin_clavulanic_acid",
                                                     "ciprofloxacin", "ampicillin_sulbactam"]) == {
        "amoxicillin_clavulanic_acid": 1, "ciprofloxacin": 0}


def test_predictions_and_metrics():
    g = ["g1", "g2", "g3", "g4"]
    calls = pd.DataFrame([("g1", "gyrA_S83L", "AMR", "POINT", "core", "QUINOLONE", "QUINOLONE"),
                          ("g2", "qnrS1", "AMR", "AMR", "plus", "QUINOLONE", "QUINOLONE")],
                         columns=ex.CALL_COLUMNS)
    p = ex.predict_amrfinder(calls, g, {"QUINOLONE"}, {"QUINOLONE"})
    assert p.tolist() == [1, 0, 0, 0]                    # plus scope does not count
    assert ex.predict_amrfinder(calls, g, {"MONOBACTAM"}, {"QUINOLONE"}) is None
    oof = pd.DataFrame({"arm": ["lineage_aware"] * 8 + ["lineage_blind"] * 4,
                        "repeat": [1] * 4 + [2] * 4 + [1] * 4, "genome_id": g * 3,
                        "p": [0.9, 0.4, 0.6, 0.1, 0.8, 0.7, 0.3, 0.2, 0, 0, 0, 0]})
    assert ex.predict_model(oof, g, 0.5).tolist() == [1, 1, 0, 0]          # means .85 .55 .45 .15
    m = ex.metrics(pd.Series([1, 1, 0, 0], index=g), pd.Series([1, 0, 1, 0], index=g))
    assert (m["sensitivity"], m["specificity"], m["very_major_error_rate"],
            m["major_error_rate"]) == (0.5, 0.5, 0.5, 0.5)
    rows = ex.compare_model("o__a", pd.Series([1, 1, 0, 0], index=g), {"model": p})
    assert rows["tool"].tolist() == list(ex.TOOLS) and rows["assessable"].sum() == 1


def _script():
    spec = importlib.util.spec_from_file_location("amrtest_16", PROJECT_ROOT / "scripts" /
                                                  "16_external.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_step_end_to_end(tmp_path):
    root = tmp_path
    config = {"paths_organism": {k: str(root / v) for k, v in {
        "panel_dir": "panel", "genome_qc_dir": "{organism}/qc", "lineage_dir": "{organism}/lin",
        "metadata_file": "{organism}/pheno.csv", "raw_genomes_dir": "{organism}/genomes",
        "external_dir": "{organism}/external",
        "unitig_store_dir": "{organism}/store", "matrix_dir": "{organism}/{antibiotic}/mm",
        "cv_dir": "{organism}/{antibiotic}/cv"}.items()},
        "card": {"near_universal": 0.95}, "cv": {"threshold": 0.5}}
    ids = [f"g{i}" for i in range(12)]
    y = np.array([1] * 6 + [0] * 6)
    org = root / "ecoli"
    for d in ("qc", "lin"):
        (org / d).mkdir(parents=True)
    (root / "panel").mkdir()
    pd.DataFrame({"organism": ["ecoli"], "antibiotic": ["ciprofloxacin"],
                  "decision": ["included"]}).to_csv(root / "panel" / "panel_decisions.csv",
                                                    index=False)
    pd.DataFrame({"Genome ID": ids, "ciprofloxacin": y}).to_csv(org / "pheno.csv", index=False)
    pd.DataFrame({"genome_id": ids, "pass_overall": True}).to_csv(
        org / "qc" / "02d_genome_qc_ecoli.csv", index=False)
    pd.DataFrame({"Genome ID": ids, "Cluster": range(12)}).to_csv(
        org / "lin" / "poppunk_clusters.csv", index=False)
    m = _script()
    ext = org / "external"
    assert m.prep("ecoli", config, ext, threads=2) == sorted(ids)
    script = (ext / "run_external.sh").read_text()
    assert "--organism Escherichia" in script and '-s "Escherichia coli"' in script
    assert "--point --ignore_missing_species" in script and "amrfinder_catalog.tsv" in script
    # run_external.sh in two shards, with stand-ins for the two tools
    db = root / "afpdb"
    db.mkdir()
    # the tables amrfinder_update fetches: class and subclass are read from every one
    # that has them, wherever the columns are
    (db / "fam.tsv").write_text("#node_id\tparent_node_id\treportable\ttype\tsubtype\tclass\t"
                                "subclass\tfamily_name\nqnrS\tQNR\t1\tAMR\tAMR\tQUINOLONE\t"
                                "QUINOLONE\tquinolone resistance protein QnrS\n")
    (db / "AMRProt-mutation.tsv").write_text("#accession\tsubclass\tclass\tsymbol\n"
                                             "WP_1\tCOLISTIN\tCOLISTIN\tpmrB_R256G\n")
    (db / "version.txt").write_text("2025-07-16.1\n")
    fake = root / "fake_amrfinder"
    fake.write_text(f"""#!/usr/bin/env bash
case "$1" in
  --version) echo 4.2.7 ;;
  --database_version) printf "Software version: 4.2.7\\nDatabase directory: '{db}'\\nDatabase version: 2025-07-16.1\\n" ;;
  *) while [ $# -gt 0 ]; do case "$1" in --name) g=$2 ;; --output) o=$2 ;; --database) d=$2 ;; esac; shift; done
     [ -n "${{d:-}}" ] && echo "$d" > "$o.db"
     n=${{g#g}}
     printf '{AFP_V4.strip()}\\n' > "$o"
     if [ "$n" -lt 5 ]; then printf 'NA\\tc1\\t1\\t9\\t+\\tgyrA_S83L\\tname\\tcore\\tAMR\\tPOINT\\tQUINOLONE\\tQUINOLONE\\tEXACTX\\n' >> "$o"; fi ;;
esac
""")
    fake_rf = root / "fake_resfinder"
    fake_rf.write_text("""#!/usr/bin/env bash
if [ "$1" = --version ]; then echo 4.5.0; exit 0; fi
while [ $# -gt 0 ]; do case "$1" in -o) o=$2 ;; -ifa) f=$2 ;; esac; shift; done
mkdir -p "$o"; g=$(basename "$f" .fna); n=${g#g}
if [ "$n" -lt 4 ]; then r=Resistant; else r='No resistance'; fi
printf 'ciprofloxacin\\tquinolone\\t%s\\t1\\tx\\n' "$r" > "$o/pheno_table_escherichia_coli.txt"
""")
    for f in (fake, fake_rf):
        f.chmod(0o755)
    import os
    import subprocess
    env = {**os.environ, "AMRFINDER": str(fake), "RESFINDER": str(fake_rf), "AMRFINDER_DB": str(db)}
    for k in (1, 0):
        subprocess.run(["bash", str(ext / "run_external.sh"), str(k), "2"], check=True, env=env)
    assert len(list((ext / "amrfinder").glob("*.done"))) == 12
    assert ex.catalog_tokens(ext / "amrfinder_catalog.tsv") >= {"QUINOLONE", "COLISTIN"}
    assert (ext / "amrfinder" / "g0.tsv.db").read_text().strip() == str(db)   # $AMRFINDER_DB
    v = m.collect("ecoli", config, ext)
    assert v["amrfinderplus_database"] == "2025-07-16.1" and v["n_genomes"] == 12
    assert len(pd.read_csv(ext / "amrfinder_calls.csv")) == 5
    # the rest of the inputs of `compare`
    with open(root / "u.rtab", "w") as f:
        f.write("Unitig_sequence\t" + "\t".join(ids) + "\n")
        f.write("ACGTACGTAC\t" + "\t".join(["1", "0"] * 6) + "\n")
    matrix_store.build_store(root / "u.rtab", org / "store", min_support=1)
    (org / "ciprofloxacin" / "mm").mkdir(parents=True)
    pd.DataFrame({"Genome ID": ids, "label": y, "lineage": range(12)}).to_csv(
        org / "ciprofloxacin" / "mm" / "genomes.csv", index=False)
    (org / "ciprofloxacin" / "cv").mkdir()
    pd.DataFrame({"arm": "lineage_aware", "repeat": 1, "fold": 0, "genome_id": ids, "y": y,
                  "p": np.where(y == 1, 0.9, 0.1)}).to_csv(
        org / "ciprofloxacin" / "cv" / "oof_predictions.csv", index=False)
    t = m.compare("ecoli", config, ext).set_index("tool")
    assert t.loc["amrfinderplus", "sensitivity"] == pytest.approx(5 / 6)
    assert t.loc["resfinder", "very_major_error_rate"] == pytest.approx(2 / 6)
    assert set(t.index) == {"amrfinderplus", "resfinder", "model"}          # RGI is no predictor
    assert t.loc["model", "balanced_accuracy"] == 1.0
    assert json.loads((ext / "versions.json").read_text())["resfinder"] == "4.5.0"
    summary = json.loads((ext / "external_summary.json").read_text())
    assert summary["n_models"] == 1 and summary["assessable_models"]["model"] == 1
    from lib import contract  # the outputs follow the output contract
    for tid, name in (("external_comparison", "comparison.csv"),
                      ("amrfinder_calls", "amrfinder_calls.csv"),
                      ("amrfinder_genomes", "amrfinder_genomes.csv"),
                      ("resfinder_calls", "resfinder_calls.csv")):
        assert contract.validate_csv(ext / name, contract.load()["tables"][tid]) == [], tid
