#!/usr/bin/env python3
"""The Nextflow skeleton connects every step: -stub-run of each entry (needs
Nextflow; runs when KANIT_RUN_NEXTFLOW=1, since it takes about a minute)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEXTFLOW = os.environ.get("NEXTFLOW", shutil.which("nextflow") or "")

pytestmark = pytest.mark.skipif(
    os.environ.get("KANIT_RUN_NEXTFLOW") != "1" or not NEXTFLOW,
    reason="set KANIT_RUN_NEXTFLOW=1 (and NEXTFLOW if nextflow is not on PATH)")


def _run(tmp_path, *args):
    cmd = [NEXTFLOW, "-q", "run", str(PROJECT_ROOT / "main.nf"), "-stub-run", "-profile", "test",
           "--python", sys.executable, "--trace_dir", str(tmp_path / "trace"), *args]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    return r, tmp_path / "trace" / "trace.tsv"


def test_default_entry_reaches_every_step(tmp_path):
    r, trace = _run(tmp_path, "--organisms", "ecoli,kpneumoniae", "--stub_panel",
                    str(PROJECT_ROOT / "tests" / "data" / "stub_panel_decisions.csv"))
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(trace, sep="\t")
    n = t["process"].str.split(":").str[-1].value_counts().to_dict()
    assert (t["status"] == "COMPLETED").all()
    # 50 outer folds and 50 permutation chunks per model, 4 per job in the stub (lib/packing.py)
    assert n["CV_UNITS"] == 3 * 13 and n["LP_CHUNKS"] == 3 * 13 and n["CPSS_CHUNK"] == 3 * 1
    for p in ("PANEL", "GRADING", "LP_ACROSS", "EXTERNAL_COMPARE", "CARD_LAYER", "PYSEER_POST"):
        assert n.get(p), p
    # the patterns nominated by association: every layer and the
    # grades of every model again
    for p in ("ASSOCIATION_SET", "PREVALENCE_ASSOCIATION", "MDA_ASSOCIATION",
              "CARD_LAYER_ASSOCIATION", "GRADING_ASSOCIATION"):
        assert n.get(p) == n["GRADING"], p
    assert n["RUN_OUTPUTS"] == n["SACCT_DUMP"] == n["RUN_RESOURCES"] == 1
    assert n["REPORTS"] == n["COMPLETENESS"] == n["COMPLETENESS_GATE"] == 1
    assert "BACKUP_PACK" not in n                     # no --backup_remote: no backup
    # the unitig store reads the panel: every store starts after the panel is written
    proc = t["process"].str.split(":").str[-1]
    assert t.loc[proc == "STORE", "start"].min() >= t.loc[proc == "PANEL", "complete"].max()
    # the run manifest and every task's receipt (its step manifest) are kept with the run
    run = json.loads((tmp_path / "trace" / "run_manifest.json").read_text())
    assert run["run"]["entry"] == "main" and run["run"]["stub"] is True
    assert len(run["code"]["commit"]) == 40 and run["completed"]["success"] is True
    assert run["config"]["config_yaml"]["sha256"]
    assert set(run["snapshots"]) == {"ecoli", "kpneumoniae"}
    receipts = list((tmp_path / "trace" / "tasks").glob("*/*/receipt.json"))
    assert len(receipts) == len(t)
    r = json.loads((tmp_path / "trace" / "tasks" / "PANEL" / "run" / "receipt.json").read_text())
    assert r["step"] == "panel" and r["process"].endswith("PANEL") and r["stub"] is True
    keys = {json.loads(f.read_text())["key"]
            for f in (tmp_path / "trace" / "tasks").glob("GRADING*/*/receipt.json")}
    assert len(keys) == 2 * n["GRADING"]
    assert sum(k.endswith(" association") for k in keys) == n["GRADING"]


def test_a_run_with_more_organisms_keeps_the_earlier_ones_cached(tmp_path):
    """The full run in two waves: the panel is rerun when the organisms of a run change,
    but an organism's store and genotype-based tools take its pairs, not the panel's
    receipt, so that adding organisms reruns nothing of the earlier ones."""
    panel = str(PROJECT_ROOT / "tests" / "data" / "stub_panel_decisions.csv")
    r, _ = _run(tmp_path, "--organisms", "ecoli", "--stub_panel", panel)
    assert r.returncode == 0, r.stdout + r.stderr
    r, trace = _run(tmp_path, "--organisms", "ecoli,kpneumoniae", "--stub_panel", panel,
                    "-resume")
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(trace, sep="\t")
    t["step"] = t["process"].str.split(":").str[-1]
    eco = t["tag"].astype(str).str.startswith("ecoli")
    assert "PANEL" in set(t.loc[t["status"] == "COMPLETED", "step"])     # the new organism
    assert {"STORE", "EXTERNAL_PREP", "MODEL_MATRIX", "CV_UNITS", "GRADING",
            "GRADING_ASSOCIATION"} <= set(t.loc[eco & (t["status"] == "CACHED"), "step"])
    assert not (eco & (t["status"] == "COMPLETED")).any(), t.loc[eco, ["step", "status"]]
    assert (~eco & (t["step"] == "STORE") & (t["status"] == "COMPLETED")).sum() == 1


def test_large_organisms_run_their_heavy_steps_on_their_partition(tmp_path):
    """-profile truba: --queue_large takes the heavy steps of --large_organisms (memory,
    cores and time of that partition), --queue_light the short steps, --queue the rest."""
    (tmp_path / "main.nf").write_text(
        "process UNITS { label 'cv_unit'; input: tuple val(meta), val(n); output: stdout\n"
        "  script: 'echo u' }\n"
        "process STORE { label 'store'; input: tuple val(org), val(abs); output: stdout\n"
        "  script: 'echo s' }\n"
        "process LIGHT { label 'light'; input: val(meta); output: stdout; script: 'echo l' }\n"
        "process MDA { label 'mda'; input: val(meta); output: stdout; script: 'echo m' }\n"
        "workflow {\n"
        "  ms = Channel.of([id: 'ecoli__a', organism: 'ecoli'], [id: 'saureus__b', organism: 'saureus'])\n"
        "  UNITS(ms.map { [it, 1] }); LIGHT(ms); MDA(ms)\n"
        "  STORE(Channel.of(['ecoli', ['a']], ['saureus', ['b']]))\n}\n")
    (tmp_path / "local.config").write_text(
        "process.executor = 'local'\nprocess.container = null\napptainer.enabled = false\n"
        "workDir = 'work'\nexecutor { cpus = 512; memory = '4 TB' }\n"
        "trace { enabled = true; overwrite = true; file = 'trace.txt'; "
        "fields = 'name,tag,queue,cpus,memory,time' }\n")
    r = subprocess.run([NEXTFLOW, "-q", "run", "main.nf", "-c",
                        str(PROJECT_ROOT / "conf" / "truba.config"), "-c", "local.config",
                        "--queue", "hamsi", "--queue_light", "debug", "--queue_large", "barbun",
                        "--large_organisms", "ecoli,kpneumoniae"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(tmp_path / "trace.txt", sep="\t")
    t["step"] = t["name"].str.split(" ").str[0]
    got = {(s, int(i)): (q, c) for s, i, q, c in zip(
        t["step"], t["name"].str.extract(r"\((\d+)\)")[0], t["queue"], t["cpus"], strict=True)}
    assert got[("UNITS", 1)] == ("barbun", 20) and got[("UNITS", 2)] == ("hamsi", 56)
    hours = dict(zip(t["name"], t["time"], strict=True))
    assert hours["UNITS (1)"] == "1d 12h" and hours["UNITS (2)"] == "1d"   # large: half as much again
    assert got[("STORE", 1)] == ("barbun", 40) and got[("STORE", 2)] == ("hamsi", 56)
    assert {q for (s, _), (q, _) in got.items() if s == "LIGHT"} == {"debug"}
    assert {q for (s, _), (q, _) in got.items() if s == "MDA"} == {"barbun", "hamsi"}
    r = subprocess.run([NEXTFLOW, "-q", "run", "main.nf", "-c",          # --queue_short
                        str(PROJECT_ROOT / "conf" / "truba.config"), "-c", "local.config",
                        "--queue", "hamsi", "--queue_light", "debug", "--queue_large", "barbun",
                        "--large_organisms", "ecoli,kpneumoniae", "--queue_short", "debug"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(tmp_path / "trace.txt", sep="\t")
    q = dict(zip(t["name"], t["queue"], strict=True))
    assert q["MDA (1)"] == q["MDA (2)"] == "debug" and q["UNITS (1)"] == "barbun"


def test_a_retry_runs_fewer_units_at_once(tmp_path):
    """A packed job that ran out of memory is retried with half as many units at once, in
    waves; units finished before are kept. Here a unit fails (137) when more than two run
    together: the first attempt fails, the second succeeds in two waves."""
    (tmp_path / "unit.sh").write_text(
        'u=$1; S=$(dirname "$0")/state; mkdir -p "$S"; [ -e "$S/done_$u" ] && exit 0\n'
        'touch "$S/run_$u"; sleep 1; n=$(ls "$S" | grep -c "^run_")\n'
        'if [ "$n" -gt 2 ]; then rm -f "$S/run_$u"; exit 137; fi\n'
        'sleep 1; rm -f "$S/run_$u"; touch "$S/done_$u"\n')
    (tmp_path / "main.nf").write_text(
        f"include {{ parallel }} from '{PROJECT_ROOT}/modules/local/common'\n"
        "process P {\n  cpus 4\n  errorStrategy { task.attempt <= 3 ? 'retry': 'terminate' }\n"
        "  maxRetries 3\n  output: path 'done.txt'\n  script:\n  \"\"\"\n"
        "  ${parallel(task, [1, 2, 3, 4], { u -> \"bash ${projectDir}/unit.sh ${u}\" })}\n"
        "  echo ok > done.txt\n  \"\"\"\n}\nworkflow { P() }\n")
    r = subprocess.run([NEXTFLOW, "-q", "run", "main.nf", "-with-trace", "trace.txt"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(tmp_path / "trace.txt", sep="\t")
    assert t["status"].tolist() == ["FAILED", "COMPLETED"]
    assert sorted(f.name for f in (tmp_path / "state").iterdir()) == [f"done_{i}" for i in range(1, 5)]


def test_parameters_are_checked(tmp_path):
    r, _ = _run(tmp_path)
    assert r.returncode != 0 and "organisms" in (r.stdout + r.stderr)
    r, _ = _run(tmp_path, "--organisms", "salmonella")
    assert r.returncode != 0 and "Unknown organism" in (r.stdout + r.stderr)


@pytest.mark.parametrize("entry, expected", [
    ("DOWNLOAD", {"DOWNLOAD_BVBRC": 2}),
    ("CONTEXT", {"CONTEXT_QUERY": 2, "CONTEXT_BUILD": 2})])
def test_internet_entries(tmp_path, entry, expected):
    expected = {**expected, "REPORTS": 1, "COMPLETENESS": 1, "RUN_OUTPUTS": 1, "BACKUP_PACK": 1,
                "BACKUP_UPLOAD": 1, "SACCT_DUMP": 1, "RUN_RESOURCES": 1, "COMPLETENESS_GATE": 1}
    r, trace = _run(tmp_path, "-entry", entry, "--organisms", "ecoli,kpneumoniae",
                    "--backup_remote", str(tmp_path / "remote"))
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(trace, sep="\t")
    assert (t["status"] == "COMPLETED").all()
    assert t["process"].str.split(":").str[-1].value_counts().to_dict() == expected


def test_cabbage_entry(tmp_path):
    """External validation: the global steps once, every per-organism step once per
    organism, the tools once per shard."""
    r, trace = _run(tmp_path, "-entry", "CABBAGE", "--organisms", "ecoli,kpneumoniae",
                    "--rgi_shards", "2", "--external_shards", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(trace, sep="\t")
    assert (t["status"] == "COMPLETED").all()
    n = t["process"].str.split(":").str[-1].value_counts().to_dict()
    assert {k: n.get(k) for k in ("CABBAGE_DOWNLOAD", "CABBAGE_SELECT", "CABBAGE_FETCH")} == \
        {"CABBAGE_DOWNLOAD": 1, "CABBAGE_SELECT": 1, "CABBAGE_FETCH": 1}
    for p in ("QC_PATHS", "CHECKM2", "PREPARE", "ASSIGN", "CALL", "PREDICT", "EXTERNAL_PREP",
              "EXTERNAL_COLLECT", "COMPARE", "LOOKUP_PREPARE", "LOOKUP_CALL",
              "LOOKUP"):
        assert n.get(f"CABBAGE_{p}") == 2, p
    assert n["CABBAGE_EXTERNAL_RUN"] == 6 and "CABBAGE_RGI" not in n
    assert n["CABBAGE_LOOKUP_SELECT"] == 1
    assert n["COMPLETENESS_GATE"] == 1


def test_internet_tasks_run_on_the_login_node(tmp_path):
    """TRUBA's compute nodes have no internet: the tasks that need it run where the
    head job runs."""
    r = subprocess.run([NEXTFLOW, "config", "-flat", "-profile", "truba", str(PROJECT_ROOT)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "process.'withLabel:internet'.executor = 'local'" in r.stdout
    for f in ("download.nf", "context.nf"):
        text = (PROJECT_ROOT / "modules" / "local" / f).read_text()
        assert text.count("label 'internet'") == {"download.nf": 1, "context.nf": 1}[f]


def test_python_output_is_unbuffered_on_truba(tmp_path):
    """The logs of a running job show its progress (the units of a packed job wrote nothing
    until they ended). Set in the env scope, which leaves the tasks' cache keys unchanged."""
    r = subprocess.run([NEXTFLOW, "config", "-flat", "-profile", "truba", str(PROJECT_ROOT)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "env.PYTHONUNBUFFERED = '1'" in r.stdout


@pytest.mark.parametrize("arm", ["B", "C"])
def test_m3_entry(tmp_path, arm):
    """Unitig–k-mer comparison: every included pair of the main run's panel once, from the budget to the
    grades; the k-mer store once per organism in arm C only; nothing of RGI, the label
    permutation test or the association set; the run's accounting at the end."""
    panel = PROJECT_ROOT / "tests" / "data" / "stub_panel_decisions.csv"
    r, trace = _run(tmp_path, "-entry", "M3", "--organisms", "ecoli,kpneumoniae",
                    "--stub_panel", str(panel),
                    "--config_overlay", str(PROJECT_ROOT / "config" / f"m3_{arm}.yml"))
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(trace, sep="\t")
    assert (t["status"] == "COMPLETED").all()
    n = t["process"].str.split(":").str[-1].value_counts().to_dict()
    d = pd.read_csv(panel)
    d = d[(d["decision"] == "included") & d["organism"].isin(["ecoli", "kpneumoniae"])]
    for p in ("M3_BUDGET", "CV_PACKING", "CV_FINAL", "CV_METRICS", "CPSS_PREFILTER",
              "CPSS_SELECT", "PREVALENCE", "MDA", "CARD_LAYER", "PYSEER_POST", "GRADING"):
        assert n.get(p) == len(d), p
    assert n.get("M3_KMER_STORE", 0) == (d["organism"].nunique() if arm == "C" else 0)
    for p in ("RGI_RUN", "RGI_COLLECT", "LP_CHUNKS", "ASSOCIATION_SET", "MODEL_MATRIX",
              "CV_FOLDS", "REPORTS"):
        assert p not in n, p
    assert n["SACCT_DUMP"] == 1 and n["RUN_RESOURCES"] == 1


def test_m3_entry_needs_an_arm(tmp_path):
    r, _ = _run(tmp_path, "-entry", "M3", "--organisms", "ecoli", "--stub_panel",
                str(PROJECT_ROOT / "tests" / "data" / "stub_panel_decisions.csv"))
    assert r.returncode != 0 and "config_overlay" in r.stdout + r.stderr
