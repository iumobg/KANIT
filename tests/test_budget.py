#!/usr/bin/env python3
"""Feature budget of the unitig–k-mer comparison (lib.budget, m3_budget.py, the 04 hook) on synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import budget, cpss, folds, kmers, matrix_store  # noqa: E402

pytestmark = pytest.mark.unit


def brute_topk(bits, y, mask, k):
    """Top k by χ² over the masked genomes, ties to the smaller id, the plain way."""
    y, mask = np.asarray(y, bool), np.asarray(mask, bool)
    a = bits[:, mask & y].sum(axis=1)
    c = bits[:, mask & ~y].sum(axis=1)
    x = budget.chi2(a, c, int((mask & y).sum()), int((mask & ~y).sum()))
    return np.lexsort((np.arange(bits.shape[0]), -x))[:k], x


def test_topk_equals_a_full_sort_in_every_set_with_ties():
    rng = np.random.default_rng(0)
    n, p, k = 40, 200, 7
    bits = (rng.random((p, n)) < 0.4).astype(np.uint8)
    bits[100:110] = bits[3]                       # duplicates: ties at any rank
    y = rng.random(n) < 0.5
    masks = rng.random((5, n)) < 0.8
    t = budget.TopK(y, masks, k)
    for s in range(0, p, 13):
        t.update(np.arange(s, min(s + 13, p)), bits[s:s + 13])
    got = t.result([f"s{i}" for i in range(5)])
    for i in range(5):
        want, x = brute_topk(bits, y, masks[i], k)
        g = got[got["set"] == f"s{i}"]
        assert g["feature_id"].tolist() == want.tolist()
        assert np.allclose(g["chi2"], x[want])
        assert g["rank"].tolist() == list(range(1, k + 1))
    assert t.n_features == p


def test_topk_keeps_everything_below_the_budget():
    t = budget.TopK([1, 0, 1, 0], np.ones((1, 4), bool), 10)
    t.update([5, 9], np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.uint8))
    assert sorted(t.result(["final"])["feature_id"]) == [5, 9]


@pytest.fixture
def arm_a(tmp_path):
    """Arm A of one model: unitig store, model matrix and a 2-repeat, 3-fold design."""
    rng = np.random.default_rng(1)
    n = 60
    ids = [f"g{i:02d}" for i in range(n)]
    y = (rng.random(n) < 0.45).astype(int)
    rows = [(f"U{j}", (rng.random(n) < rng.uniform(0.2, 0.8)).astype(int)) for j in range(300)]
    rows += [("SIG", y), ("SIG2", y)]                       # collapse into one pattern
    rtab = tmp_path / "u.rtab"
    with open(rtab, "w") as f:
        f.write("Unitig_sequence\t" + "\t".join(ids) + "\n")
        for seq, b in rows:
            f.write(seq + "\t" + "\t".join(map(str, b)) + "\n")
    matrix_store.build_store(rtab, tmp_path / "store", min_support=3)
    genomes = pd.DataFrame({"Genome ID": ids, "label": y, "lineage": np.arange(n) // 3})
    matrix_store.build_model_matrix(tmp_path / "store", genomes, tmp_path / "A_matrix",
                                    min_support=3)
    cv = tmp_path / "A_cv"
    cv.mkdir()
    parts = []
    for arm in folds.ARMS:
        for r in (1, 2):
            parts.append(pd.DataFrame({"repeat": r, "arm": arm,
                                       "fold": rng.permutation(np.arange(n) % 3),
                                       "genome_id": ids}))
    pd.concat(parts).to_csv(cv / "folds.csv", index=False)
    seeds = pd.DataFrame([{"arm": a, "repeat": r, "seed": 7, "attempts": 1,
                           "evaluable": not (a == folds.LINEAGE_BLIND and r == 2)}
                          for a in folds.ARMS for r in (1, 2)])
    seeds.to_csv(cv / "seeds.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(cv / "fold_composition.csv", index=False)
    (cv / "cv_design.json").write_text(json.dumps({"evaluable": True}))
    return tmp_path


def _config(root: Path, representation: str, k: int) -> dict:
    t = str(root)
    return {"m3": {"representation": representation, "feature_budget": k},
            "unitig": {"min_support": 3, "k": 9},
            "paths_organism": {
                "source_matrix_dir": f"{t}/A_matrix", "source_cv_dir": f"{t}/A_cv",
                "matrix_dir": f"{t}/arm/matrix", "cv_dir": f"{t}/arm/cv",
                "kmer_store_dir": f"{t}/kmer_store"}}


def test_training_sets_follow_the_design(arm_a):
    mm = matrix_store.ModelMatrix(arm_a / "A_matrix")
    ids = mm.genomes["Genome ID"].astype(str)
    sets = budget.training_sets(arm_a / "A_cv/folds.csv", arm_a / "A_cv/seeds.csv", ids)
    names = [s for s, _ in sets]
    assert len(names) == 3 * 3 + 1 and names[-1] == budget.FINAL      # blind r2 not evaluable
    assert folds.unit_name(folds.LINEAGE_BLIND, 2, 0) not in names
    ft = pd.read_csv(arm_a / "A_cv/folds.csv", dtype={"genome_id": str})
    f = ft[(ft["arm"] == folds.LINEAGE_AWARE) & (ft["repeat"] == 2)].set_index("genome_id")
    mask = dict(sets)[folds.unit_name(folds.LINEAGE_AWARE, 2, 1)]
    assert mask.tolist() == (f.loc[ids, "fold"] != 1).tolist()


def test_arm_b_final_choice_is_the_cpss_prefilter_and_units_see_their_own(arm_a, load_script):
    r = load_script("m3_budget.py")
    cfg = _config(arm_a, "unitig", 20)
    out = r.run("org", "ab", cfg)
    src = matrix_store.ModelMatrix(arm_a / "A_matrix")
    fin = matrix_store.ModelMatrix(arm_a / "arm/matrix")
    pre = cpss.chi2_prefilter(src, 20)
    src_ids = pd.read_csv(arm_a / "arm/matrix/patterns.csv")["source_id"].to_numpy()
    assert sorted(src_ids.tolist()) == sorted(pre["pattern_id"].tolist())
    assert np.array_equal(fin.columns(np.arange(fin.n_patterns)), src.columns(src_ids))
    mem = fin.members()                                        # unitigs follow their pattern
    sm = src.members().set_index("unitig_index")["pattern_id"]
    assert (src_ids[mem["pattern_id"]] == sm.loc[mem["unitig_index"]].to_numpy()).all()
    assert out["n_final"] == 20 and out["n_sets"] == 10
    sel = pd.read_csv(arm_a / "arm/matrix/selection.csv.gz")
    ids = src.genomes["Genome ID"].astype(str)
    for name, mask in budget.training_sets(arm_a / "A_cv/folds.csv", arm_a / "A_cv/seeds.csv",
                                           ids)[:-1]:
        want, _ = brute_topk(src.columns(np.arange(src.n_patterns)).T, src.labels, mask, 20)
        assert sorted(sel.loc[sel["set"] == name, "feature_id"]) == sorted(want.tolist())
        um = budget.unit_matrix(arm_a / "arm/matrix", name)
        assert um.n_patterns == 20
        assert np.array_equal(um.rows(np.arange(5)), src.rows(np.arange(5))[:, np.sort(want)])
        assert np.array_equal(um.present_in(src.labels.astype(bool)),
                              src.present_in(src.labels.astype(bool))[np.sort(want)])
    for name in ("folds.csv", "seeds.csv", "fold_composition.csv", "cv_design.json"):
        assert (arm_a / "arm/cv" / name).read_bytes() == (arm_a / "A_cv" / name).read_bytes()
    assert budget.has_budget(arm_a / "arm/matrix") and not budget.has_budget(arm_a / "A_matrix")


def test_arm_c_chooses_raw_kmers_over_the_model_genomes(arm_a, tmp_path, load_script):
    rng = np.random.default_rng(5)
    mm = matrix_store.ModelMatrix(arm_a / "A_matrix")
    ids = mm.genomes["Genome ID"].astype(str).tolist()
    store = tmp_path / "kstore_src"
    (store / "call").mkdir(parents=True)
    extra = ["x1", "x2", "x3"]                       # genomes of the organism, not the model
    allg = extra + ids
    pd.DataFrame({"Genome ID": allg}).to_csv(store / "genomes.csv", index=False)
    (store / "store_summary.json").write_text("{}")
    blocks = ["".join(rng.choice(list("ACGT"), 12)) for _ in range(30)]
    gdir = tmp_path / "g"
    gdir.mkdir()
    seqs = {}
    for g in allg:
        seqs[g] = "".join(b for b in blocks if rng.random() < 0.5) or blocks[0]
        (gdir / f"{g}.fna").write_text(f">{g}\n{seqs[g]}\n")
    with open(store / "call/unitigs.rtab", "w") as f:
        f.write("u\t" + "\t".join(allg) + "\n")
        for g in allg:
            f.write(seqs[g] + "\t" + "\t".join("1" for _ in allg) + "\n")
    kmers.build_store(store, [gdir / f"{g}.fna" for g in allg], arm_a / "kmer_store", k=9,
                      min_support=1)
    r = load_script("m3_budget.py")
    r.run("org", "ab", _config(arm_a, "kmer", 15))
    ks = matrix_store.Store(arm_a / "kmer_store")
    cols = [ks.genomes.index(g) for g in ids]
    bits = np.unpackbits(np.asarray(ks.presence), axis=1)[:, cols]
    cnt = bits.sum(axis=1)
    passing = np.flatnonzero((cnt >= 3) & (len(ids) - cnt >= 3))       # the model's filter
    want, _ = brute_topk(bits[passing], mm.labels, np.ones(len(ids), bool), 15)
    fin = matrix_store.ModelMatrix(arm_a / "arm/matrix")
    p = pd.read_csv(arm_a / "arm/matrix/patterns.csv")
    assert sorted(p["source_id"]) == sorted(passing[want].tolist())
    assert (p["n_members"] == 1).all() and fin.n_patterns == 15
    assert np.array_equal(fin.columns(np.arange(15)), bits[p["source_id"]].T)
    assert fin.members()["unitig_index"].tolist() == p["source_id"].tolist()


def test_budget_never_writes_over_arm_a(arm_a, load_script):
    r = load_script("m3_budget.py")
    cfg = _config(arm_a, "unitig", 20)
    cfg["paths_organism"]["matrix_dir"] = str(arm_a / "A_matrix")
    with pytest.raises(SystemExit):
        r.run("org", "ab", cfg)
    assert not (arm_a / "A_matrix" / budget.UNION).exists()


def test_nested_cv_trains_an_outer_fold_on_its_own_features(arm_a, load_script):
    pytest.importorskip("xgboost")
    pytest.importorskip("optuna")
    b = load_script("m3_budget.py")
    b.run("org", "ab", _config(arm_a, "unitig", 20))
    r = load_script("04_nested_cv.py")
    mm = matrix_store.ModelMatrix(arm_a / "arm/matrix")
    um = r.unit_matrix(mm, folds.LINEAGE_AWARE, 1, 0)
    assert isinstance(um, budget.SubsetMatrix) and um.fingerprint != mm.fingerprint
    assert r.unit_matrix(matrix_store.ModelMatrix(arm_a / "A_matrix"),
                         folds.LINEAGE_AWARE, 1, 0).n_patterns > 20      # arm A unchanged
    hpo = {"n_trials": 2, "search_max_genomes": 60, "pruner_startup_trials": 1,
           "pruner_warmup_rounds": 5, "early_stopping_rounds": 5, "max_rounds": 30,
           "batch_rows": 25}
    out = arm_a / "arm/cv"
    r.run_unit(um, out, folds.LINEAGE_AWARE, 1, 0, hpo, threads=1)
    unit = out / "units" / folds.unit_name(folds.LINEAGE_AWARE, 1, 0)
    oof = pd.read_csv(unit / "oof.csv")
    assert len(oof) == 20 and oof["p"].between(0, 1).all()
    assert r.unit_is_current(um, out, folds.LINEAGE_AWARE, 1, 0, hpo)


def test_pyseer_of_a_budgeted_arm_takes_population_structure_from_arm_a(arm_a, load_script,
                                                                         tmp_path):
    from lib import pyseer_lmm as pl
    load_script("m3_budget.py").run("org", "ab", _config(arm_a, "unitig", 20))
    src = matrix_store.ModelMatrix(arm_a / "A_matrix")
    arm = matrix_store.ModelMatrix(arm_a / "arm/matrix")
    pl.write_inputs(arm, tmp_path / "a", [0, 3], every=10, background_max=50, structure=src)
    pl.write_inputs(src, tmp_path / "b", [0], every=10, background_max=50)
    for name in ("kinship.tsv", "background.Rtab", "phenotypes.tsv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    tested = pd.read_csv(tmp_path / "a/tested.Rtab", sep="\t", index_col=0)
    assert np.array_equal(tested.to_numpy(), arm.columns([0, 3]).T)
    other = matrix_store.ModelMatrix(arm_a / "A_matrix")
    other.genomes = other.genomes.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        pl.write_inputs(arm, tmp_path / "c", [0], every=10, background_max=50, structure=other)


def test_mda_permutes_a_candidate_in_every_fold_model_that_chose_it(arm_a, load_script):
    pytest.importorskip("xgboost")
    from lib import mda
    load_script("m3_budget.py").run("org", "ab", _config(arm_a, "unitig", 20))
    r = load_script("04_nested_cv.py")
    mm = matrix_store.ModelMatrix(arm_a / "arm/matrix")
    out = arm_a / "arm/cv"
    hpo = {"n_trials": 2, "search_max_genomes": 60, "pruner_startup_trials": 1,
           "pruner_warmup_rounds": 5, "early_stopping_rounds": 5, "max_rounds": 30,
           "batch_rows": 25}
    for rep in (1, 2):
        for fold in range(3):
            r.run_unit(r.unit_matrix(mm, folds.LINEAGE_AWARE, rep, fold), out,
                       folds.LINEAGE_AWARE, rep, fold, hpo, threads=1)
    models = mda.load_fold_models(mm, out)          # reproduces every fold's oof.csv
    assert len(models) == 6
    final_src = pd.read_csv(arm_a / "arm/matrix/patterns.csv")["source_id"].to_numpy()
    union_src = pd.read_csv(arm_a / "arm/matrix/cv_union/patterns.csv")["source_id"].to_numpy()
    for fm in models:
        um = budget.unit_matrix(mm.dir, folds.unit_name(folds.LINEAGE_AWARE, fm.repeat, fm.fold))
        src = union_src[um.cols[fm.used]]
        for pid, s in zip(fm.ids, src, strict=True):
            assert (pid == -1 and s not in final_src) or final_src[pid] == s
    design = mda.Design(models, 20, 0, mm.labels.astype(int))
    used = {int(i) for fm in models for i in fm.ids if i >= 0}
    if used:
        g = np.array([min(used)])
        _, n_used = mda.permuted_auc(design, g)
        assert n_used == sum(int(g[0] in fm.ids) for fm in models) > 0
