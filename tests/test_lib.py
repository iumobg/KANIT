#!/usr/bin/env python3
"""
Unit tests for the shared library.

Run with:
    pytest tests/
or directly:
    python tests/test_lib.py
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import registry, run_metadata  # noqa: E402
from lib.config import get_target, load_config, resolve_path  # noqa: E402


def test_registry_classes_structure():
    classes = registry.load_antibiotic_classes()
    assert "Aminoglycosides" in classes
    assert "gentamicin" in classes["Aminoglycosides"]
    # legacy structure: {DisplayName: [members]}
    assert all(isinstance(v, list) for v in classes.values())


def test_registry_reverse_index():
    assert registry.antibiotic_to_class("gentamicin") == "aminoglycosides"
    assert registry.antibiotic_to_class("ciprofloxacin") == "quinolones"
    # carbapenems are their own class (schema 2.0), not lumped with beta-lactams
    assert registry.antibiotic_to_class("meropenem") == "carbapenems"
    assert registry.antibiotic_to_class("not_a_real_drug") is None


def test_registry_metadata_accessors():
    # mechanism_type resolves via class; who_aware is per-antibiotic (registry = source)
    assert registry.antibiotic_mechanism_type("ciprofloxacin") == "target_snp"
    assert registry.antibiotic_mechanism_type("meropenem") == "acquired"
    assert registry.antibiotic_mechanism_type("not_a_real_drug") is None
    assert registry.antibiotic_who_aware("colistin") == "Reserve"
    assert registry.antibiotic_who_aware("ampicillin") == "Access"
    assert registry.antibiotic_who_aware("not_a_real_drug") is None


def test_registry_single_drugs_and_path_safe_names():
    # labels that are not a single drug never enter the panel
    assert registry.is_single_drug("extended spectrum beta lactamase") is False
    assert registry.is_single_drug("Fluoroquinolones") is False
    assert registry.is_single_drug("ciprofloxacin") is True
    # canonical names are path components; other spellings fold onto them
    assert registry.normalize_antibiotic("Ceftazidime/Avibactam") == "ceftazidime_avibactam"
    assert registry.normalize_antibiotic("polymyxin B") == "polymyxin_b"
    assert registry.normalize_antibiotic("fusidic acid") == "fusidic_acid"
    assert registry.antibiotic_to_class("fusidic_acid") == "fusidanes"


# ---------------------------------------------------------------------------
# config / path resolution
# ---------------------------------------------------------------------------
def test_resolve_path_organism_antibiotic():
    p = resolve_path("matrix_dir", organism="ecoli", antibiotic="gentamicin")
    assert p.as_posix().endswith("data/processed/ecoli/gentamicin/model_matrix")


def test_resolve_path_run_id():
    p = resolve_path("run_dir", organism="ecoli", antibiotic="gentamicin", run_id="RID123")
    assert p.name == "RID123"


def test_resolve_path_global_key_no_placeholder():
    # A global key with no placeholder (data_dir) resolves directly, with or
    # without organism/antibiotic supplied.
    p = resolve_path("data_dir")
    assert p.name == "data"


def test_resolve_path_unknown_key():
    raised = False
    try:
        resolve_path("definitely_not_a_real_key")
    except KeyError:
        raised = True
    assert raised, "resolve_path should raise KeyError for an unknown key"


def test_get_target_defaults_from_config():
    org, ab = get_target()
    assert org == "ecoli"
    assert ab  # a non-empty antibiotic from config


# ---------------------------------------------------------------------------
# run metadata
# ---------------------------------------------------------------------------
def test_make_run_id_format():
    rid = run_metadata.make_run_id("ecoli", "gentamicin")
    parts = rid.split("__")
    assert len(parts) == 4
    assert parts[0] == "ecoli" and parts[1] == "gentamicin"


def test_hash_files_stable(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h1 = run_metadata.hash_files([f])
    h2 = run_metadata.hash_files([f])
    assert h1 == h2 and len(h1) == 64


if __name__ == "__main__":
    # Minimal runner so the file works without pytest installed.
    import tempfile
    import traceback
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in funcs:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


# ---- compute-resource env overrides ---------------------------------------
# threads describe machine size, not science. They used to be a hand-edit
# on the HPC that `git reset --hard` wiped on every deploy — a forgotten re-edit
# meant jobs ran with laptop resources, slowly or OOM, with no clue in the logs.

def test_resource_keys_default_to_laptop_safe_values(monkeypatch):
    monkeypatch.delenv("AMR_THREADS", raising=False)
    pre = load_config()["preprocessing"]
    assert pre["threads"] == 10


def test_env_overrides_resource_keys(monkeypatch):
    monkeypatch.setenv("AMR_THREADS", "20")
    pre = load_config()["preprocessing"]
    assert pre["threads"] == 20


def test_bad_resource_env_raises_instead_of_silently_defaulting(monkeypatch):
    monkeypatch.setenv("AMR_THREADS", "twenty")
    with pytest.raises(ValueError, match="AMR_THREADS"):
        load_config()


def test_env_int_helper(monkeypatch):
    from lib.config import env_int
    monkeypatch.delenv("AMR_TEST_INT", raising=False)
    assert env_int("AMR_TEST_INT", 7) == 7
    monkeypatch.setenv("AMR_TEST_INT", "42")
    assert env_int("AMR_TEST_INT", 7) == 42
    monkeypatch.setenv("AMR_TEST_INT", "  ")      # blank -> default, not a crash
    assert env_int("AMR_TEST_INT", 7) == 7


# ---- registry validation ---------------------------------------------------

def _validate_registry_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vr", PROJECT_ROOT / "scripts" / "validate_registry.py")
    vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
    return vr


def test_validate_registry_passes_on_the_repository():
    vr = _validate_registry_module()
    errors, warnings = [], []
    vr._check_registry(errors, warnings)
    assert errors == [], errors


def test_validate_registry_rejects_a_path_unsafe_name(monkeypatch):
    """A slash or space in a canonical name would split or break the model's paths."""
    vr = _validate_registry_module()
    doc = dict(registry._antibiotics_doc())
    doc["classes"] = {**doc["classes"],
                      "testclass": {"display_name": "Test", "members": ["foo/bar acid"]}}
    monkeypatch.setattr(vr.registry, "_antibiotics_doc", lambda: doc)
    errors, warnings = [], []
    vr._check_registry(errors, warnings)
    assert any("foo/bar acid" in e for e in errors), errors


def test_validate_registry_rejects_a_class_without_card_terms(monkeypatch):
    """A class without CARD terms would leave its models without b and on-target genes."""
    vr = _validate_registry_module()
    doc = dict(registry._antibiotics_doc())
    doc["classes"] = {**doc["classes"],
                      "testclass": {"display_name": "Test", "members": ["foo_acid"]}}
    doc["card_drug_classes"] = {**doc["card_drug_classes"], "deadclass": ["x"]}
    monkeypatch.setattr(vr.registry, "_antibiotics_doc", lambda: doc)
    errors, warnings = [], []
    vr._check_registry(errors, warnings)
    assert any("'testclass' has no card_drug_classes" in e for e in errors), errors
    assert any("unknown class 'deadclass'" in e for e in errors), errors


def test_validate_registry_checks_taxids(monkeypatch):
    """Every organism lists its taxa; a taxon belongs to one organism only."""
    vr = _validate_registry_module()
    orgs = {k: dict(v) for k, v in registry.load_organisms().items()}
    assert registry.organism_taxids("enterobacter_cloacae") == [550, 61645, 158836, 1812935,
                                                                299767, 881260]
    orgs["ecoli"]["taxids"] = 562
    orgs["kpneumoniae"]["taxids"] = [573, 550]
    monkeypatch.setattr(vr.registry, "load_organisms", lambda: orgs)
    errors, warnings = [], []
    vr._check_registry(errors, warnings)
    assert any("'ecoli': taxids must be a list" in e for e in errors), errors
    assert any("taxon 550 is listed under" in e for e in errors), errors


# ---- tool-version provenance --------------------------------
# The KB records the tools the results depend on: unitig-caller (builds the
# features) and PopPUNK (defines the CV groups). graph_tool is tracked
# because pinning PopPUNK does NOT pin its behaviour: on 2026-07-15 a rebuild held
# poppunk at 2.7.8 while graph-tool went 2.98 -> 3.0 and E. coli re-clustered.

def test_collect_versions_captures_the_science_defining_tools():
    from lib import run_metadata
    v = run_metadata.collect_versions()
    for tool in ("unitig_caller", "bcalm", "poppunk", "graph_tool", "pyseer"):
        assert tool in v, f"{tool} missing from collect_versions"


def test_version_line_passes_over_warnings_and_refuses_errors():
    from lib.run_metadata import version_line
    assert version_line("WARNING: Python locale settings can't be changed\nQUAST v5.3.0\n") \
        == "QUAST v5.3.0"
    assert version_line("\n  1.1.0  \n") == "1.1.0"
    assert version_line("ERROR: Unknown parameter '--version'") is None
    assert version_line("WARNING: x\nusage: tool [-h]") is None
    assert version_line("") is None and version_line(None) is None
