#!/usr/bin/env python3
"""
Shared pytest fixtures & helpers for the AMR pipeline test suite.

Goal: let you validate the pipeline in SECONDS/MINUTES instead of re-running the
full multi-day job. Nothing here modifies the 9 numbered scripts — the suite is
purely additive.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Make `lib`, `utils`, `constants`, and the numbered scripts importable exactly
# as they are when launched via `python scripts/0X_*.py`.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope="session")
def repo_root():
    return PROJECT_ROOT


@pytest.fixture
def load_script():
    """
    Return a loader that imports a numbered script module by filename.

    The numbered scripts can't be imported with a normal `import` (names start
    with a digit), so we load them via importlib from their path. If the module
    needs a heavy dependency that isn't installed in this environment
    (e.g. xgboost, optuna), the loader raises pytest.skip with a clear reason —
    so the same test file runs fully in your real env and skips gracefully here.

    Usage:
        def test_x(load_script):
            mod = load_script("02e_panel.py")
    """
    def _load(filename):
        path = SCRIPTS_DIR / filename
        if not path.exists():
            pytest.fail(f"Script not found: {path}")
        mod_name = "amrtest_" + path.stem.replace(".", "_")
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            pytest.skip(f"{filename} needs an unavailable dependency: {e}")
        except SystemExit as e:
            pytest.skip(f"{filename} called sys.exit at import (config/data missing): {e}")
        return module

    return _load
