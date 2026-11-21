#!/usr/bin/env python3
"""
Centralised config loading + path resolution.

This module is the forward-looking, {organism}-aware path layer. It is
ADDITIVE: the existing numbered scripts keep reading the legacy ``paths:`` keys
directly and continue to work unchanged. New code (orchestrator, run metadata,
the migration script, future multi-organism runs) uses resolve_path() so that
adding an organism never requires touching path-construction code.

Public API:
    load_config()                                  -> dict   (global config.yaml)
    get_target(args=None)                          -> (organism, antibiotic)
    resolve_path(key, organism=, antibiotic=, run_id=) -> Path
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

# scripts/lib/config.py  ->  parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"


# Compute-resource keys an HPC job may override from the environment. These are
# NOT science — they are how big the machine is — so they must not live only in a
# hand-edited config.yaml: `git reset --hard` wipes such edits on every deploy,
# and a forgotten re-edit means jobs silently run with laptop-sized resources
# (threads 10 instead of 40) — slow, with nothing in the logs saying why.
# Set AMR_THREADS in the SLURM script instead; the repo default stays laptop-safe.
_ENV_RESOURCE_OVERRIDES = {
    ("preprocessing", "threads"): "AMR_THREADS",   # CPU threads (03u and 02c fall back to it)
}


def env_int(name: str, default: int) -> int:
    """Read an int from environment variable ``name``, falling back to ``default``.

    Raises on a non-integer value rather than silently falling back: a typo'd
    AMR_THREADS=twenty must not quietly run the job on the default thread count.
    """
    v = os.environ.get(name)
    if v is None or not v.strip():
        return int(default)
    try:
        return int(v.strip())
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {v!r}") from e


# A YAML file whose keys are merged over config.yaml (the workflow's --config_overlay),
# e.g. the small settings and the separate paths of the end-to-end test
OVERLAY_ENV = "KANIT_CONFIG_OVERLAY"


def deep_merge(base: dict, over: dict) -> dict:
    """``over`` merged into a copy of ``base``: dictionaries key by key, the rest replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the global config.yaml as a dict.

    Compute-resource keys (see ``_ENV_RESOURCE_OVERRIDES``) are overlaid from the
    environment here, at the single point every caller goes through, so no call
    site can forget to honour them.
    """
    path = Path(config_path) if config_path else CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overlay = os.environ.get(OVERLAY_ENV, "").strip() if config_path is None else ""
    if overlay:
        with open(overlay, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})

    for (section, key), env_var in _ENV_RESOURCE_OVERRIDES.items():
        block = cfg.get(section)
        if isinstance(block, dict) and block.get(key) is not None:
            block[key] = env_int(env_var, block[key])
    return cfg


def get_target(args: Any = None, config: dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    """
    Resolve the (organism, antibiotic) target with the precedence:
        CLI args  >  environment variables  >  config.yaml defaults.

    This preserves the legacy "edit config.yaml, run the script" workflow
    (falls back to config), while enabling parameterised invocation.

    Args:
        args:   optional argparse.Namespace with .organism / .antibiotic.
        config: optional pre-loaded config dict (avoids re-reading the file).

    Returns:
        tuple(str, str): (organism, antibiotic)
    """
    cfg = config if config is not None else load_config()
    proj = cfg.get("project", {})

    organism = (
        getattr(args, "organism", None)
        or os.environ.get("AMR_ORGANISM")
        or proj.get("organism")
    )
    antibiotic = (
        getattr(args, "antibiotic", None)
        or os.environ.get("AMR_ANTIBIOTIC")
        or proj.get("target_antibiotic")
    )
    return organism, antibiotic


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean from environment variable ``name`` (``1/true/yes/on`` =>
    True, ``0/false/no/off`` => False), falling back to ``default`` when unset.

    Lets an HPC job flip a config boolean (e.g. ``AMR_EXTERNAL_MEMORY=false`` to
    train the small unitig matrix IN-CORE — much faster than spilling to scratch)
    without editing a manually-tuned config.yaml.
    """
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "on")


def resolve_path(key: str, organism: str | None = None, antibiotic: str | None = None,
                 run_id: str | None = None, config: dict[str, Any] | None = None) -> Path:
    """
    Resolve a path template from config into an absolute Path.

    Keys live in the ``paths_organism:`` block ({organism}-aware templates). Any
    ``{organism}`` / ``{antibiotic}`` / ``{run_id}`` placeholders in the template
    are filled in.

    The lookup also falls back to a legacy ``paths:`` block. That block no longer
    exists in the shipped config.yaml (removed in the M3 review) — the fallback is
    kept only so an older, hand-edited config (e.g. on the HPC) still resolves
    rather than dying. Do not add new keys there.

    Args:
        key:        path key, e.g. "matrix_dir", "genomes_dir", "run_dir".
        organism:   organism slug (required if the template uses {organism}).
        antibiotic: antibiotic id (required if the template uses {antibiotic}).
        run_id:     run identifier (required if the template uses {run_id}).
        config:     optional pre-loaded config dict.

    Returns:
        Path: PROJECT_ROOT-anchored absolute path.
    """
    cfg = config if config is not None else load_config()
    paths_org = cfg.get("paths_organism", {}) or {}
    paths_legacy = cfg.get("paths", {}) or {}

    template = paths_org.get(key, paths_legacy.get(key))
    if template is None:
        raise KeyError(
            f"Path key '{key}' not found in config 'paths_organism:' or 'paths:'."
        )

    fmt = {}
    if organism is not None:
        fmt["organism"] = organism
    if antibiotic is not None:
        fmt["antibiotic"] = antibiotic
    if run_id is not None:
        fmt["run_id"] = run_id

    # paths_prefix (an overlay's) moves every location but the reference databases
    prefix = cfg.get("paths_prefix")
    if prefix and key not in (cfg.get("paths_prefix_exclude") or []):
        template = f"{prefix.rstrip('/')}/{template}"
    try:
        resolved = template.format(**fmt) if "{" in template else template
    except KeyError as missing:
        raise KeyError(
            f"Path template for '{key}' needs placeholder {missing} "
            f"but it was not provided (organism={organism}, "
            f"antibiotic={antibiotic}, run_id={run_id})."
        ) from missing

    return PROJECT_ROOT / resolved


def resolve_tool(command_name: str, env_var: str | None = None) -> str | None:
    """
    Locate an external tool executable.

    Resolution order (first hit wins):
        1. Environment override (``env_var``, default ``AMR_<COMMAND>_BIN`` with
           hyphens as underscores, e.g. ``AMR_UNITIG_CALLER_BIN``).
        2. ``command_name`` on PATH (``shutil.which``) — the normal case, with the
           tools coming from the container, conda or a loaded module.

    Returns the resolved executable as a ``str``, or ``None`` if not found.
    """
    env_var = env_var or f"AMR_{command_name.upper().replace('-', '_')}_BIN"

    # 1) explicit environment override
    override = os.environ.get(env_var)
    if override and os.access(override, os.X_OK):
        return override

    # 2) PATH lookup (conda / system / module-loaded) — portable everywhere
    return shutil.which(command_name)
