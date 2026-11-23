#!/usr/bin/env python3
"""
Registry access — single source of truth for organisms and antibiotic classes.

Reads:
    config/registry/organisms.yaml
    config/registry/antibiotics.yaml

Public API:
    load_organisms()                        -> dict   (raw organisms block)
    load_antibiotic_classes()               -> dict   {DisplayName: [members]}
    antibiotic_to_class(ab_id)              -> class_id | None
    is_single_drug(name)                    -> bool
    normalize_antibiotic(name)              -> canonical name
    get_organism(organism_id)               -> dict

The panel (which organism–antibiotic pairs are modelled) is not part of the
registry: it follows from the panel rule (lib.panel, step 02e).
"""

from functools import lru_cache
from pathlib import Path

import yaml

# scripts/lib/registry.py  ->  parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORGANISMS_FILE = PROJECT_ROOT / "config" / "registry" / "organisms.yaml"
ANTIBIOTICS_FILE = PROJECT_ROOT / "config" / "registry" / "antibiotics.yaml"


def _read_yaml(path):
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _organisms_doc():
    return _read_yaml(ORGANISMS_FILE)


@lru_cache(maxsize=1)
def _antibiotics_doc():
    return _read_yaml(ANTIBIOTICS_FILE)


def load_amrfinder_keywords():
    """Return {antibiotic: set(UPPER keyword)} — the AMRFinderPlus Class/Subclass
    tokens that map to each antibiotic for M13 concordance (registry-driven so
    adding an antibiotic needs no code change; audit Issue 9). {} if the section
    is absent (caller falls back to its built-in defaults)."""
    raw = _antibiotics_doc().get("amrfinder_keywords", {}) or {}
    return {str(ab): {str(k).upper() for k in kws} for ab, kws in raw.items()}


def load_organisms():
    """Return the raw {organism_id: {...}} mapping from organisms.yaml."""
    return _organisms_doc().get("organisms", {})


def get_organism(organism_id):
    """Return the config block for one organism, or raise KeyError."""
    organisms = load_organisms()
    if organism_id not in organisms:
        raise KeyError(
            f"Unknown organism '{organism_id}'. "
            f"Known: {sorted(organisms.keys())}"
        )
    return organisms[organism_id]


def organism_taxids(organism_id) -> list[int]:
    """NCBI taxonomy IDs of an organism: its genomes are those at or below these taxa."""
    return [int(t) for t in get_organism(organism_id)["taxids"]]


def load_antibiotic_classes():
    """
    Return {ClassDisplayName: [members]} — the exact structure the legacy
    ANTIBIOTIC_CLASSES dictionary had, so 01/01b work unchanged.
    """
    classes = _antibiotics_doc().get("classes", {})
    out = {}
    for _cid, block in classes.items():
        display = block.get("display_name", _cid)
        out[display] = list(block.get("members", []))
    return out


@lru_cache(maxsize=1)
def _ab_to_class_index():
    """Build {antibiotic_id: class_id} reverse index (case-insensitive keys)."""
    index = {}
    for class_id, block in _antibiotics_doc().get("classes", {}).items():
        for member in block.get("members", []):
            index[str(member).lower()] = class_id
    return index


def antibiotic_to_class(ab_id):
    """Return the class_id for an antibiotic, or None if unregistered."""
    return _ab_to_class_index().get(str(ab_id).lower())


def card_drug_classes(ab_id):
    """CARD drug class terms that are class-concordant with an antibiotic (lowercase;
    empty for an unregistered antibiotic)."""
    own = (_antibiotics_doc().get("card_drug_classes_antibiotic") or {}).get(str(ab_id).lower())
    if own:                     # a single drug whose class terms cover other drugs too
        return {str(x).strip().lower() for x in own}
    cls = antibiotic_to_class(ab_id)
    terms = (_antibiotics_doc().get("card_drug_classes") or {}).get(cls, []) if cls else []
    return {str(x).strip().lower() for x in terms}


@lru_cache(maxsize=1)
def _not_single_drugs():
    return {str(x).strip().lower() for x in (_antibiotics_doc().get("not_single_drugs") or [])}


def is_single_drug(name):
    """False for a label listed under `not_single_drugs:` (a phenotype category or
    a drug-class name), True otherwise."""
    return str(name).strip().lower() not in _not_single_drugs()


@lru_cache(maxsize=1)
def _class_mechanism_index():
    """{class_id: 'acquired'|'target_snp'} from the class_mechanism_type block."""
    return dict(_antibiotics_doc().get("class_mechanism_type", {}) or {})


@lru_cache(maxsize=1)
def _who_aware_index():
    """{antibiotic: 'Access'|'Watch'|'Reserve'} from the who_aware block."""
    return {str(k): v for k, v in (_antibiotics_doc().get("who_aware", {}) or {}).items()}


def antibiotic_mechanism_type(ab_id):
    """Dominant resistance-mechanism style ('acquired' | 'target_snp') for an
    antibiotic, resolved via its class. None if antibiotic/class is unregistered."""
    cls = antibiotic_to_class(ab_id)
    return _class_mechanism_index().get(cls) if cls else None


def antibiotic_who_aware(ab_id):
    """WHO AWaRe 2023 category ('Access' | 'Watch' | 'Reserve') or None."""
    return _who_aware_index().get(str(ab_id))


def clear_cache():
    """Drop all registry lru_caches — call in tests that mutate the registry YAML
    files at runtime so a stale cached parse is not reused."""
    for fn in (_organisms_doc, _antibiotics_doc, _ab_to_class_index, _alias_index,
               _class_mechanism_index, _who_aware_index, _not_single_drugs):
        fn.cache_clear()


@lru_cache(maxsize=1)
def _alias_index():
    """
    Build {lowercased name -> canonical antibiotic name}.

    Includes every class member mapped to itself, plus every alias from the
    `aliases:` section mapped to its canonical. Used by normalize_antibiotic()
    to fold raw-source spelling variants / typos onto one canonical name.
    """
    doc = _antibiotics_doc()
    idx = {}
    # canonical members map to themselves
    for block in doc.get("classes", {}).values():
        for m in block.get("members", []):
            idx[str(m).strip().lower()] = m
    # explicit aliases (override / extend)
    for canonical, aliases in (doc.get("aliases", {}) or {}).items():
        idx[str(canonical).strip().lower()] = canonical
        for a in (aliases or []):
            idx[str(a).strip().lower()] = canonical
    return idx


def normalize_antibiotic(name):
    """
    Normalise a raw antibiotic name to its canonical registry spelling.

    Case-insensitive; trims surrounding whitespace. A name that matches a known
    member or alias returns the canonical spelling; an unknown name is returned
    trimmed but otherwise unchanged (so new drugs are never silently dropped).
    Returns None for a None/blank input.
    """
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return None
    return _alias_index().get(key.lower(), key)
