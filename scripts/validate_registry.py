#!/usr/bin/env python3
"""
Registry consistency guard.

Checks the registry files (config/registry/{organisms,antibiotics}.yaml) for the
invariants the pipeline relies on, and — with --db — that the KB agrees with the
registry (the registry is the single source of truth; the KB is its derivative).

    python scripts/validate_registry.py
    python scripts/validate_registry.py --db results/kb/kanit.db

Exit code 0 = all checks pass, 1 = one or more violations (suitable for CI).
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import registry  # noqa: E402

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_AWARE_VALUES = {"Access", "Watch", "Reserve"}
_MECH_VALUES = {"acquired", "target_snp"}


def _check_registry(errors: list[str], warnings: list[str]) -> None:
    doc = registry._antibiotics_doc()
    classes = doc.get("classes", {})

    # 1) every member belongs to exactly one class, under a path-safe name
    #    (canonical names become directory names; other spellings are aliases)
    seen: dict[str, str] = {}
    for cid, block in classes.items():
        for m in block.get("members", []):
            key = str(m).lower()
            if key in seen:
                errors.append(f"antibiotic '{m}' is in two classes: {seen[key]} and {cid}")
            seen[key] = cid
            if not _SLUG_RE.match(str(m)):
                errors.append(f"antibiotic '{m}' is not a path-safe name ([a-z0-9_])")

    # 1a) every class has its CARD drug class terms, and only known classes do
    card_map = doc.get("card_drug_classes", {}) or {}
    for cid in classes:
        if not card_map.get(cid):
            errors.append(f"class '{cid}' has no card_drug_classes entry")
    for cid in card_map:
        if cid not in classes:
            errors.append(f"card_drug_classes lists unknown class '{cid}'")
    # a single drug's own terms narrow its class's terms and never add to them
    for ab, terms in (doc.get("card_drug_classes_antibiotic", {}) or {}).items():
        cid = seen.get(str(ab).lower())
        if cid is None:
            errors.append(f"card_drug_classes_antibiotic lists unknown antibiotic '{ab}'")
            continue
        extra = {str(t).lower() for t in terms or []} - {str(t).lower() for t in card_map.get(cid, [])}
        if not terms or extra:
            errors.append(f"card_drug_classes_antibiotic for '{ab}' is empty or not within class '{cid}'")

    # 1b) a label that is not a single drug is never also a class member
    for label in (doc.get("not_single_drugs", []) or []):
        if str(label).lower() in seen:
            errors.append(f"not_single_drugs lists '{label}', which is a class member")

    # 2) every alias canonical is a member
    for canonical in (doc.get("aliases", {}) or {}):
        if str(canonical).lower() not in seen:
            errors.append(f"alias canonical '{canonical}' is not a member of any class")

    # 3) class_mechanism_type keys valid, values in {acquired,target_snp}
    for cid, val in (doc.get("class_mechanism_type", {}) or {}).items():
        if cid not in classes:
            errors.append(f"class_mechanism_type references unknown class '{cid}'")
        if val not in _MECH_VALUES:
            errors.append(f"class_mechanism_type['{cid}'] = '{val}' not in {sorted(_MECH_VALUES)}")

    # 4) who_aware antibiotics are members; values valid
    for ab, val in (doc.get("who_aware", {}) or {}).items():
        if str(ab).lower() not in seen:
            warnings.append(f"who_aware lists '{ab}' which is not a class member")
        if val not in _AWARE_VALUES:
            errors.append(f"who_aware['{ab}'] = '{val}' not in {sorted(_AWARE_VALUES)}")

    # 5) amrfinder_keywords antibiotics are members
    for ab in registry.load_amrfinder_keywords():
        if str(ab).lower() not in seen:
            warnings.append(f"amrfinder_keywords lists '{ab}' which is not a class member")

    # 6) organisms: lowercase slugs and the fields the pipeline reads; taxids a list
    #    of positive integers, no taxon in two organisms
    owner: dict[int, str] = {}
    for slug, block in registry.load_organisms().items():
        if not _SLUG_RE.match(slug):
            errors.append(f"organism slug '{slug}' is not lowercase snake_case")
        for field in ("display_name", "taxids", "gram_stain", "phylum"):
            if block.get(field) in (None, "", []):
                errors.append(f"organism '{slug}' has no {field}")
        taxids = block.get("taxids") or []
        if not isinstance(taxids, list) or not all(
                isinstance(t, int) and not isinstance(t, bool) and t > 0 for t in taxids):
            errors.append(f"organism '{slug}': taxids must be a list of positive integers")
            continue
        for t in taxids:
            if t in owner:
                errors.append(f"taxon {t} is listed under '{owner[t]}' and '{slug}'")
            owner[t] = slug
        names = block.get("species_names") or []
        if not isinstance(names, list) or len(names) != len(taxids) or not all(
                isinstance(x, str) and x.strip() for x in names):
            errors.append(f"organism '{slug}': species_names must name every taxid, in order")


def _check_kb(db: Path, errors: list[str], warnings: list[str]) -> None:
    if not db.exists():
        warnings.append(f"KB not found at {db} — skipped KB↔registry checks")
        return
    conn = sqlite3.connect(str(db))
    known_orgs = set(registry.load_organisms())

    for (org,) in conn.execute("SELECT DISTINCT organism FROM pipeline_runs"):
        if org not in known_orgs:
            errors.append(f"KB model organism '{org}' is not in the registry")

    for ab, dc, mech, aware in conn.execute(
            "SELECT antibiotic, drug_class, mechanism_type, who_aware FROM antibiotics"):
        exp_dc = registry.antibiotic_to_class(ab)
        if dc != exp_dc:
            errors.append(f"KB drug_class['{ab}'] = {dc!r} but registry says {exp_dc!r}")
        exp_mech = registry.antibiotic_mechanism_type(ab)
        if mech != exp_mech:
            errors.append(f"KB mechanism_type['{ab}'] = {mech!r} but registry says {exp_mech!r}")
        exp_aware = registry.antibiotic_who_aware(ab)
        if aware != exp_aware:
            errors.append(f"KB who_aware['{ab}'] = {aware!r} but registry says {exp_aware!r}")
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the AMR registry (and optionally the KB).")
    ap.add_argument("--db", default=None, help="KB SQLite path for registry↔KB consistency checks")
    args = ap.parse_args()

    registry.clear_cache()
    errors: list[str] = []
    warnings: list[str] = []
    _check_registry(errors, warnings)
    if args.db:
        _check_kb(Path(args.db), errors, warnings)

    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  FAIL  {e}")
    print(f"\nregistry validation: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
