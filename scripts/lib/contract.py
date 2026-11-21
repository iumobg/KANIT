"""The output contract (config/output_contract.yaml): table schemas and the
outputs of every step, and the validation of a CSV against its schema.

A CSV is read as text (an empty cell is missing) and checked for: the columns of the
schema (and no other column unless the table has `extra`), the type of every value
(string, integer, number, boolean), required cells, `enum`, `minimum` and `maximum`,
a unique primary key and the least number of rows.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from lib.config import PROJECT_ROOT, resolve_path

CONTRACT_FILE = PROJECT_ROOT / "config" / "output_contract.yaml"
_INTEGER = re.compile(r"^-?\d+(\.0+)?$")
_BOOLEAN = {"true", "false", "1", "0", "1.0", "0.0"}


@lru_cache(maxsize=1)
def load() -> dict:
    with open(CONTRACT_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_path(location: str, file: str, config: dict, organism=None, antibiotic=None) -> Path:
    """A file of the contract: under a directory key, or a *_file key (file "" = itself)."""
    p = resolve_path(location, organism=organism, antibiotic=antibiotic, config=config)
    if location.endswith("_file"):
        p = p if not file else p.parent / file
    else:
        p = p / file
    return Path(str(p).replace("{organism}", organism or "{organism}"))


def table_path(table_id: str, config: dict, organism=None, antibiotic=None) -> Path:
    t = load()["tables"][table_id]
    return output_path(t["location"], t["file"], config, organism, antibiotic)


def _bad_value(value: str, field: dict) -> str | None:
    """Why one non-empty cell breaks its field, or None."""
    kind = field.get("type", "string")
    number: float | None = None
    if kind == "integer":
        if not _INTEGER.match(value):
            return "not an integer"
        number = float(value)
    elif kind == "number":
        try:
            number = float(value)
        except ValueError:
            return "not a number"
    elif kind == "boolean":
        if value.lower() not in _BOOLEAN:
            return "not a boolean"
    if "enum" in field:
        allowed = {str(x) for x in field["enum"]}
        if kind in ("integer", "number") and number is not None:
            if not any(math.isclose(number, float(x)) for x in field["enum"]):
                return f"not in {sorted(allowed)}"
        elif value not in allowed:
            return f"not in {sorted(allowed)}"
    if number is not None and not math.isnan(number):
        if "minimum" in field and number < field["minimum"]:
            return f"below {field['minimum']}"
        if "maximum" in field and number > field["maximum"]:
            return f"above {field['maximum']}"
    return None


def validate_frame(df: pd.DataFrame, table: dict) -> list[str]:
    """Problems of a table (read as text, '' = missing) against its schema."""
    problems = []
    fields = {f["name"]: f for f in table["fields"]}
    missing = [c for c in fields if c not in df.columns]
    if missing:
        problems.append(f"no column {missing}")
    extra = [c for c in df.columns if c not in fields]
    if extra and "extra" not in table:
        problems.append(f"undocumented column {extra}")
    checks = {**{c: fields[c] for c in fields if c in df.columns},
              **({c: table["extra"] for c in extra} if "extra" in table else {})}
    for col, field in checks.items():
        values = df[col]
        if field.get("required") and (values == "").any():
            problems.append(f"{col}: {int((values == '').sum())} empty cell(s)")
        for v in values[values != ""].unique():
            why = _bad_value(str(v), field)
            if why:
                problems.append(f"{col}: {v!r} {why}")
                break
    key = table.get("primary_key")
    if key and all(k in df.columns for k in key) and df.duplicated(key).any():
        problems.append(f"primary key {key} not unique")
    if len(df) < table.get("min_rows", 0):
        problems.append(f"{len(df)} row(s), needs {table['min_rows']}")
    return problems


def validate_csv(path: Path, table: dict) -> list[str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return validate_frame(df, table)
