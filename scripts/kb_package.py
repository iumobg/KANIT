#!/usr/bin/env python3
"""The release package of the knowledge base.

Reads the built kanit.sqlite and writes the files of a deposit into package/: the
database itself; kanit_tables.zip, which holds one CSV per table described by a
Frictionless Data Package (datapackage.json) and the data dictionary; the data
dictionary (DATA_DICTIONARY.md); a README; the citation (CITATION.cff); and the
checksums of all of them (SHA256SUMS). The tables go into a zip because a Zenodo
record holds files, not folders.

Nothing is computed. Types, NOT NULL, keys and allowed values are read from the
database, descriptions from config/kb_dictionary.yaml, which must describe exactly
the database's tables, columns and views; title, abstract, authors and keywords
come from CITATION.cff. The licence and the DOI are those of the release record:
a build without them gives a package marked as a draft. The database must be the
file the build checked (the checksum in kb_report.json). The same database gives
the same package, byte for byte.

Output: paths_organism.kb_dir / package/ and package_report.json.
"""
import argparse
import base64
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.config import load_config, resolve_path  # noqa: E402
from lib.matrix_store import sha256_file  # noqa: E402

DICTIONARY_FILE = PROJECT_ROOT / "config" / "kb_dictionary.yaml"
CITATION_FILE = PROJECT_ROOT / "CITATION.cff"
DATABASE, TABLES_ZIP = "kanit.sqlite", "kanit_tables.zip"
FILES = (DATABASE, TABLES_ZIP, "README.md", "DATA_DICTIONARY.md", "CITATION.cff")
TYPES = {"INTEGER": "integer", "REAL": "number", "TEXT": "string", "BLOB": "string"}
LICENSES = {
    "CC-BY-4.0": ("Creative Commons Attribution 4.0 International",
                  "https://creativecommons.org/licenses/by/4.0/"),
    "CC0-1.0": ("Creative Commons Zero v1.0 Universal",
                "https://creativecommons.org/publicdomain/zero/1.0/"),
}
_IN = re.compile(r"CHECK\s*\(\s*(\w+)\s+IN\s*\(([^)]*)\)\s*\)", re.S)
_BETWEEN = re.compile(r"CHECK\s*\(\s*(\w+)\s+BETWEEN\s+(-?\d+)\s+AND\s+(-?\d+)\s*\)", re.S)
_VALUE = re.compile(r"'([^']*)'|(-?\d+)")


# ---- the schema, from the database and the dictionary -------------------------------
def _allowed(sql: str) -> tuple[dict, dict]:
    """({column: allowed values}, {column: (minimum, maximum)}) of a CREATE TABLE."""
    enums = {c: [m.group(1) if m.group(1) is not None else int(m.group(2))
                 for m in _VALUE.finditer(body)] for c, body in _IN.findall(sql)}
    ranges = {c: (int(lo), int(hi)) for c, lo, hi in _BETWEEN.findall(sql)}
    return enums, ranges


def _columns(conn: sqlite3.Connection, name: str) -> list[tuple]:
    return conn.execute(f"PRAGMA table_info({name})").fetchall()


def describe(conn: sqlite3.Connection, dictionary: dict) -> tuple[list[dict], list[dict]]:
    """(tables, views) in schema order, each column with its type, NOT NULL, key and
    allowed values from the database and its description from the dictionary."""
    objects = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE type IN "
                           "('table', 'view') ORDER BY rowid").fetchall()
    found = {kind: {n: sql for k, n, sql in objects if k == kind} for kind in ("table", "view")}
    said = {"table": dictionary.get("tables") or {}, "view": dictionary.get("views") or {}}
    problems = []
    for kind in ("table", "view"):
        problems += [f"{kind} {n} is not described" for n in found[kind] if n not in said[kind]]
        problems += [f"{n} is described but is not a {kind}" for n in said[kind]
                     if n not in found[kind]]
    pks = {}
    for name in found["table"]:
        cols = _columns(conn, name)
        pks[name] = [c[1] for c in sorted((c for c in cols if c[5]), key=lambda c: c[5])]
        if name not in said["table"]:
            continue
        d = said["table"][name]
        problems += [f"{name}: no {k}" for k in ("group", "description") if k not in d]
        described = d.get("columns") or {}
        names = [c[1] for c in cols]
        problems += [f"{name}.{c} is not described" for c in names if not described.get(c)]
        problems += [f"{name}.{c} is described but is not a column" for c in described
                     if c not in names]
    if problems:
        raise ValueError(f"{DICTIONARY_FILE.name} does not match the database: "
                         + "; ".join(problems))

    tables = []
    for name, sql in found["table"].items():
        d = said["table"][name]
        enums, ranges = _allowed(sql)
        refs = {}
        for fk in conn.execute(f"PRAGMA foreign_key_list({name})").fetchall():
            refs.setdefault(fk[0], []).append((fk[1], fk[3], fk[2], fk[4]))  # seq, from, table, to
        foreign = []
        for _, parts in sorted(refs.items()):
            parts.sort()
            target, to = parts[0][2], [p[3] for p in parts]
            foreign.append({"fields": [p[1] for p in parts], "resource": target,
                            "reference": to if all(to) else pks[target]})
        columns = [{"name": c[1], "sql_type": c[2].upper(), "not_null": bool(c[3]),
                    "enum": enums.get(c[1]), "range": ranges.get(c[1]),
                    "description": d["columns"][c[1]]} for c in _columns(conn, name)]
        tables.append({"name": name, "group": d["group"],
                       "description": d["description"], "primary_key": pks[name],
                       "foreign_keys": foreign, "columns": columns})
    views = [{"name": n, "description": said["view"][n], "sql": sql.strip(),
              "columns": [c[1] for c in _columns(conn, n)]} for n, sql in found["view"].items()]
    return tables, views


# ---- the tables as CSV ----------------------------------------------------------------
def _cell(value, column: dict, table: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, float):
        if column["sql_type"] == "INTEGER":
            raise ValueError(f"{table}.{column['name']} holds {value}, not an integer")
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "INF" if value > 0 else "-INF"
        return repr(value)
    return str(value)


def export_table(conn: sqlite3.Connection, table: dict, path: Path) -> dict:
    """One table as CSV, rows in primary key order; returns its row count and the
    columns that hold an empty value somewhere."""
    cols = table["columns"]
    order = ", ".join(table["primary_key"]) or "rowid"
    names = ", ".join(c["name"] for c in cols)
    empty: set[str] = set()
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([c["name"] for c in cols])
        for row in conn.execute(f"SELECT {names} FROM {table['name']} ORDER BY {order}"):
            cells = [_cell(v, c, table["name"]) for v, c in zip(row, cols, strict=True)]
            empty.update(c["name"] for c, x in zip(cols, cells, strict=True) if x == "")
            w.writerow(cells)
            n += 1
    return {"rows": n, "empty": empty, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


# ---- metadata ------------------------------------------------------------------------
def release_record(conn: sqlite3.Connection) -> dict:
    cur = conn.execute("SELECT * FROM release")
    rows = cur.fetchall()
    if len(rows) != 1:
        raise ValueError(f"the release table holds {len(rows)} rows, not one")
    return dict(zip([c[0] for c in cur.description], rows[0], strict=True))


def _author_name(a: dict) -> str:
    return a.get("name") or " ".join(x for x in (a.get("given-names"), a.get("family-names")) if x)


def _schema(table: dict, stats: dict) -> dict:
    fields = []
    for c in table["columns"]:
        f = {"name": c["name"], "type": TYPES.get(c["sql_type"], "string"),
             "description": c["description"]}
        if c["sql_type"] == "BLOB":
            f["format"] = "binary"
        cons = {}
        # an empty field is read as missing, so a column that holds the empty string
        # somewhere is not required, even when the database declares it NOT NULL
        if c["not_null"] and c["name"] not in stats["empty"]:
            cons["required"] = True
        if c["enum"]:
            cons["enum"] = [v for v in c["enum"] if v != ""]
        if c["range"]:
            cons["minimum"], cons["maximum"] = c["range"]
        if cons:
            f["constraints"] = cons
        fields.append(f)
    schema = {"fields": fields, "missingValues": [""]}
    if table["primary_key"]:
        schema["primaryKey"] = table["primary_key"]
    if table["foreign_keys"]:
        schema["foreignKeys"] = [{"fields": k["fields"], "reference": {
            "resource": k["resource"], "fields": k["reference"]}} for k in table["foreign_keys"]]
    return schema


def datapackage(rel: dict, cite: dict, tables: list[dict], stats: dict) -> dict:
    dp = {"profile": "tabular-data-package", "name": "kanit"}
    if rel["doi"]:
        dp["id"] = f"https://doi.org/{rel['doi']}"
    dp.update({"title": cite["title"], "description": cite.get("abstract", "").strip(),
               "version": rel["kb_version"], "created": rel["created_at"],
               "keywords": cite.get("keywords", [])})
    if rel["license"]:
        title, url = LICENSES.get(rel["license"], (rel["license"], None))
        dp["licenses"] = [{"name": rel["license"], "title": title, **({"path": url} if url else {})}]
    dp["contributors"] = [{"title": _author_name(a), "role": "author"}
                          for a in cite.get("authors", [])]
    if cite.get("repository-code"):
        dp["homepage"] = cite["repository-code"]
    dp["resources"] = [{
        "profile": "tabular-data-resource", "name": t["name"], "path": f"tables/{t['name']}.csv",
        "description": t["description"], "format": "csv", "mediatype": "text/csv",
        "encoding": "utf-8", "bytes": stats[t["name"]]["bytes"],
        "hash": f"sha256:{stats[t['name']]['sha256']}", "schema": _schema(t, stats[t["name"]])}
        for t in tables]
    return dp


def citation(rel: dict, cite: dict) -> str:
    out = {"cff-version": "1.2.0",
           "message": "If you use this knowledge base, please cite it using these metadata.",
           "type": "dataset", "title": cite["title"]}
    if cite.get("abstract"):
        out["abstract"] = cite["abstract"].strip()
    out["authors"] = cite.get("authors", [])
    out["version"] = rel["kb_version"]
    out["date-released"] = rel["created_at"][:10]
    if rel["doi"]:
        out["doi"] = rel["doi"]
    if rel["license"]:
        out["license"] = rel["license"]
    for key in ("repository-code", "keywords"):
        if cite.get(key):
            out[key] = cite[key]
    return yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=100)


# ---- the two documents ----------------------------------------------------------------
def _md(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _draft_note(rel: dict) -> str:
    missing = [x for x, k in (("licence", "license"), ("DOI", "doi")) if not rel[k]]
    if not missing:
        return ""
    return (f"> **Draft package, not a release.** The release record has no "
            f"{' and no '.join(missing)} yet; a package for deposit is built from a "
            f"knowledge base that records both.\n\n")


def data_dictionary(rel: dict, tables: list[dict], views: list[dict], stats: dict) -> str:
    out = [f"# KANIT knowledge base {rel['kb_version']}: data dictionary\n",
           f"Schema {rel['schema_version']}. Generated from the database (types, NOT NULL, keys, "
           f"allowed values) and from the project's `config/kb_dictionary.yaml` (descriptions).\n",
           _draft_note(rel),
           "## Conventions\n",
           "- The database (`kanit.sqlite`, SQLite 3) is the primary form. Each table is also a "
           "CSV file, `tables/<table>.csv` in `kanit_tables.zip`: UTF-8, comma-separated, a "
           "header row, quoted where needed (RFC 4180), rows in primary key order. "
           "`datapackage.json` describes the CSV files as a Frictionless Data Package.",
           "- An empty CSV field is NULL or, in a text column that the database declares NOT "
           "NULL, the empty string; both mean that there is no value.",
           "- Lists inside one field are semicolon-separated, except `genome.sra_accession` "
           "(comma-separated). JSON values are noted in the column's description.",
           "- Numbers are written in full precision; `INF`, `-INF` and `NaN` are written so.",
           "- `pattern.carriers` holds one bit per genome of the model, most significant bit "
           "first, bit *i* being the genome with `model_genome.row_index` = *i*; the last byte "
           "is padded with zeros. In the CSV it is base64. In Python: "
           "`numpy.unpackbits(numpy.frombuffer(blob, numpy.uint8))[:n_genomes]`.",
           "- `model_id` is `organism__antibiotic`. `unitig_id` is derived from the sequence and "
           "is the same in every model and release. `pattern_id` is valid only within its model "
           "and release; the persistent reference to a biomarker is (`model_id`, `unitig_id`).",
           "- Measurements (the evidence layers), inferences (the grades) and context (NCBI) are "
           "kept in separate tables. No table links context to a grade.",
           "- Thresholds are in the `parameter` table; with them every `passes` column and every "
           "grade can be derived again from the stored statistics.\n",
           "## Tables\n",
           "| Table | Group | Rows | Description |", "|:--|:--|--:|:--|"]
    out += [f"| [`{t['name']}`](#{t['name']}) | {t['group']} | {stats[t['name']]['rows']:,} | "
            f"{_md(t['description'])} |" for t in tables]
    out.append("")
    group = None
    for t in tables:
        if t["group"] != group:
            group = t["group"]
            out.append(f"## {group}\n")
        out.append(f"### {t['name']}\n")
        out.append(f"{t['description']} Rows: {stats[t['name']]['rows']:,}.\n")
        refs = {f: f"→ `{k['resource']}`" for k in t["foreign_keys"] for f in k["fields"]}
        out += ["| Column | Type | NOT NULL | Key | Allowed values | Description |",
                "|:--|:--|:-:|:--|:--|:--|"]
        for c in t["columns"]:
            key = " ".join(x for x in ("PK" if c["name"] in t["primary_key"] else "",
                                       refs.get(c["name"], "")) if x)
            allowed = ""
            if c["enum"]:
                allowed = ", ".join(f"`{v}`" if v != "" else "(empty)" for v in c["enum"])
            elif c["range"]:
                allowed = f"{c['range'][0]}–{c['range'][1]}"
            kind = TYPES.get(c["sql_type"], "string") + (" (base64)" if c["sql_type"] == "BLOB"
                                                         else "")
            out.append(f"| `{c['name']}` | {kind} | {'yes' if c['not_null'] else ''} | {key} | "
                       f"{allowed} | {_md(c['description'])} |")
        out.append("")
    out.append("## Views\n")
    out.append("The views are in the database only; they join the tables for the common "
               "questions.\n")
    for v in views:
        out += [f"### {v['name']}\n", f"{v['description']}\n",
                "Columns: " + ", ".join(f"`{c}`" for c in v["columns"]) + ".\n",
                "```sql", v["sql"], "```\n"]
    return "\n".join(out)


def _bare_version(tool: str, version) -> str:
    """A recorded version without the tool's own name in front of it, which most tools
    print ("poppunk 2.7.8", "unitig-caller 1.3.2", "QUAST v5.3.0"): the README already
    names the tool. The release record keeps what the tool printed."""
    text = str(version)
    for name in {tool, tool.replace("_", "-"), tool.replace("_", " ")}:
        if text.lower().startswith(name.lower() + " "):
            return text[len(name) + 1:].strip()
    return text


def readme(rel: dict, cite: dict, tables: list[dict]) -> str:
    tools = json.loads(rel["tools"] or "{}")
    authors = "; ".join(_author_name(a) for a in cite.get("authors", []))
    lic = rel["license"]
    out = [f"# {cite['title']}\n", f"Version {rel['kb_version']}, built {rel['created_at']}.\n",
           _draft_note(rel)]
    if cite.get("abstract"):
        out.append(cite["abstract"].strip() + "\n")
    out += ["A grade states association with resistance, not that the biomarker causes it.\n",
            "## Files\n", "| File | Content |", "|:--|:--|",
            f"| `{DATABASE}` | The knowledge base, a SQLite 3 database: {len(tables)} tables and "
            "the views `v_biomarker`, `v_unitig`, `v_model` and `v_novel` |",
            f"| `{TABLES_ZIP}` | Every table as CSV (`tables/`), described by `datapackage.json` "
            "(Frictionless Data Package), with the data dictionary |",
            "| `DATA_DICTIONARY.md` | Every table, column and view |",
            "| `CITATION.cff` | How to cite this release |",
            "| `SHA256SUMS` | Checksums of the files above: `sha256sum -c SHA256SUMS` |", "",
            "## Querying\n",
            "The graded candidates of the models, the strongest grade first:\n",
            "```sql", "SELECT model_id, pattern_id, grade, n_layers, layers_passed, card_genes",
            "  FROM v_biomarker", " WHERE route = 'model'",
            " ORDER BY CASE grade WHEN 'confirmed' THEN 1 WHEN 'strong_novel' THEN 2",
            "                     WHEN 'candidate' THEN 3 WHEN 'weak' THEN 4 ELSE 5 END,",
            "          model_id, pattern_id;", "```\n",
            "From Python: `pandas.read_sql_query(sql, sqlite3.connect('kanit.sqlite'))`. "
            "From R: `DBI::dbGetQuery(DBI::dbConnect(RSQLite::SQLite(), 'kanit.sqlite'), sql)`.\n",
            "## Provenance\n",
            f"- Configuration SHA-256 `{rel['config_sha256']}`",
            f"- Code commit `{rel['code_commit'] or 'unknown'}`"
            + (" (with uncommitted changes)" if rel["code_dirty"] else ""),
            f"- CARD {rel['card_version'] or 'not recorded'}",
            "- Tools: " + (", ".join(f"{k} {_bare_version(k, v)}" for k, v in sorted(tools.items()))
                           or "none recorded"),
            "- Every input file the build read, with its checksum: table `source_file`; the "
            "reference databases: `reference_database`; the data snapshots: `data_snapshot`; "
            "every threshold: `parameter`.\n",
            "## Licence and citation\n"]
    if lic:
        title, url = LICENSES.get(lic, (lic, None))
        out.append(f"Data: {title} (`{lic}`)" + (f", {url}." if url else ".")
                   + " The code that built it is released separately"
                   + (f" ({cite['repository-code']})." if cite.get("repository-code") else "."))
    else:
        out.append("Licence: not yet decided.")
    cite_line = f"{authors}. {cite['title']}. Version {rel['kb_version']}."
    out.append("\nPlease cite: " + cite_line + (f" https://doi.org/{rel['doi']}" if rel["doi"]
                                                 else "") + "\n")
    return "\n".join(out)


# ---- the package ---------------------------------------------------------------------
def _zip(path: Path, members: list[tuple[str, Path]], when: datetime) -> None:
    """A zip whose bytes depend only on its members and ``when``."""
    stamp = when.timetuple()[:6]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arcname, src in members:
            info = zipfile.ZipInfo(arcname, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            big = src.stat().st_size >= 2 ** 31
            with open(src, "rb") as f, z.open(info, "w", force_zip64=big) as dst:
                shutil.copyfileobj(f, dst, 1 << 20)


def package(db: Path, out_dir: Path, *, dictionary_file: Path = DICTIONARY_FILE,
            citation_file: Path = CITATION_FILE) -> dict:
    db, out_dir = Path(db), Path(out_dir)
    build = json.loads((db.parent / "kb_report.json").read_text())
    if build.get("sha256") != sha256_file(db):
        raise ValueError(f"{db} is not the file the build checked (kb_report.json)")
    dictionary = yaml.safe_load(dictionary_file.read_text(encoding="utf-8"))
    cite = yaml.safe_load(citation_file.read_text(encoding="utf-8"))
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rel = release_record(conn)
            if dictionary.get("schema_version") != rel["schema_version"]:
                raise ValueError(f"{dictionary_file.name} describes schema "
                                 f"{dictionary.get('schema_version')}, the database is "
                                 f"{rel['schema_version']}")
            tables, views = describe(conn, dictionary)
            (tmp / "tables").mkdir(parents=True)
            stats = {t["name"]: export_table(conn, t, tmp / "tables" / f"{t['name']}.csv")
                     for t in tables}
        finally:
            conn.close()
        (tmp / "datapackage.json").write_text(
            json.dumps(datapackage(rel, cite, tables, stats), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        (tmp / "DATA_DICTIONARY.md").write_text(data_dictionary(rel, tables, views, stats),
                                                encoding="utf-8")
        (tmp / "README.md").write_text(readme(rel, cite, tables), encoding="utf-8")
        (tmp / "CITATION.cff").write_text(citation(rel, cite), encoding="utf-8")
        members = [("datapackage.json", tmp / "datapackage.json"),
                   ("DATA_DICTIONARY.md", tmp / "DATA_DICTIONARY.md")]
        members += [(f"tables/{t['name']}.csv", tmp / "tables" / f"{t['name']}.csv")
                    for t in tables]
        _zip(tmp / TABLES_ZIP, members, datetime.fromisoformat(rel["created_at"]))
        shutil.rmtree(tmp / "tables")
        (tmp / "datapackage.json").unlink()
        try:                                   # the release database does not change
            os.link(db, tmp / DATABASE)
        except OSError:
            shutil.copy2(db, tmp / DATABASE)
        sums = {f: sha256_file(tmp / f) for f in FILES}
        if sums[DATABASE] != build["sha256"]:
            raise ValueError(f"the copy of {db} differs from it")
        (tmp / "SHA256SUMS").write_text("".join(f"{sums[f]}  {f}\n" for f in FILES))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(tmp, out_dir)
    missing = [k for k in ("license", "doi") if not rel[k]]
    report = {"kb_version": rel["kb_version"], "draft": bool(missing), "missing": missing,
              "directory": str(out_dir),
              "files": {f: {"sha256": sums[f], "bytes": (out_dir / f).stat().st_size}
                        for f in FILES},
              "tables": {t["name"]: stats[t["name"]]["rows"] for t in tables},
              "views": [v["name"] for v in views]}
    (out_dir.parent / "package_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="Write the release package of the knowledge base.")
    ap.add_argument("--db", type=Path, default=None,
                    help="SQLite file (default: paths_organism.kb_dir/kanit.sqlite)")
    args = ap.parse_args()
    db = args.db or resolve_path("kb_dir", config=config) / DATABASE
    r = package(db, db.parent / "package")
    state = f"draft, no {' and no '.join(r['missing'])}" if r["draft"] else "release"
    print(f"KANIT {r['kb_version']} package ({state}) -> {r['directory']}: "
          f"{len(r['tables'])} tables, {sum(r['tables'].values()):,} rows")


if __name__ == "__main__":
    main()
