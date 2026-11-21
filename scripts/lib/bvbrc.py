"""BV-BRC data of one organism: the queries, the assemblies and the binary
phenotypes (step 00a).

Genomes: the public genomes at or below the organism's taxa (``taxon_lineage_ids``),
with their NCBI identifiers. Records: their antimicrobial susceptibility records
with laboratory evidence, asked for by genome identifier (``genome_amr`` cannot
select by lineage).

A record gives a phenotype when its evidence is "Laboratory Method", it was tested
against EUCAST or CLSI breakpoints and its phenotype is Resistant (1) or Susceptible
(0). The standard must name EUCAST or CLSI and no other: every part of it, split at
",", ";", "/" and "and", begins with "eucast" or "clsi", case-insensitively ("CLSI",
"eucast", "EUCAST, CLSI", "EUCAST and CLSI", "CLSI M100"; not "CLSI, NARMS",
"British Society for Antimicrobial Chemotherapy (EUCAST)" or "veterinary CLSI").
The report counts the records of every standard and phenotype as written.
Antibiotic names are normalised through the registry; a name that is not a
registered antibiotic is kept and reported (the panel excludes labels that are not
a single drug). Identical records count once. The records of one genome and
antibiotic are resolved by majority; a tie by the records of the most recent
testing-standard year; a cell still tied is dropped.

An assembly passes when its number of sequences and its length are those of the
genome record.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

import numpy as np
import pandas as pd

from lib.registry import antibiotic_to_class, is_single_drug
from lib.registry import normalize_antibiotic as _default_normalize

API = "https://www.bv-brc.org/api"
USER_AGENT = "KANIT/1.0 (scripts/00a_download_bvbrc.py)"   # the API refuses requests without one
PAGE = 25000                  # rows per request
ID_BATCH = 1000               # genome identifiers per record query

LABORATORY = "Laboratory Method"
STANDARDS = ("eucast", "clsi")
_PARTS = re.compile(r",|;|/|\band\b")
PHENOTYPES = {"resistant": 1, "susceptible": 0}

GENOME_FIELDS = ["genome_id", "genome_name", "taxon_id", "genome_status", "assembly_accession",
                 "sra_accession", "biosample_accession", "bioproject_accession", "contigs",
                 "genome_length"]
RECORD_FIELDS = ["genome_id", "antibiotic", "resistant_phenotype", "evidence",
                 "testing_standard", "testing_standard_year", "laboratory_typing_method",
                 "measurement", "measurement_sign", "measurement_value", "measurement_unit"]

# (url, POST body or None, Accept) -> (body, headers)
Fetch = Callable[[str, bytes | None, str], tuple[bytes, dict]]


def _quote(rql: str) -> str:
    return urllib.parse.quote(rql, safe="(),&=+*/:")


def http_fetch(url: str, data: bytes | None, accept: str, *, timeout: int = 300) -> tuple[bytes, dict]:
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    if data is not None:
        headers["Content-Type"] = "application/rqlquery+x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return r.read(), dict(r.headers)


def _ssl_context():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:                       # pragma: no cover - certifi is a requirement
        return ssl.create_default_context()


class Api:
    """The BV-BRC data API; transient failures (network, 429, 5xx) are retried."""

    def __init__(self, fetch: Fetch = http_fetch, *, tries: int = 4, wait: float = 15.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.fetch, self.tries, self.wait, self.sleep = fetch, tries, wait, sleep

    def _call(self, url: str, data: bytes | None, accept: str) -> tuple[bytes, dict]:
        for attempt in range(self.tries):
            try:
                return self.fetch(url, data, accept)
            except urllib.error.HTTPError as e:
                if (e.code < 500 and e.code != 429) or attempt == self.tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.tries - 1:
                    raise
            self.sleep(self.wait * 2 ** attempt)
        raise AssertionError("unreachable")

    def version(self) -> str:
        """The version the API's root page reports."""
        body, _ = self._call(f"{API}/", None, "text/html")
        m = re.search(r"Version:\s*([^<\s]+)", body.decode("utf-8", "replace"))
        if not m:
            raise RuntimeError("the BV-BRC API root page reports no version")
        return m.group(1)

    def page(self, endpoint: str, rql: str) -> tuple[pd.DataFrame, int]:
        """One page of a query (CSV) and the number of rows of the whole result."""
        body, headers = self._call(f"{API}/{endpoint}/", _quote(rql).encode(), "text/csv")
        rng = {k.lower(): v for k, v in headers.items()}.get("content-range", "")
        m = re.search(r"/(\d+)\s*$", rng)
        if not m:
            raise RuntimeError(f"{endpoint}: no Content-Range in the response")
        text = body.decode("utf-8")
        df = (pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
              if text.strip() else pd.DataFrame())
        return df, int(m.group(1))

    def rows(self, endpoint: str, flt: str, fields: list[str], sort: str) -> pd.DataFrame:
        """Every row of a query, page by page; a result that changes meanwhile stops it."""
        parts, start, total = [], 0, None
        while total is None or start < total:
            df, n = self.page(endpoint, f"{flt}&select({','.join(fields)})&sort({sort})"
                                        f"&limit({PAGE},{start})")
            if total is not None and n != total:
                raise RuntimeError(f"{endpoint}: the result changed while it was read "
                                   f"({total} -> {n} rows); query again")
            total = n
            parts.append(df)
            start += PAGE
        out = pd.concat([p for p in parts if not p.empty] or [pd.DataFrame(columns=fields)],
                        ignore_index=True)
        out = out.reindex(columns=fields, fill_value="")
        if len(out) != total:
            raise RuntimeError(f"{endpoint}: read {len(out)} of {total} rows")
        return out

    def fasta(self, genome_id: str) -> bytes:
        body, _ = self._call(f"{API}/genome_sequence/?eq(genome_id,{genome_id})"
                             "&sort(+sequence_id)&limit(100000)", None, "application/dna+fasta")
        return body


# ---- queries ------------------------------------------------------------------------
def genome_filter(taxids: list[int]) -> str:
    return f"and(in(taxon_lineage_ids,({','.join(str(t) for t in taxids)})),eq(public,true))"


def record_filter(genome_ids: list[str]) -> str:
    return f'and(in(genome_id,({",".join(genome_ids)})),eq(evidence,"{LABORATORY}"))'


def genomes(api: Api, taxids: list[int]) -> pd.DataFrame:
    g = api.rows("genome", genome_filter(taxids), GENOME_FIELDS, "+genome_id")
    if g["genome_id"].duplicated().any():
        raise RuntimeError("genome: one identifier twice")
    return g


def records(api: Api, genome_ids: list[str], *,
            progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Laboratory records of the genomes, in batches of ID_BATCH identifiers."""
    parts = []
    for s in range(0, len(genome_ids), ID_BATCH):
        parts.append(api.rows("genome_amr", record_filter(genome_ids[s:s + ID_BATCH]),
                              RECORD_FIELDS, "+id"))
        if progress:
            progress(min(s + ID_BATCH, len(genome_ids)), len(genome_ids))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RECORD_FIELDS)
    return out.sort_values(RECORD_FIELDS, ignore_index=True)


# ---- phenotypes -----------------------------------------------------------------------
def eucast_or_clsi(standard) -> bool:
    """True when the testing standard names EUCAST or CLSI and no other standard."""
    parts = [p.strip() for p in _PARTS.split(str(standard).casefold())]
    parts = [p for p in parts if p]
    return bool(parts) and all(p.startswith(STANDARDS) for p in parts)


def resolve(labels, years) -> tuple[int | None, str]:
    """Label of one genome × antibiotic cell and how it was reached: 'single',
    'majority', 'recent_year' or 'tied' (label None)."""
    lab = np.asarray(labels, dtype=int)
    n1 = int(lab.sum())
    n0 = lab.size - n1
    if n0 == 0 or n1 == 0:
        return int(lab[0]), "single"
    if n1 != n0:
        return int(n1 > n0), "majority"
    yr = pd.to_numeric(pd.Series(list(years)), errors="coerce").to_numpy(dtype=float)
    if np.isnan(yr).all():
        return None, "tied"
    recent = lab[yr == np.nanmax(yr)]
    r1 = int(recent.sum())
    r0 = recent.size - r1
    return (None, "tied") if r1 == r0 else (int(r1 > r0), "recent_year")


def clean_amr_table(df: pd.DataFrame, normalize_fn=None) -> tuple[pd.DataFrame, dict]:
    """(genome_id, antibiotic, label) of every resolved cell and the counts of every rule."""
    normalize_fn = normalize_fn or _default_normalize
    need = ["genome_id", "antibiotic", "resistant_phenotype", "evidence", "testing_standard",
            "testing_standard_year"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"records lack column(s) {missing}")
    rep: dict = {"records": len(df)}
    df = df.drop_duplicates()
    rep["records_distinct"] = len(df)
    df = df[df["evidence"].fillna("").astype(str).str.strip() == LABORATORY]
    rep["records_laboratory"] = len(df)
    std = df["testing_standard"].fillna("").astype(str)
    used = std.map(eucast_or_clsi)
    rep["testing_standards"] = {k: {"records": int(n), "used": eucast_or_clsi(k)}
                                for k, n in std.value_counts().sort_index().items()}
    df = df[used]
    rep["records_eucast_clsi"] = len(df)
    pheno = df["resistant_phenotype"].fillna("").astype(str).str.strip().str.lower()
    keep = pheno.isin(list(PHENOTYPES))
    rep["phenotypes_dropped"] = {str(k): int(v) for k, v in pheno[~keep].value_counts().items()}
    df = df[keep].assign(label=pheno[keep].map(PHENOTYPES).astype(int))
    rep["records_resistant_susceptible"] = len(df)
    names = df["antibiotic"].map(normalize_fn)
    rep["antibiotic_names_normalised"] = {
        str(a): str(b) for a, b in sorted(set(zip(df["antibiotic"], names, strict=True)))
        if pd.notna(b) and a != b}
    df = df.assign(antibiotic=names)[names.notna() & (names.astype(str).str.len() > 0)]
    df = df.assign(genome_id=df["genome_id"].astype(str))
    names = sorted(df["antibiotic"].unique())
    rep["labels_not_single_drug"] = [a for a in names if not is_single_drug(a)]
    rep["antibiotics_not_registered"] = [a for a in names
                                         if is_single_drug(a) and antibiotic_to_class(a) is None]

    key = ["genome_id", "antibiotic"]
    agg = df.groupby(key, sort=True)["label"].agg(["sum", "size"])
    cells = (agg["sum"] > 0).astype(int)
    mixed = agg.index[(agg["sum"] > 0) & (agg["sum"] < agg["size"])]
    how = {"majority": 0, "recent_year": 0, "tied": 0}
    conflicting = df.merge(mixed.to_frame(index=False), on=key)
    for k, grp in conflicting.groupby(key, sort=True):
        label, h = resolve(grp["label"].to_numpy(), grp["testing_standard_year"].to_numpy())
        how[h] += 1
        cells[k] = -1 if label is None else label
    cells = cells[cells >= 0]
    cleaned = pd.DataFrame({"genome_id": cells.index.get_level_values(0),
                            "antibiotic": cells.index.get_level_values(1),
                            "label": cells.to_numpy(dtype=int)})
    rep["cells"] = len(agg)
    rep["cells_conflicting"] = len(mixed)
    rep["cells_resolved_by_majority"] = how["majority"]
    rep["cells_resolved_by_recent_year"] = how["recent_year"]
    rep["cells_dropped_tied"] = how["tied"]
    rep["n_genomes"] = int(cleaned["genome_id"].nunique())
    rep["n_antibiotics"] = int(cleaned["antibiotic"].nunique())
    return cleaned, rep


def pivot_binary(cleaned_long: pd.DataFrame) -> pd.DataFrame:
    """'Genome ID' and one 0/1 column per antibiotic (NaN: not tested)."""
    if cleaned_long.empty:
        return pd.DataFrame(columns=["Genome ID"])
    wide = cleaned_long.pivot_table(index="genome_id", columns="antibiotic", values="label",
                                    aggfunc="first")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide = wide.reset_index().rename(columns={"genome_id": "Genome ID"})
    wide.columns.name = None
    return wide


def drugs_without_class(wide: pd.DataFrame, min_minority: int) -> dict[str, dict[str, int]]:
    """Single drugs without a registry class whose smaller class reaches min_minority
    before quality control. They may enter the panel, and a drug that does needs its
    class in the registry, with its CARD drug-class terms, before any model is trained."""
    out = {}
    for ab in wide.columns.drop("Genome ID"):
        if not is_single_drug(ab) or antibiotic_to_class(ab) is not None:
            continue
        n_r, n_s = int((wide[ab] == 1).sum()), int((wide[ab] == 0).sum())
        if min(n_r, n_s) >= min_minority:
            out[str(ab)] = {"resistant": n_r, "susceptible": n_s}
    return out


# ---- assemblies -----------------------------------------------------------------------
def check_fasta(data: bytes, contigs: str, length: str) -> tuple[int, int, str]:
    """(sequences, length, problem) of an assembly against its genome record; problem
    is '' when both agree, or when the record gives neither."""
    if not data.lstrip().startswith(b">"):
        return 0, 0, "not FASTA"
    n = total = 0
    for line in data.splitlines():
        if line.startswith(b">"):
            n += 1
        else:
            total += len(line.strip())
    problems = []
    if str(contigs).strip() and int(float(contigs)) != n:
        problems.append(f"{n} sequences, the record has {int(float(contigs))}")
    if str(length).strip() and int(float(length)) != total:
        problems.append(f"{total} bp, the record has {int(float(length))}")
    return n, total, "; ".join(problems)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
