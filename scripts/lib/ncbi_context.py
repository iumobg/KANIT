"""Genomic context of the candidate unitigs from NCBI; not evidence.

Candidate unitigs are searched with remote blastn against nt through NCBI's
BLAST URL API, restricted to the organism (``txid<N>[Organism:exp]``), word size
11, at most 50 targets; a pattern with more than 10 members contributes its 10
longest. A search is submitted once and its XML report (the hits) and text
report (the nt release date) are both retrieved by its RID. The gene and product
at the best hit come from the GenBank record through Entrez. Every response is
cached as received and the context table is built from the cache alone, so the
knowledge base is rebuilt without querying NCBI. Requests follow NCBI's usage
rules: one at a time, at least 10 s apart (Entrez: 3 per second, 10 with a key),
a RID polled at most once a minute, with the tool name and e-mail, and a new
search only on weekends or between 21:00 and 05:00 US Eastern time on weekdays,
as NCBI asks of anyone who submits more than 50 searches; a search already
submitted is followed to its end. A request that fails on the network or with
HTTP 429 or 5xx is sent again up to ten times, 1, 2, 4 and 8 min later and then
every 15 min; any other error ends the step. Each such failure is logged with
the time, the request (submit, status, XML or text report, GenBank record) and
the RID. Requests go through curl when it is installed: on 10.11.2026 NCBI's BLAST
server closed every HTTPS request Python itself made from the TRUBA login node
without a response, while it answered curl from the same node (Entrez answered both).
"""
from __future__ import annotations

import email.message
import hashlib
import http.client
import io
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SOURCE = "ncbi_nt_remote"
TOOL = "kanit"
EASTERN = ZoneInfo("America/New_York")


def select_unitigs(members: pd.DataFrame, max_members: int) -> pd.DataFrame:
    """Distinct unitigs to search: every member of a pattern, or its ``max_members``
    longest (ties: smaller identifier). ``members``: model_id, pattern_id,
    unitig_id, sequence."""
    m = members.assign(length=members["sequence"].str.len())
    m = m.sort_values(["model_id", "pattern_id", "length", "unitig_id"],
                      ascending=[True, True, False, True])
    top = m.groupby(["model_id", "pattern_id"], sort=False).head(max_members)
    return top.drop_duplicates("unitig_id")[["unitig_id", "sequence"]].sort_values(
        "unitig_id", ignore_index=True)


TRANSIENT_HTTP = {429, 500, 502, 503, 504}


class SearchFailed(RuntimeError):
    """NCBI returned no RID for a search, or the search ended without a report."""


def transient(e: Exception) -> bool:
    """A network failure or an HTTP status that asks for the request again later."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code in TRANSIENT_HTTP
    return isinstance(e, (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                          TimeoutError))


CURL_MARK = "\n__kanit_http_status__"


def curl(url: str, data: bytes | None, *, run=subprocess.run) -> str:
    """One request through curl (POST when ``data`` is given). A network failure,
    including a connection closed without a response, raises ConnectionError and an
    HTTP status of 400 or above raises HTTPError, so ``transient`` judges them as it
    judges urllib's."""
    cmd = ["curl", "-sS", "-m", "300", "-A", TOOL, "-w", CURL_MARK + "%{http_code}", url]
    if data is not None:
        cmd[1:1] = ["--data-binary", "@-"]
    r = run(cmd, input=data, capture_output=True, timeout=330)
    out = r.stdout.decode("utf-8", errors="replace")
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise ConnectionError(f"curl exit {r.returncode}: {err}")
    body, _, code = out.rpartition(CURL_MARK)
    status = int(code) if code.strip().isdigit() else 0
    if status >= 400 or status == 0:
        raise urllib.error.HTTPError(url, status, f"HTTP {status}", email.message.Message(), None)
    return body


def until_off_hours(t: datetime) -> float:
    """Seconds from ``t`` to the hours in which NCBI takes more than 50 searches from
    one user: weekends, and 21:00 to 05:00 US Eastern time on weekdays."""
    t = t.astimezone(EASTERN)
    if t.weekday() >= 5 or t.hour >= 21 or t.hour < 5:
        return 0.0
    return (t.replace(hour=21, minute=0, second=0, microsecond=0) - t).total_seconds()


class Ncbi:
    """Sequential NCBI client; ``fetch(url, data)``, ``sleep``, ``clock`` and ``now``
    (the time of day) can be replaced."""

    def __init__(self, email: str, api_key: str = "", *, fetch=None, sleep=time.sleep,
                 clock=time.monotonic, now=None, blast_interval: float = 10.0,
                 poll_interval: float = 60.0, retries: int = 10, retry_wait: float = 60.0,
                 retry_max: float = 900.0, off_hours_only: bool = True,
                 max_search_wait: float = 6 * 3600):
        if not email:
            raise ValueError("NCBI requires an e-mail address (config ncbi.entrez_email)")
        self.email, self.api_key = email, api_key
        self.fetch = fetch or self._urlopen
        self.sleep, self.clock = sleep, clock
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.off_hours_only = off_hours_only
        self.blast_interval, self.poll_interval = blast_interval, poll_interval
        self.retries, self.retry_wait, self.retry_max = retries, retry_wait, retry_max
        # NCBI has kept a search WAITING for more than a day (E. coli, 2026-11-12) while
        # the others took minutes; past this the search counts as failed and is sent again
        self.max_search_wait = max_search_wait
        self.entrez_interval = 0.11 if api_key else 0.34
        self._last = {"blast": -1e9, "entrez": -1e9}

    @staticmethod
    def _urlopen(url: str, data: bytes | None) -> str:
        if shutil.which("curl"):
            return curl(url, data)
        req = urllib.request.Request(url, data=data, headers={"User-Agent": TOOL})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read().decode("utf-8", errors="replace")

    def _log(self, message: str) -> None:
        print(f"  {self.now().astimezone():%d.%m %H:%M:%S} {message}", file=sys.stderr,
              flush=True)

    def _wait_for_off_hours(self) -> None:
        while (wait := until_off_hours(self.now())) > 0:
            until = (self.now() + timedelta(seconds=wait)).astimezone()
            self._log(f"NCBI takes large BLAST jobs on weekends and 21:00-05:00 US Eastern "
                      f"time on weekdays; the next search waits until {until:%d.%m %H:%M} "
                      f"({wait / 3600:.1f} h)")
            self.sleep(wait)

    def _call(self, kind: str, url: str, params: dict, post: bool = False, what: str = "",
              new_search: bool = False) -> str:
        """One request; ``what`` names it in the log, and a ``new_search`` waits for
        NCBI's hours."""
        gap = (self.blast_interval if kind == "blast" else self.entrez_interval)
        body = urllib.parse.urlencode(params)
        attempt = 0
        while True:
            if new_search and self.off_hours_only:
                self._wait_for_off_hours()
            wait = self._last[kind] + gap - self.clock()
            if wait > 0:
                self.sleep(wait)
            self._last[kind] = self.clock()
            try:
                if post:
                    return self.fetch(url, body.encode())
                return self.fetch(f"{url}?{body}", None)
            except Exception as e:
                if attempt >= self.retries or not transient(e):
                    raise
                pause = min(self.retry_wait * 2 ** attempt, self.retry_max)
                attempt += 1
                self._log(f"NCBI {kind} {what}: {type(e).__name__}: {e}; attempt {attempt + 1} "
                          f"of {self.retries + 1} in {pause:.0f} s")
                self.sleep(pause)

    def blast(self, fasta: str, params: dict) -> tuple[str, str, str]:
        """(rid, XML report, text report) of one search."""
        put = self._call("blast", BLAST_URL, {"CMD": "Put", "PROGRAM": "blastn", "DATABASE": "nt",
                                              "QUERY": fasta, **params, "TOOL": TOOL,
                                              "EMAIL": self.email}, post=True,
                         what="submit", new_search=True)
        found = re.search(r"RID = (\S+)", put)
        rtoe = re.search(r"RTOE = (\d+)", put)
        if not found:
            said = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", put)).strip()[:200]
            raise SearchFailed(f"BLAST did not return a request id: {said}")
        rid = found.group(1)
        wait = max(int(rtoe.group(1)) if rtoe else 30, 10)
        self._log(f"BLAST {rid} submitted; NCBI expects it in {wait} s")
        submitted = self.clock()
        self.sleep(wait)
        while True:
            info = self._call("blast", BLAST_URL, {"CMD": "Get", "FORMAT_OBJECT": "SearchInfo",
                                                   "RID": rid}, what=f"status {rid}")
            status = re.search(r"Status=(\w+)", info)
            state = status.group(1) if status else "UNKNOWN"
            if state == "READY":
                break
            if state in ("FAILED", "UNKNOWN"):
                raise SearchFailed(f"BLAST search {rid} ended with status {state}")
            if self.clock() - submitted >= self.max_search_wait:
                raise SearchFailed(f"BLAST search {rid} still {state} after "
                                   f"{(self.clock() - submitted) / 3600:.1f} h")
            self.sleep(self.poll_interval)
        get = {"CMD": "Get", "RID": rid}
        xml = self._call("blast", BLAST_URL, {**get, "FORMAT_TYPE": "XML"},
                         what=f"XML report {rid}")
        text = self._call("blast", BLAST_URL, {**get, "FORMAT_TYPE": "Text", "ALIGNMENTS": "0"},
                          what=f"text report {rid}")
        return rid, xml, text

    def genbank(self, accession: str, start: int, stop: int) -> str:
        params = {"db": "nuccore", "id": accession, "rettype": "gb", "retmode": "text",
                  "seq_start": start, "seq_stop": stop, "tool": TOOL, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return self._call("entrez", EFETCH_URL, params, what=f"GenBank {accession}")


# ---- cache ---------------------------------------------------------------------------
def batch_key(unitig_ids, params: dict) -> str:
    h = hashlib.sha256(json.dumps([sorted(unitig_ids), sorted(params.items())]).encode())
    return h.hexdigest()[:16]


def query_batch(client: Ncbi, cache: Path, batch: pd.DataFrame, params: dict) -> Path:
    """Search one batch unless its cache entry is complete; returns the entry."""
    d = Path(cache) / "blast" / batch_key(batch["unitig_id"], params)
    if (d / "meta.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    fasta = "".join(f">{u}\n{s}\n" for u, s in batch[["unitig_id", "sequence"]].itertuples(
        index=False))
    (d / "query.fasta").write_text(fasta)
    submitted = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rid, xml, text = client.blast(fasta, params)
    (d / "report.xml").write_text(xml)
    (d / "report.txt").write_text(text)
    (d / "meta.json").write_text(json.dumps({"rid": rid, "submitted_at": submitted,
                                             "params": params,
                                             "unitig_ids": batch["unitig_id"].tolist()}, indent=2))
    return d


def genbank_file(cache: Path, accession: str, start: int, stop: int) -> Path:
    return Path(cache) / "entrez" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', accession)}_{start}_{stop}.gb"


# ---- parsing ---------------------------------------------------------------------------
def _num(el, tag, cast=float):
    v = el.findtext(tag)
    return cast(v) if v not in (None, "") else None


def parse_xml(xml: str) -> pd.DataFrame:
    """One row per hit and query of a BLAST XML report (best HSP of each hit)."""
    root = ET.fromstring(xml)
    rows = []
    for it in root.iter("Iteration"):
        query = (it.findtext("Iteration_query-def") or "").split()[0]
        qlen = _num(it, "Iteration_query-len", int)
        for rank, hit in enumerate(it.iter("Hit"), 1):
            hsp = hit.find("Hit_hsps/Hsp")
            if hsp is None:
                continue
            align = _num(hsp, "Hsp_align-len", int) or 0
            qf, qt = _num(hsp, "Hsp_query-from", int), _num(hsp, "Hsp_query-to", int)
            hf, ht = _num(hsp, "Hsp_hit-from", int), _num(hsp, "Hsp_hit-to", int)
            rows.append({
                "unitig_id": query, "rank": rank, "accession": hit.findtext("Hit_accession"),
                "title": hit.findtext("Hit_def") or "",
                "identity": 100.0 * (_num(hsp, "Hsp_identity", int) or 0) / align if align else None,
                "coverage": (abs(qt - qf) + 1) / qlen if qlen and qf and qt else None,
                "evalue": _num(hsp, "Hsp_evalue"), "bitscore": _num(hsp, "Hsp_bit-score"),
                "hit_start": min(hf, ht) if hf and ht else None,
                "hit_stop": max(hf, ht) if hf and ht else None})
    return pd.DataFrame(rows, columns=["unitig_id", "rank", "accession", "title", "identity",
                                       "coverage", "evalue", "bitscore", "hit_start", "hit_stop"])


def nt_release(text: str) -> str | None:
    """The 'Posted date' of the database in a BLAST text report."""
    m = re.search(r"Posted date:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


def gene_and_product(genbank: str) -> tuple[str | None, str | None]:
    """Gene and product of the first CDS (else gene) feature of a GenBank record."""
    from Bio import SeqIO
    try:
        rec = SeqIO.read(io.StringIO(genbank), "genbank")
    except ValueError:
        return None, None
    for kind in ("CDS", "gene"):
        for f in rec.features:
            if f.type == kind:
                q = f.qualifiers
                return (q.get("gene", [None])[0] or q.get("locus_tag", [None])[0],
                        q.get("product", [None])[0])
    return None, None


def summarize(hits: pd.DataFrame, features: dict) -> pd.DataFrame:
    """One row per unitig: number of hits, best hit, its gene and product, and the share
    of hits on plasmid records. ``features``: {(accession, start, stop): (gene, product)}."""
    rows = []
    for uid, h in hits.groupby("unitig_id", sort=True):
        h = h.sort_values("rank")
        best = h.iloc[0]
        gene, product = features.get((best["accession"], best["hit_start"], best["hit_stop"]),
                                     (None, None))
        rows.append({"unitig_id": uid, "n_hits": len(h), "best_accession": best["accession"],
                     "best_title": best["title"], "best_identity": best["identity"],
                     "best_coverage": best["coverage"], "best_evalue": best["evalue"],
                     "gene": gene, "product": product,
                     "plasmid_share": float(h["title"].str.contains("plasmid", case=False).mean())})
    return pd.DataFrame(rows)


def build_table(cache: Path, organism_id: str) -> pd.DataFrame:
    """The context table of every searched unitig, from the cache only."""
    parts = []
    for d in sorted((Path(cache) / "blast").glob("*/meta.json")):
        meta = json.loads(d.read_text())
        hits = parse_xml((d.parent / "report.xml").read_text())
        release = nt_release((d.parent / "report.txt").read_text())
        features = {}
        for acc, s, e in hits.loc[hits["rank"] == 1, ["accession", "hit_start", "hit_stop"]
                                  ].itertuples(index=False):
            f = genbank_file(cache, acc, int(s), int(e))
            if f.exists():
                features[(acc, s, e)] = gene_and_product(f.read_text())
        t = summarize(hits, features)
        searched = pd.DataFrame({"unitig_id": meta["unitig_ids"]})
        t = searched.merge(t, on="unitig_id", how="left")
        t["n_hits"] = t["n_hits"].fillna(0).astype(int)
        parts.append(t.assign(organism_id=organism_id, source=SOURCE,
                              queried_on=meta["submitted_at"], nt_release=release))
    cols = ["unitig_id", "organism_id", "source", "queried_on", "nt_release", "n_hits",
            "best_accession", "best_title", "best_identity", "best_coverage", "best_evalue",
            "gene", "product", "plasmid_share"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).drop_duplicates("unitig_id")[cols]
