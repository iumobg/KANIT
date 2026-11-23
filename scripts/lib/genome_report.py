"""Figures and the HTML report of a KANIT search (lib.genome_search).

Three figures:
  signature      the markers found, per antibiotic and grouped by drug class: resistant
                 direction to the right, susceptible direction to the left, split by grade
  genome map     a circular map of the assembly: contigs end to end, the markers' positions
                 (outer track resistant direction, inner track susceptible), coloured by grade
  neighbourhood  windows around clusters of resistant-direction markers: the markers with their
                 CARD gene or NCBI context and, when an annotation is given
                 (lib.genome_annotation), the genes of the window and the contig's class
"""
from __future__ import annotations

import base64
import html
import io
import math
import textwrap
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.patches import FancyArrow, Patch, Rectangle, Wedge  # noqa: E402

GRADE_COLOUR = {"confirmed": "#1c5cab", "strong_novel": "#eb6834", "candidate": "#3987e5",
                "weak": "#86b6ef", "none": "#bdbdbd"}
GRADES = ("confirmed", "strong_novel", "candidate", "weak")
GENE_COLOUR = {"amr": "#4a3aa7", "mge": "#00897b", "other": "#c4c4c4", "hypothetical": "#ececec"}
GENE_LABEL = {"amr": "resistance gene", "mge": "mobile element gene", "other": "other gene",
              "hypothetical": "hypothetical protein"}
INK, MUTED, RULE = "#222222", "#6b6b6b", "#d0d0d0"
CLUSTER_GAP = 10_000          # resistant-direction markers closer than this form one window
FLANK = 5_000                 # bases shown on either side of a window's markers
MAX_WINDOWS = 6

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "svg.fonttype": "none"})


def _label(r) -> str:
    """CARD gene, else the NCBI context gene or product, else the pattern."""
    for v in (r.get("card_genes"), r.get("context_gene"), r.get("context_product")):
        if isinstance(v, str) and v.strip():
            return v.strip()[:28]
    return f"#{r['pattern_id']}"


def _best(hits: pd.DataFrame) -> pd.DataFrame:
    """One row per occurrence (contig, start, end) and direction, with its best grade: a unitig
    can belong to a resistant-direction pattern of one model and a susceptible one of another."""
    h = hits.assign(_g=hits["grade"].map(GRADES.index)).sort_values("_g")
    return h.drop_duplicates(["contig", "start", "end", "direction"]).drop(columns="_g")


# ---- figure 1: signature -----------------------------------------------------------------------
def signature(markers: pd.DataFrame, calls: pd.DataFrame, ab_name=lambda a: a.replace("_", " ")):
    m = markers[markers["present"]]
    if m.empty:
        return None
    n = (m.groupby(["antibiotic_id", "drug_class", "direction", "grade"]).size()
          .rename("n").reset_index())
    abs_ = (n[["antibiotic_id", "drug_class"]].drop_duplicates()
             .sort_values(["drug_class", "antibiotic_id"], na_position="last"))
    call = dict(zip(calls["antibiotic_id"], calls["call"], strict=True))
    ys, y, last, heads = {}, 0.0, None, []
    for ab, cls in abs_.itertuples(index=False):
        if cls != last:
            y += 0.6 if last is not None else 0
            heads.append((y, (cls or "other").replace("_", " ")))
            y += 0.8
            last = cls
        ys[ab] = y
        y += 1
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, 1.1 + 0.28 * y)))
    for ab, yy in ys.items():
        for d, sign in (("R", 1), ("S", -1)):
            left = 0
            for g in GRADES:
                k = n[(n.antibiotic_id == ab) & (n.direction == d) & (n.grade == g)]["n"].sum()
                if k:
                    ax.barh(yy, sign * k, left=sign * left, height=0.62, color=GRADE_COLOUR[g],
                            hatch="////" if d == "S" else None, edgecolor="white", linewidth=0.6)
                    left += k
        ax.text(1.01, yy, call.get(ab, ""), transform=ax.get_yaxis_transform(), va="center",
                fontsize=8, fontweight="bold", color=INK)
    for yy, name in heads:
        ax.text(0, yy, name, transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=7.5, color=MUTED, fontstyle="italic")
    ax.set_yticks(list(ys.values()), [ab_name(a) for a in ys])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    lim = max(1, n.groupby(["antibiotic_id", "direction"])["n"].sum().max())
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.set_xlabel("markers found   ← more frequent in susceptible genomes  |  in resistant genomes →")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    hand = [Patch(color=GRADE_COLOUR[g], label=g) for g in GRADES]
    hand.append(Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="susceptible direction"))
    ax.legend(handles=hand, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
              fontsize=7.5)
    ax.text(1.01, 1.0, "call", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout()
    return fig


# ---- figure 2: genome map ----------------------------------------------------------------------
def genome_map(contigs: pd.DataFrame, hits: pd.DataFrame, max_labels: int = 10):
    c = contigs.sort_values("length", ascending=False).reset_index(drop=True)
    total = float(c["length"].sum())
    gap = total * 0.002 if len(c) > 1 else 0.0
    span = total + gap * len(c)
    offset, pos = 0.0, {}
    for r in c.itertuples():
        pos[r.contig] = offset
        offset += r.length + gap
    ang = lambda contig, x: 90 - 360 * (pos[contig] + x) / span   # noqa: E731  clockwise from top
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, r in enumerate(c.itertuples()):
        a0, a1 = ang(r.contig, r.length), ang(r.contig, 0)
        ax.add_patch(Wedge((0, 0), 1.0, a0, a1, width=0.06, color="#9a9a9a" if i % 2 else "#cfcfcf",
                           lw=0))
    b = _best(hits) if len(hits) else hits
    for d, (r0, r1) in (("R", (1.02, 1.16)), ("S", (0.80, 0.92))):
        for g in reversed(GRADES):                        # the best grades drawn last, on top
            for h in b[(b["direction"] == d) & (b["grade"] == g)].itertuples() if len(b) else []:
                t = math.radians(ang(h.contig, (h.start + h.end) / 2))
                ax.plot([r0 * math.cos(t), r1 * math.cos(t)], [r0 * math.sin(t), r1 * math.sin(t)],
                        color=GRADE_COLOUR[g], lw=1.4, solid_capstyle="butt")
    # labels: the best resistant-direction markers, one per gene, spread so that they do not
    # overlap (at least MIN_GAP degrees apart) and tied to their position by a leader line
    if len(b):
        lab = b[(b["direction"] == "R") & b["grade"].isin(GRADES[:2])].copy()
        lab["label"] = [_label(r) for r in lab.to_dict("records")]
        lab = lab.drop_duplicates("label").head(max_labels)
        lab["a"] = [ang(h.contig, (h.start + h.end) / 2) % 360 for h in lab.itertuples()]
        lab = lab.sort_values("a")
        min_gap = 9.0
        placed: list[float] = []
        for a in lab["a"]:
            placed.append(max(a, placed[-1] + min_gap) if placed else a)
        for h, pa in zip(lab.itertuples(), placed, strict=True):
            t, tp = math.radians(h.a), math.radians(pa)
            x0, y0 = 1.17 * math.cos(t), 1.17 * math.sin(t)
            x1, y1 = 1.27 * math.cos(tp), 1.27 * math.sin(tp)
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=0.5)
            ax.text(1.30 * math.cos(tp), 1.30 * math.sin(tp), h.label, fontsize=7, color=INK,
                    ha="left" if math.cos(tp) >= 0 else "right", va="center")
    ax.text(0, 0.08, f"{len(c)} contigs", ha="center", fontsize=10, color=INK)
    ax.text(0, -0.08, f"{total / 1e6:.2f} Mb", ha="center", fontsize=10, color=MUTED)
    hand: list[Artist] = [Patch(color=GRADE_COLOUR[g], label=g) for g in GRADES]
    hand += [plt.Line2D([], [], color=MUTED, lw=0, label="outer track: resistant direction"),
             plt.Line2D([], [], color=MUTED, lw=0, label="inner track: susceptible direction")]
    ax.legend(handles=hand, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False,
              fontsize=7.5)
    fig.tight_layout()
    return fig


# ---- figure 3: neighbourhoods ------------------------------------------------------------------
def windows(hits: pd.DataFrame, contigs: pd.DataFrame, gap: int = CLUSTER_GAP, flank: int = FLANK,
            limit: int = MAX_WINDOWS) -> pd.DataFrame:
    """Clusters of resistant-direction marker occurrences on one contig (closer than ``gap``),
    the best first: contig, start, end (with ``flank`` on either side, inside the contig),
    best grade, n_markers."""
    h = _best(hits) if len(hits) else hits
    h = h[h["direction"] == "R"] if len(h) else h
    if h is None or not len(h):
        return pd.DataFrame(columns=["contig", "start", "end", "grade", "n_markers"])
    length = dict(zip(contigs["contig"], contigs["length"], strict=True))
    rows: list[dict] = []
    for contig, d in h.sort_values("start").groupby("contig"):
        cur: dict | None = None
        for r in d.itertuples():
            if cur and r.start - cur["hi"] <= gap:
                cur["hi"] = max(cur["hi"], r.end)
                cur["g"] = min(cur["g"], GRADES.index(r.grade))
                cur["n"] += 1
            else:
                if cur:
                    rows.append(cur)
                cur = {"contig": contig, "lo": r.start, "hi": r.end, "g": GRADES.index(r.grade), "n": 1}
        if cur:
            rows.append(cur)
    w = pd.DataFrame(rows)
    w["start"] = (w["lo"] - flank).clip(lower=0)
    w["end"] = [min(hi + flank, length[c]) for hi, c in zip(w["hi"], w["contig"], strict=True)]
    w["grade"] = [GRADES[g] for g in w["g"]]
    w = w.sort_values(["g", "n"], ascending=[True, False]).head(limit)
    return w[["contig", "start", "end", "grade", "n"]].rename(columns={"n": "n_markers"}).reset_index(drop=True)


def neighbourhoods(hits: pd.DataFrame, contigs: pd.DataFrame, win: pd.DataFrame,
                   genes: pd.DataFrame | None = None, contig_class: dict | None = None):
    if win.empty:
        return None
    length = dict(zip(contigs["contig"], contigs["length"], strict=True))
    b = _best(hits)
    with_genes = genes is not None and len(genes)
    panels = []
    for w in win.itertuples():
        k = b[(b["contig"] == w.contig) & (b["end"] >= w.start) & (b["start"] <= w.end)]
        keys = []                                   # one number per distinct label, left to right
        for r in k.sort_values("start").to_dict("records"):
            lab = _label(r)
            ps = r.get("context_plasmid_share")
            key = lab + (f" (NCBI plasmid {ps:.0%})" if isinstance(ps, float) and not math.isnan(ps) else "")
            if key not in keys:
                keys.append(key)
        panels.append((w, k, keys))
    key_lines = [len(textwrap.wrap("   ".join(f"{i + 1} {t}" for i, t in enumerate(keys)), 125)) or 1
                 for _, _, keys in panels]
    heights = [1.25 + 0.16 * n + (0.45 if with_genes else 0) for n in key_lines]
    fig, axes = plt.subplots(len(win), 1, figsize=(7.2, sum(heights) + 0.4), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (w, k, keys) in zip(axes[:, 0], panels, strict=True):
        lo, hi = w.start, w.end
        below = with_genes or (k["direction"] == "S").any()
        ax.set_xlim(lo / 1e3, hi / 1e3)
        ax.set_ylim(-1.3 if below else -0.3, 1.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        for end in (0, length[w.contig]):
            if lo <= end <= hi:
                ax.axvline(end / 1e3, color=INK, lw=0.8, ls=(0, (2, 2)))
                ax.text(end / 1e3, 1.45, " contig end ", fontsize=6.5, color=MUTED, va="top",
                        ha="left" if end == 0 else "right")
        minw = (hi - lo) * 0.004
        for r in k.sort_values("start").to_dict("records"):
            x0, x1 = r["start"], max(r["end"], r["start"] + minw)
            up = r["direction"] == "R"
            ax.add_patch(Rectangle((x0 / 1e3, 0.25 if up else -0.75), (x1 - x0) / 1e3, 0.5,
                                   facecolor=GRADE_COLOUR[r["grade"]], lw=0,
                                   edgecolor="white", hatch=None if up else "////"))
        done = set()
        for r in k.sort_values("start").to_dict("records"):
            lab = _label(r)
            if lab in done:
                continue
            done.add(lab)
            i = next(j for j, t in enumerate(keys) if t.startswith(lab))
            up = r["direction"] == "R"
            ax.text(r["start"] / 1e3, 0.8 if up else -0.8, str(i + 1), fontsize=6.5, fontweight="bold",
                    color=INK, va="bottom" if up else "top", ha="center", clip_on=True)
        if with_genes and genes is not None:
            gk = genes[(genes["contig"] == w.contig) & (genes["end"] >= lo) & (genes["start"] <= hi)]
            for g in gk.itertuples():
                dx = (g.end - g.start) / 1e3 * (1 if g.strand == "+" else -1)
                x = (g.start if g.strand == "+" else g.end) / 1e3
                ax.add_patch(FancyArrow(x, 0, dx, 0, width=0.32, length_includes_head=True,
                                        head_width=0.42, head_length=min(abs(dx), (hi - lo) / 1e3 * 0.02),
                                        color=GENE_COLOUR.get(g.category, GENE_COLOUR["other"]), lw=0))
                if g.category in ("amr", "mge") and isinstance(g.label, str) and g.label:
                    ax.text((g.start + g.end) / 2e3, -0.3, g.label[:18], fontsize=6, color=INK,
                            ha="center", va="top", clip_on=True, rotation=20)
        cls = (contig_class or {}).get(w.contig)
        title = f"{w.contig}  ·  {length[w.contig] / 1e3:,.1f} kb" + (f"  ·  {cls}" if cls else "")
        key = "\n".join(textwrap.wrap("   ".join(f"{i + 1} {t}" for i, t in enumerate(keys)), 125))
        ax.set_title(f"{title}\n{key}", loc="left", fontsize=6.8, color=INK, pad=3)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1, 0].set_xlabel("position on the contig (kb)")
    hand = [Patch(color=GRADE_COLOUR[g], label=g) for g in GRADES]
    if genes is not None and len(genes):
        hand += [Patch(color=GENE_COLOUR[c], label=GENE_LABEL[c]) for c in ("amr", "mge", "other")]
    fig.legend(handles=hand, loc="upper center", ncol=len(hand), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---- the report ---------------------------------------------------------------------------------
def svg(fig) -> str:
    if fig is None:
        return ""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def png_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


CSS = """
:root{--ink:#222;--muted:#6b6b6b;--rule:#d0d0d0;--bg:#fff;--soft:#f5f5f5}
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg);
     max-width:1080px;margin:2em auto;padding:0 16px;line-height:1.45}
h1{margin-bottom:.1em}h2{margin-top:1.8em;border-bottom:1px solid var(--rule);padding-bottom:.2em}
.meta{color:var(--muted);font-size:.9em}.note{background:var(--soft);border-left:3px solid #eb6834;
     padding:.6em .9em;margin:1em 0;font-size:.92em}
table{border-collapse:collapse;font-size:.85em;margin:.6em 0;width:100%}
th,td{border-bottom:1px solid var(--rule);padding:.3em .45em;text-align:left;vertical-align:top}
th{background:var(--soft)}td.R{font-weight:700;color:#b3261e}td.S{color:var(--muted)}
.fig{overflow-x:auto;margin:.5em 0}.fig svg{max-width:100%;height:auto}
.small{font-size:.85em;color:var(--muted)}
"""


def _table(df: pd.DataFrame, call_col: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, r, strict=True):
            v = "" if v is None or (isinstance(v, float) and math.isnan(v)) else v
            cls = f' class="{v}"' if c == call_col and v in ("R", "S") else ""
            cells.append(f"<td{cls}>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def report_html(result, genome_name: str, organism_name: str, figures: dict[str, str],
                annotation_note: str, ab_name=lambda a: a.replace("_", " ")) -> str:
    rel = result.release
    c = result.calls.copy()
    c["antibiotic"] = c["antibiotic_id"].map(ab_name)
    c["drug class"] = c["drug_class"].fillna("").str.replace("_", " ")
    c = c.sort_values(["call", "drug class", "antibiotic"], ascending=[True, True, True])
    calls_tab = c[["antibiotic", "drug class", "call", "n_markers_rule", "n_markers_r", "n_markers_s",
                   "evidence"]].rename(columns={
                       "n_markers_rule": "markers of the rule", "n_markers_r": "R markers (all grades)",
                       "n_markers_s": "S markers (all grades)", "evidence": "markers behind the call"})
    h = result.hits.copy()
    h["antibiotic"] = h["antibiotic_id"].map(ab_name)
    h["position"] = h["contig"] + ":" + (h["start"] + 1).astype(str) + "-" + h["end"].astype(str)
    h["NCBI plasmid share"] = h["context_plasmid_share"].map(
        lambda v: "" if pd.isna(v) else f"{v:.0%}")
    hits_tab = h[["antibiotic", "pattern_id", "grade", "direction", "position", "strand", "card_genes",
                  "context_gene", "context_product", "NCBI plasmid share", "unitig_id"]].rename(columns={
                      "pattern_id": "pattern", "card_genes": "CARD gene", "context_gene": "NCBI gene",
                      "context_product": "NCBI product", "unitig_id": "unitig"})
    n_r = int((result.calls["call"] == "R").sum())
    fig_block = "".join(
        f"<h2>{html.escape(t)}</h2><p class='small'>{html.escape(cap)}</p><div class='fig'>{s}</div>"
        for t, cap, s in (
            ("Markers by antibiotic", "Markers found in the assembly, per antibiotic and grouped by drug "
             "class. Right of zero: markers more frequent in resistant training genomes; left: in "
             "susceptible ones. The letter is the call of the rule.", figures.get("signature", "")),
            ("Genome map", "Contigs end to end, longest first, clockwise from the top. Outer track: "
             "resistant-direction markers; inner track: susceptible-direction markers.",
             figures.get("map", "")),
            ("Neighbourhoods", "Windows around clusters of resistant-direction markers. " + annotation_note,
             figures.get("neighbourhood", ""))) if s)
    total = int(result.contigs["length"].sum())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>KANIT search report</title>
<style>{CSS}</style></head><body>
<h1>KANIT search report</h1>
<div class="meta">{html.escape(genome_name)} · <i>{html.escape(organism_name)}</i> ·
{len(result.contigs)} contigs, {total:,} bases · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC<br>
Knowledge base {html.escape(str(rel.get('kb_version', '')))} (CARD {html.escape(str(rel.get('card_version', '')))}
{(' · doi:' + html.escape(str(rel['doi']))) if rel.get('doi') else ''}) · rule <code>{html.escape(result.rule)}</code></div>
<div class="note"><b>Research use only.</b> A call states that the assembly carries a biomarker that the
knowledge base associates with resistance; it is not a susceptibility test result and does not
guide treatment. A grade states an association, not causation.</div>
<h2>Calls</h2><p>{n_r} of {len(result.calls)} antibiotics called resistant.</p>
{_table(calls_tab, call_col="call")}
{fig_block}
<h2>Markers found</h2>{_table(hits_tab) if len(hits_tab) else '<p>No marker found.</p>'}
<h2>How the search works</h2>
<p class="small">The member unitigs of every graded candidate pattern of the organism's models are
searched in the assembly on either strand by exact match. A pattern is present when at least half of
its member unitigs are found. A marker is a present pattern graded <code>weak</code> or better; its
direction is the side its prevalence difference favoured in the training genomes. Rule
<code>{html.escape(result.rule)}</code>: an antibiotic is called resistant when the assembly carries a
resistant-direction marker of the rule's grades. The NCBI context (gene, product, share of hits on
plasmid records) describes where the marker's longest unitig is found in NCBI nt; it is not evidence.
A short-read assembly breaks at repeats such as insertion sequences, so a neighbourhood often ends at a
contig end and the absence of a neighbouring gene does not show that it is absent from the genome.</p>
</body></html>"""
