"""The grades against CARD's own RGI tier.

RGI gives a hit a tier by its sequence similarity to the CARD reference: Perfect, Strict or Loose
(Loose hits were not produced). A grade measures evidence of association with the
phenotype, so no association between the two is expected. The comparison is post-hoc and
exploratory; it enters no grade or hypothesis decision.

The hit table does not say which hit a unitig lies on, so the tier is read per (genome, ARO): the
tier of that ARO's hits in a genome the unitig was located in. A pair with hits in both tiers is
ambiguous and left out.
"""
import math

import pandas as pd
from scipy.stats import chi2

PRIMARY = "allele_aware"               # the primary CARD mode
TIERS = ("Perfect", "Strict")
TIER_ORDER = ("Perfect", "Strict", "mixed")
AMBIGUOUS, NO_HIT = "ambiguous", "no_hit"
NO_CARD_HIT, B = "no_card_hit", "b"
# a grade with and without b are kept apart: with b = 1 the CARD layer is fixed, so only the
# statistical layers separate confirmed from candidate
GROUPS = ("confirmed", "candidate_b", "candidate_no_b", "weak", "none")
COMPARED = ("confirmed", "candidate_b")


def _split(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    out = []
    for x in str(value).split(";"):
        x = x.strip()
        if x.endswith(".0") and x[:-2].isdigit():      # an ARO read as a float
            x = x[:-2]
        if x:
            out.append(x)
    return out


def pair_tiers(hits: pd.DataFrame) -> pd.Series:
    """Tier of every (genome, ARO) of an organism: Perfect, Strict or ambiguous."""
    h = hits[hits["cut_off"].isin(TIERS)]
    g = h.groupby([h["genome_id"].astype(str), h["aro"].astype(str)])["cut_off"]
    return g.agg(lambda s: s.iloc[0] if s.nunique() == 1 else AMBIGUOUS).rename("tier")


def pattern_tiers(unitigs: pd.DataFrame, tiers: pd.Series, mode: str = PRIMARY) -> pd.DataFrame:
    """Per candidate pattern with a CARD hit: its (genome, ARO) pairs by tier and its own tier.

    ``unitigs`` is grades_unitigs of one model, ``tiers`` the output of pair_tiers for its
    organism. The genome IDs must have been read as text: as floats, 562.10010 becomes 562.1001."""
    u = unitigs[unitigs[f"{mode}_card_state"] != NO_CARD_HIT]
    rows = [(pid, g, a)
            for pid, gs, aros in zip(u["pattern_id"], u["located_genomes"],
                                     u[f"{mode}_card_aros"], strict=True)
            for g in _split(gs) for a in _split(aros)]
    pairs = pd.DataFrame(rows, columns=["pattern_id", "genome_id", "aro"]).drop_duplicates()
    if pairs.empty:
        return pd.DataFrame(columns=["pattern_id", "n_perfect", "n_strict", "n_ambiguous",
                                     "n_no_hit", "tier", "aros"])
    pairs["tier"] = [tiers.get((g, a), NO_HIT) for g, a in zip(pairs["genome_id"], pairs["aro"],
                                                              strict=True)]
    n = pairs.groupby(["pattern_id", "tier"]).size().unstack(fill_value=0)
    for c in (*TIERS, AMBIGUOUS, NO_HIT):
        if c not in n:
            n[c] = 0
    out = pd.DataFrame({"pattern_id": n.index,
                        "n_perfect": n["Perfect"].to_numpy(), "n_strict": n["Strict"].to_numpy(),
                        "n_ambiguous": n[AMBIGUOUS].to_numpy(), "n_no_hit": n[NO_HIT].to_numpy()})
    out["tier"] = [("mixed" if s else "Perfect") if p else ("Strict" if s else None)
                   for p, s in zip(out["n_perfect"] > 0, out["n_strict"] > 0, strict=True)]
    aros = pairs.groupby("pattern_id")["aro"].agg(lambda s: ";".join(sorted(set(s))))
    return out.merge(aros.rename("aros").reset_index(), on="pattern_id", how="left")


def group(grade: str, card_state: str) -> str | None:
    if grade == "candidate":
        return "candidate_b" if card_state == B else "candidate_no_b"
    return grade if grade in GROUPS else None


def model_table(patterns: pd.DataFrame, unitigs: pd.DataFrame, tiers: pd.Series,
                mode: str = PRIMARY) -> pd.DataFrame:
    """One row per candidate pattern with a CARD hit: grade, CARD state, group and tier."""
    p = patterns[patterns[f"{mode}_card_state"] != NO_CARD_HIT][
        ["pattern_id", f"{mode}_grade", f"{mode}_card_state"]].rename(
        columns={f"{mode}_grade": "grade", f"{mode}_card_state": "card_state"})
    p["group"] = [group(g, s) for g, s in zip(p["grade"], p["card_state"], strict=True)]
    return p.merge(pattern_tiers(unitigs, tiers, mode), on="pattern_id", how="left")


def collect(get, models) -> pd.DataFrame:
    """model_table rows of every (organism, antibiotic) in ``models``, with model and organism
    columns; empty when no organism has its hit table. ``get(table_id, organism, antibiotic,
    **read_csv)`` reads a table of the run, None when absent."""
    parts = []
    for org in dict.fromkeys(o for o, _ in models):
        hits = get("rgi_hits", org, usecols=["genome_id", "aro", "cut_off"],
                   dtype={"genome_id": str, "aro": str})
        if hits is None:
            continue
        tiers = pair_tiers(hits)
        for o, ab in models:
            if o != org:
                continue
            pats = get("grades_patterns", org, ab)
            units = get("grades_unitigs", org, ab,
                        dtype={"located_genomes": str, f"{PRIMARY}_card_aros": str})
            if pats is None or units is None:
                continue
            parts.append(model_table(pats, units, tiers).assign(model=f"{org}__{ab}",
                                                                organism=org))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mantel_haenszel(strata: list[tuple[int, int, int, int]]) -> dict:
    """Common odds ratio over 2×2 tables (a, b, c, d) = (row 1 with, row 1 without, row 2 with,
    row 2 without), its Robins–Breslow–Greenland 95% interval and the Cochran–Mantel–Haenszel
    test without continuity correction. A stratum without both rows and both columns adds
    nothing to any of the three and is not counted."""
    used = [(a, b, c, d) for a, b, c, d in strata
            if a + b > 0 and c + d > 0 and a + c > 0 and b + d > 0]
    out: dict[str, float | int | None] = {"n_strata": len(used), "odds_ratio": None,
                                          "ci_low": None, "ci_high": None, "p": None}
    if not used:
        return out
    r = [a * d / (a + b + c + d) for a, b, c, d in used]
    s = [b * c / (a + b + c + d) for a, b, c, d in used]
    R, S = sum(r), sum(s)
    diff = sum(a - (a + b) * (a + c) / (a + b + c + d) for a, b, c, d in used)
    var = sum((a + b) * (c + d) * (a + c) * (b + d) / ((a + b + c + d) ** 2 * (a + b + c + d - 1))
              for a, b, c, d in used if a + b + c + d > 1)
    if var > 0:
        out["p"] = float(chi2.sf(diff ** 2 / var, 1))
    if R > 0 and S > 0:
        out["odds_ratio"] = R / S
        pp = [(a + d) / (a + b + c + d) for a, b, c, d in used]
        qq = [(b + c) / (a + b + c + d) for a, b, c, d in used]
        v = (sum(p_ * r_ for p_, r_ in zip(pp, r, strict=True)) / (2 * R ** 2)
             + sum(p_ * s_ + q_ * r_ for p_, q_, r_, s_ in zip(pp, qq, r, s, strict=True))
             / (2 * R * S)
             + sum(q_ * s_ for q_, s_ in zip(qq, s, strict=True)) / (2 * S ** 2))
        half = 1.959963984540054 * math.sqrt(v)
        out["ci_low"] = math.exp(math.log(R / S) - half)
        out["ci_high"] = math.exp(math.log(R / S) + half)
    return out


def compare(table: pd.DataFrame) -> dict:
    """confirmed against candidate among patterns with b = 1, Perfect against Strict or mixed,
    stratified by organism. ``table`` holds model_table rows of every model with an organism
    column."""
    t = table[table["group"].isin(COMPARED) & table["tier"].notna()]
    strata = []
    for _, s in t.groupby("organism"):
        perf = s["tier"] == "Perfect"
        conf = s["group"] == "confirmed"
        strata.append((int((conf & perf).sum()), int((conf & ~perf).sum()),
                       int((~conf & perf).sum()), int((~conf & ~perf).sum())))
    out = mantel_haenszel(strata)
    out["n_confirmed"] = int((t["group"] == "confirmed").sum())
    out["n_candidate_b"] = int((t["group"] == "candidate_b").sum())
    return out


def describe(table: pd.DataFrame) -> pd.DataFrame:
    """Patterns by organism, group and tier (patterns without a tier left out)."""
    t = table[table["tier"].notna() & table["group"].notna()]
    n = t.groupby(["organism", "group", "tier"]).size().rename("n").reset_index()
    return n.pivot_table(index=["organism", "group"], columns="tier", values="n",
                         fill_value=0).reindex(columns=list(TIER_ORDER), fill_value=0).reset_index()
