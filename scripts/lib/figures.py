"""QC figures of the steps, drawn from the steps' CSV tables only.

Every function takes the tables (pandas DataFrames, as the contract describes them)
and returns a matplotlib Figure; reports.py reads the tables and saves the figures.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R_COLOUR, S_COLOUR, GREY = "#c0392b", "#2c7fb8", "#7f8c8d"
GRADES = ["confirmed", "strong_novel", "candidate", "weak", "none"]


def _fig(nrows=1, ncols=1, w=6.0, h=4.0):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    return fig, axes.ravel()


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _empty(ax, text="no data"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=GREY)
    ax.set_xticks([])
    ax.set_yticks([])


# ---- organism level -----------------------------------------------------------------
def phenotype_counts(long: pd.DataFrame, title: str):
    fig, (ax,) = _fig(w=max(6, 0.3 * long["antibiotic"].nunique()), h=4)
    t = long.groupby(["antibiotic", "label"]).size().unstack(fill_value=0).reindex(
        columns=[1, 0], fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    ax.bar(t.index, t[1], color=R_COLOUR, label="resistant")
    ax.bar(t.index, t[0], bottom=t[1], color=S_COLOUR, label="susceptible")
    ax.set_ylabel("genomes")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.tight_layout()
    return fig


def genome_qc(qc: pd.DataFrame, thresholds: dict, title: str):
    fig, axes = _fig(2, 2, 4.5, 3.2)
    spec = [("completeness", "completeness_min", "CheckM2 completeness (%)"),
            ("contamination", "contamination_max", "CheckM2 contamination (%)"),
            ("n50", "n50_min", "N50 (bp)"), ("n_contigs", "max_contigs", "contigs")]
    for ax, (col, key, label) in zip(axes, spec, strict=True):
        v = pd.to_numeric(qc[col], errors="coerce").dropna()
        if v.empty:
            _empty(ax)
            continue
        ax.hist(v, bins=40, color=GREY)
        if key in thresholds:
            ax.axvline(thresholds[key], color=R_COLOUR, ls="--")
        ax.set_xlabel(label)
    n_fail = int((~_bool(qc["pass_overall"])).sum())
    fig.suptitle(f"{title} — {len(qc)} genomes, {n_fail} excluded")
    fig.tight_layout()
    return fig


def lineage_sizes(clusters: pd.DataFrame, title: str):
    fig, (ax,) = _fig()
    sizes = clusters["Cluster"].value_counts().to_numpy()
    ax.plot(np.arange(1, sizes.size + 1), sizes, marker=".", color=GREY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lineage rank")
    ax.set_ylabel("genomes")
    ax.set_title(f"{title} — {sizes.size} lineages")
    fig.tight_layout()
    return fig


def rgi_hits(hits: pd.DataFrame, title: str):
    fig, axes = _fig(1, 2, 5, 4)
    if hits.empty:
        _empty(axes[0])
        _empty(axes[1])
        return fig
    cls = hits["drug_class"].fillna("").str.split(";").explode().str.strip()
    cls = cls[cls != ""].value_counts().head(15)
    axes[0].barh(cls.index[::-1], cls.to_numpy()[::-1], color=GREY)
    axes[0].set_xlabel("hits")
    mt = hits["model_type"].value_counts()
    axes[1].bar(mt.index, mt.to_numpy(), color=GREY)
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def external_agreement(comp: pd.DataFrame, title: str):
    ok = comp[comp["assessable"].astype(int) == 1]
    fig, (ax,) = _fig(w=max(6, 0.6 * ok["model_id"].nunique()), h=4)
    if ok.empty:
        _empty(ax)
        return fig
    t = ok.pivot_table(index="model_id", columns="tool", values="balanced_accuracy")
    t.plot.bar(ax=ax, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def context_plasmid(ctx: pd.DataFrame, title: str):
    fig, (ax,) = _fig()
    v = pd.to_numeric(ctx.get("plasmid_share", pd.Series(dtype=float)), errors="coerce").dropna()
    if v.empty:
        _empty(ax)
    else:
        ax.hist(v, bins=20, range=(0, 1), color=GREY)
        ax.set_xlabel("share of the hits on plasmid records")
        ax.set_ylabel("unitigs")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---- global ----------------------------------------------------------------------------
def panel_minority(decisions: pd.DataFrame, threshold: int, title: str):
    d = decisions.sort_values("minority", ascending=False)
    fig, (ax,) = _fig(w=max(6, 0.12 * len(d)), h=4)
    colours = np.where(d["decision"] == "included", R_COLOUR, GREY)
    ax.bar(np.arange(len(d)), d["minority"].astype(float), color=colours)
    ax.axhline(threshold, color="black", ls="--", lw=1)
    ax.set_yscale("symlog")
    ax.set_xticks([])
    ax.set_xlabel(f"{len(d)} organism–antibiotic pairs ({int((d['decision'] == 'included').sum())} "
                  "included)")
    ax.set_ylabel("minority class (genomes)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def permutation_models(lp: pd.DataFrame, title: str):
    d = lp.dropna(subset=["auc_observed"]).sort_values("auc_observed")
    fig, (ax,) = _fig(w=6, h=max(3, 0.18 * len(d)))
    if d.empty:
        _empty(ax)
        return fig
    y = np.arange(len(d))
    labels = d["organism"] + " / " + d["antibiotic"]
    ax.errorbar(d["null_mean"], y, xerr=2 * d["null_sd"].fillna(0), fmt="|", color=GREY,
                label="null mean ± 2 SD")
    ax.scatter(d["auc_observed"], y, color=R_COLOUR, s=12, label="observed", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("pooled out-of-fold ROC AUC")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def overlap_heatmap(pairs: pd.DataFrame, organism: str):
    p = pairs[pairs["organism_id"] == organism]
    models = sorted(set(p["model_a"]) | set(p["model_b"]))
    fig, (ax,) = _fig(w=max(4, 0.45 * len(models)), h=max(3.5, 0.45 * len(models)))
    if not models:
        _empty(ax)
        return fig
    m = pd.DataFrame(np.nan, index=models, columns=models)
    for r in p.itertuples():
        m.loc[r.model_a, r.model_b] = m.loc[r.model_b, r.model_a] = r.overlap
    im = ax.imshow(m.to_numpy(dtype=float), cmap="viridis", vmin=0)
    short = [x.split("__", 1)[-1] for x in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short, fontsize=7)
    fig.colorbar(im, ax=ax, label="Jaccard overlap of the candidates")
    ax.set_title(organism)
    fig.tight_layout()
    return fig


# ---- model level -----------------------------------------------------------------------
def matrix_patterns(patterns: pd.DataFrame, n_genomes: int, title: str):
    fig, axes = _fig(1, 2, 4.5, 3.5)
    axes[0].hist(patterns["n_present"] / max(n_genomes, 1), bins=50, color=GREY)
    axes[0].set_xlabel("carriage (share of genomes)")
    axes[1].hist(patterns["n_members"], bins=50, color=GREY, log=True)
    axes[1].set_xlabel("unitigs per pattern")
    fig.suptitle(f"{title} — {len(patterns):,} patterns")
    fig.tight_layout()
    return fig


def _roc(y, p):
    order = np.argsort(-p)
    y = y[order]
    tp = np.concatenate([[0], np.cumsum(y)]) / max(y.sum(), 1)
    fp = np.concatenate([[0], np.cumsum(1 - y)]) / max((1 - y).sum(), 1)
    prec = np.cumsum(y) / np.arange(1, y.size + 1)
    return fp, tp, tp[1:], prec


def cv_quality(oof: pd.DataFrame, reliability: pd.DataFrame, fold_metrics: pd.DataFrame,
               composition: pd.DataFrame, title: str):
    fig, axes = _fig(2, 2, 4.5, 3.6)
    for arm, colour in (("lineage_aware", R_COLOUR), ("lineage_blind", S_COLOUR)):
        o = oof[oof["arm"] == arm]
        if o.empty:
            continue
        g = o.groupby("genome_id").agg(y=("y", "first"), p=("p", "mean"))
        fp, tp, rec, prec = _roc(g["y"].to_numpy(int), g["p"].to_numpy(float))
        axes[0].plot(fp, tp, color=colour, label=arm)
        axes[1].plot(rec, prec, color=colour, label=arm)
        r = reliability[reliability["arm"] == arm].groupby("bin")[["mean_predicted",
                                                                   "observed_rate"]].mean()
        axes[2].plot(r["mean_predicted"], r["observed_rate"], marker="o", color=colour, label=arm)
        f = fold_metrics[fold_metrics["arm"] == arm]
        axes[3].scatter(f["fold"] + (0.1 if arm == "lineage_blind" else -0.1), f["roc_auc"],
                        color=colour, s=10, label=arm)
    axes[0].plot([0, 1], [0, 1], color=GREY, ls=":")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[2].plot([0, 1], [0, 1], color=GREY, ls=":")
    axes[2].set_xlabel("mean predicted")
    axes[2].set_ylabel("observed resistance rate")
    axes[3].set_xlabel("outer fold")
    axes[3].set_ylabel("fold ROC AUC")
    for ax in axes:
        ax.legend(fontsize=7)
    rate = composition.groupby("arm")["resistance_rate"].agg(["min", "max"])
    fig.suptitle(f"{title} — fold resistance rate "
                 + ", ".join(f"{a}: {r['min']:.2f}–{r['max']:.2f}" for a, r in rate.iterrows()))
    fig.tight_layout()
    return fig


def permutation_null(null: pd.DataFrame, observed: float | None, title: str):
    fig, (ax,) = _fig()
    ax.hist(null["auc"].astype(float), bins=40, color=GREY)
    if observed is not None and not np.isnan(observed):
        ax.axvline(observed, color=R_COLOUR, label=f"observed {observed:.3f}")
        ax.legend()
    ax.set_xlabel("ROC AUC with permuted labels")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def cpss_frequency(cpss: pd.DataFrame, threshold: float, title: str):
    fig, (ax,) = _fig()
    pi = np.sort(cpss["pi"].astype(float).to_numpy())[::-1]
    ax.plot(np.arange(1, pi.size + 1), pi, color=GREY)
    ax.axhline(threshold, color=R_COLOUR, ls="--", label=f"π = {threshold}")
    ax.set_xscale("log")
    ax.set_xlabel("prefilter pattern rank")
    ax.set_ylabel("selection frequency π")
    ax.set_title(f"{title} — {int((pi >= threshold - 1e-12).sum())} stable")
    ax.legend()
    fig.tight_layout()
    return fig


def _volcano(ax, x, q, passes, xlabel):
    y = -np.log10(np.clip(q.astype(float).to_numpy(), 1e-300, 1))
    ax.scatter(x, y, s=8, c=np.where(passes, R_COLOUR, GREY))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("−log10 q")


def prevalence_volcano(prev: pd.DataFrame, title: str):
    fig, (ax,) = _fig()
    _volcano(ax, prev["delta"].astype(float), prev["q"], _bool(prev["passes"]),
             "Δ prevalence (resistant − susceptible)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def mda_scatter(mda: pd.DataFrame, title: str):
    fig, (ax,) = _fig()
    _volcano(ax, mda["mda"].astype(float), mda["q"], _bool(mda["passes"]),
             "mean decrease in AUC")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def pyseer_qq(qq: pd.DataFrame, title: str, lambdas: dict | None = None):
    fig, (ax,) = _fig()
    for name, colour in (("tested", R_COLOUR), ("background", GREY)):
        s = qq[qq["set"] == name]
        lab = name + (f" (λ = {lambdas[name]:.2f})" if lambdas and lambdas.get(name) else "")
        ax.scatter(s["expected"], s["observed"], s=6, color=colour, label=lab)
    top = float(np.nanmax(qq[["expected", "observed"]].to_numpy(dtype=float))) if len(qq) else 1
    ax.plot([0, top], [0, top], color="black", lw=0.8)
    ax.set_xlabel("expected −log10 p")
    ax.set_ylabel("observed −log10 p")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def card_states(card: pd.DataFrame, title: str):
    fig, (ax,) = _fig()
    t = pd.DataFrame({m: card[f"{m}_state"].value_counts()
                      for m in ("allele_aware", "homolog_only")}).fillna(0)
    t.plot.bar(ax=ax)
    ax.set_ylabel("candidate patterns")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def evidence_funnel(grades: pd.DataFrame, title: str):
    fig, axes = _fig(1, 2, 4.5, 3.6)
    n = len(grades)
    fired = [n] + [int((grades["n_layers"].astype(int) >= k).sum()) for k in (1, 2, 3, 4)]
    axes[0].bar(["candidates", "≥1", "≥2", "≥3", "4"], fired, color=GREY)
    axes[0].set_ylabel("patterns")
    axes[0].set_title("layers that fire")
    t = pd.DataFrame({m: grades[f"{m}_grade"].value_counts().reindex(GRADES, fill_value=0)
                      for m in ("allele_aware", "homolog_only")})
    t.plot.bar(ax=axes[1])
    axes[1].set_title("grades")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
