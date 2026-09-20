from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PALETTE = {"NF": "#4C72B0", "DCM": "#C44E52", "HCM": "#DD8452", "Donor": "#4C72B0", "ICM": "#C44E52"}
UP, DOWN, NS = "#C44E52", "#4C72B0", "#BBBBBB"

plt.rcParams.update(
    {"figure.dpi": 120, "savefig.dpi": 200, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pca(pca_df: pd.DataFrame, var_exp: list[float], path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for grp, sub in pca_df.groupby("etiology", observed=True):
        ax.scatter(sub["PC1"], sub["PC2"], s=12, alpha=0.75, label=f"{grp} (n={len(sub)})", color=PALETTE.get(str(grp)))
    ax.set_xlabel(f"PC1 ({var_exp[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.0%})")
    ax.set_title("LV transcriptome, top-variance genes")
    ax.legend(frameon=False)
    _save(fig, path)


def library_sizes(qc: pd.DataFrame, meta: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    groups = meta.loc[qc.index, "etiology"].astype(str)
    order = sorted(groups.unique())
    ax.boxplot([qc.loc[groups == g, "library_size"] / 1e6 for g in order], tick_labels=order)
    ax.set_ylabel("Library size (million reads)")
    ax.set_title("Sequencing depth by group")
    _save(fig, path)


def volcano(res: pd.DataFrame, title: str, path, alpha=0.05, lfc=0.585, label_col=None, n_labels=12) -> None:
    df = res.dropna(subset=["log2FoldChange", "padj"]).copy()
    df["nlp"] = -np.log10(df["padj"].clip(lower=1e-300))
    sig = (df["padj"] < alpha) & (df["log2FoldChange"].abs() >= lfc)
    colors = np.where(sig & (df["log2FoldChange"] > 0), UP, np.where(sig, DOWN, NS))
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(df["log2FoldChange"], df["nlp"], c=colors, s=5, alpha=0.7, linewidths=0)
    ax.axhline(-np.log10(alpha), color="k", lw=0.6, ls="--")
    for x in (-lfc, lfc):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    for idx, row in df[sig].nsmallest(n_labels, "padj").iterrows():
        name = row[label_col] if label_col else idx
        ax.annotate(str(name), (row["log2FoldChange"], row["nlp"]), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
    n_up, n_down = int((sig & (df["log2FoldChange"] > 0)).sum()), int((sig & (df["log2FoldChange"] < 0)).sum())
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 adjusted p")
    ax.set_title(f"{title}\n{n_up} up, {n_down} down")
    _save(fig, path)


def missingness(miss: pd.DataFrame, rho: float, path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(miss["mean_log2"], miss["missing_frac"], s=4, alpha=0.4, color="#555555", linewidths=0)
    bins = pd.qcut(miss["mean_log2"], 20, duplicates="drop")
    trend = miss.groupby(bins, observed=True).agg(x=("mean_log2", "mean"), y=("missing_frac", "mean"))
    ax.plot(trend["x"], trend["y"], color=UP, lw=2, label="binned mean")
    ax.set_xlabel("Mean observed log2 intensity")
    ax.set_ylabel("Fraction missing")
    ax.set_title(f"Missingness vs abundance (Spearman rho = {rho:.2f})")
    ax.legend(frameon=False)
    _save(fig, path)


def concordance(df: pd.DataFrame, rho: float, path) -> None:
    colors = {"concordant_both_sig": UP, "discordant_both_sig": "#8172B2", "one_layer_sig": "#999999", "neither": "#DDDDDD"}
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for cls in ["neither", "one_layer_sig", "discordant_both_sig", "concordant_both_sig"]:
        sub = df[df["class"] == cls]
        size = 6 if cls in ("neither", "one_layer_sig") else 16
        ax.scatter(sub["rna_log2fc"], sub["prot_log2fc"], s=size, color=colors[cls], alpha=0.8, linewidths=0,
                   label=f"{cls.replace('_', ' ')} ({len(sub)})")
    for idx, row in df[df["class"] == "concordant_both_sig"].iterrows():
        ax.annotate(str(idx), (row["rna_log2fc"], row["prot_log2fc"]), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("RNA log2FC (DCM vs NF, MAGNet)")
    ax.set_ylabel("Protein log2FC (ICM vs donor, PXD042155)")
    ax.set_title(f"Cross-omics fold-change concordance (rho = {rho:.2f})")
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    _save(fig, path)


def signature_scores(scores: pd.DataFrame, stats: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    order = sorted(scores["condition"].unique())
    for i, grp in enumerate(order):
        vals = scores.loc[scores["condition"] == grp, "score"]
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(vals))
        ax.boxplot([vals], positions=[i], widths=0.5, showfliers=False)
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, color=PALETTE.get(grp, "#555"), zorder=3)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("RNA-derived HF signature score (protein z)")
    ax.set_title(f"Signature transfer to proteome\nAUROC {stats['auroc']:.2f}, perm p = {stats['perm_p']:.3g}")
    _save(fig, path)


def nes_scatter(df: pd.DataFrame, stats: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    both = (df["rna_fdr"] < 0.05) & (df["prot_fdr"] < 0.05)
    ax.scatter(df.loc[~both, "rna_NES"], df.loc[~both, "prot_NES"], s=5, color=NS, linewidths=0)
    ax.scatter(df.loc[both, "rna_NES"], df.loc[both, "prot_NES"], s=12, color=UP, linewidths=0,
               label=f"FDR<0.05 in both ({int(both.sum())})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Reactome NES, transcriptome")
    ax.set_ylabel("Reactome NES, proteome")
    ax.set_title(f"Pathway concordance (rho = {stats.get('spearman_rho', float('nan')):.2f})")
    ax.legend(frameon=False)
    _save(fig, path)


def gsea_bar(gsea: pd.DataFrame, title: str, path, n: int = 10) -> None:
    sig = gsea[gsea["fdr"] < 0.05]
    top = pd.concat([sig.nlargest(n, "NES"), sig.nsmallest(n, "NES")]).drop_duplicates("Term").sort_values("NES")
    fig, ax = plt.subplots(figsize=(6.2, max(2.5, 0.22 * len(top) + 0.8)))
    labels = [t if len(t) < 60 else t[:57] + "..." for t in top["Term"]]
    ax.barh(labels, top["NES"], color=np.where(top["NES"] > 0, UP, DOWN))
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Normalised enrichment score")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=6.5)
    _save(fig, path)


def roc(roc_df: pd.DataFrame, metrics: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    for model, sub in roc_df.groupby("model"):
        auc = metrics["auroc"] if model == "elastic-net" else metrics.get("baseline_auroc", np.nan)
        ax.plot(sub["fpr"], sub["tpr"], lw=1.6, label=f"{model} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="k", lw=0.5, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC (nested CV)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def network(G: nx.Graph, node_fc: dict, path, max_nodes: int = 80) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no STRING edges at this confidence", ha="center")
        ax.axis("off")
        _save(fig, path)
        return
    nodes = sorted(G.degree, key=lambda x: -x[1])[:max_nodes]
    H = G.subgraph([n for n, _ in nodes])
    pos = nx.spring_layout(H, seed=7, k=0.35)
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[UP if node_fc.get(n, 0) > 0 else DOWN for n in H.nodes],
                           node_size=[30 + 18 * H.degree(n) for n in H.nodes], alpha=0.85)
    shown = list(H.nodes) if H.number_of_nodes() <= 60 else [n for n, _ in nodes[:20]]
    nx.draw_networkx_labels(H, pos, labels={n: n for n in shown}, font_size=6.5, ax=ax)
    ax.set_title("STRING (score >= 0.7) network of top DE genes\nred = up in DCM, blue = down; size = degree")
    ax.axis("off")
    _save(fig, path)
