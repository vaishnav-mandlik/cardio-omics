from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import roc_auc_score


def gene_level_protein(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    t["n_valid"] = t["n_valid_case"] + t["n_valid_control"]
    t = t.sort_values(["n_valid", "pvalue"], ascending=[False, True])
    return t.drop_duplicates("gene").set_index("gene")


def concordance(rna: pd.DataFrame, prot_gene: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, dict]:
    shared = rna.index.intersection(prot_gene.index)
    df = pd.DataFrame(
        {
            "rna_log2fc": rna.loc[shared, "log2FoldChange"],
            "rna_padj": rna.loc[shared, "padj"],
            "prot_log2fc": prot_gene.loc[shared, "log2FoldChange"],
            "prot_padj": prot_gene.loc[shared, "padj"],
        }
    ).dropna(subset=["rna_log2fc", "prot_log2fc"])
    df["rna_sig"] = rna.loc[df.index, "significant"].astype(bool) if "significant" in rna.columns else df["rna_padj"] < alpha
    df["prot_sig"] = df["prot_padj"] < alpha
    same = np.sign(df["rna_log2fc"]) == np.sign(df["prot_log2fc"])
    df["class"] = np.select(
        [df["rna_sig"] & df["prot_sig"] & same, df["rna_sig"] & df["prot_sig"] & ~same, df["rna_sig"] | df["prot_sig"]],
        ["concordant_both_sig", "discordant_both_sig", "one_layer_sig"],
        default="neither",
    )
    rho, p = sstats.spearmanr(df["rna_log2fc"], df["prot_log2fc"])
    both = df[df["rna_sig"] & df["prot_sig"]]
    summary = {
        "n_shared_genes": int(len(df)),
        "spearman_rho_all": float(rho),
        "spearman_p_all": float(p),
        "n_both_significant": int(len(both)),
        "direction_agreement_both_sig": float(same[both.index].mean()) if len(both) else float("nan"),
    }
    return df.sort_values("rna_padj"), summary


def _signature_scores(Z: pd.DataFrame, up: list[str], down: list[str]) -> pd.Series:
    parts = []
    if up:
        parts.append(Z[up].mean(axis=1))
    if down:
        parts.append(-Z[down].mean(axis=1))
    return sum(parts) / len(parts)


def signature_transfer(rna: pd.DataFrame, prot_log2: pd.DataFrame, prot_meta: pd.DataFrame, case: str, cfg, seed: int):
    cols = pd.Series(prot_log2.columns, index=prot_log2.columns).str.split(";").str[0]
    order = prot_log2.notna().mean(axis=0).sort_values(ascending=False).index
    first = cols.loc[order].drop_duplicates()
    P = prot_log2[first.index]
    P.columns = first.to_numpy()
    P = P.loc[:, P.notna().mean(axis=0) >= 0.5]
    Z = ((P - P.mean(axis=0)) / P.std(axis=0, ddof=0)).fillna(0.0)

    sig = rna[(rna["padj"] < cfg.signature_padj) & (rna["log2FoldChange"].abs() >= cfg.signature_lfc)]
    sig = sig[sig.index.isin(Z.columns)].sort_values("padj")
    up = sig[sig["log2FoldChange"] > 0].index[: cfg.signature_max_genes // 2].tolist()
    down = sig[sig["log2FoldChange"] < 0].index[: cfg.signature_max_genes // 2].tolist()
    if not up and not down:
        raise ValueError("no RNA signature genes are measured in the proteome")
    y = (prot_meta.loc[Z.index, "condition"] == case).astype(int).to_numpy()
    score = _signature_scores(Z, up, down)
    auc = roc_auc_score(y, score)

    rng = np.random.default_rng(seed)
    pool = np.array(Z.columns)
    null = np.empty(cfg.n_permutations)
    for i in range(cfg.n_permutations):
        pick = rng.choice(pool, len(up) + len(down), replace=False)
        null[i] = roc_auc_score(y, _signature_scores(Z, list(pick[: len(up)]), list(pick[len(up) :])))
    scores = pd.DataFrame({"score": score, "condition": prot_meta.loc[Z.index, "condition"]})
    return scores, {
        "n_up": len(up),
        "n_down": len(down),
        "auroc": float(auc),
        "perm_p": float((1 + np.sum(null >= auc)) / (1 + cfg.n_permutations)),
        "null_auroc_mean": float(null.mean()),
        "null_auroc_95pct": float(np.quantile(null, 0.95)),
        "up_genes": up,
        "down_genes": down,
    }


def pathway_concordance(gsea_rna: pd.DataFrame, gsea_prot: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = gsea_rna.set_index("Term")[["NES", "fdr"]].add_prefix("rna_")
    b = gsea_prot.set_index("Term")[["NES", "fdr"]].add_prefix("prot_")
    df = a.join(b, how="inner").dropna()
    if len(df) < 3:
        return df, {"n_shared_pathways": int(len(df))}
    rho, p = sstats.spearmanr(df["rna_NES"], df["prot_NES"])
    both = df[(df["rna_fdr"] < 0.05) & (df["prot_fdr"] < 0.05)]
    return df.sort_values("rna_fdr"), {
        "n_shared_pathways": int(len(df)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_both_fdr05": int(len(both)),
        "n_both_fdr05_same_direction": int((np.sign(both["rna_NES"]) == np.sign(both["prot_NES"])).sum()),
    }
