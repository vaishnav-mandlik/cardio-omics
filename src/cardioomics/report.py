from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _md_table(df: pd.DataFrame | None, n: int = 10) -> str:
    if df is None or len(df) == 0:
        return "_none_\n"
    df = df.head(n)
    cols = [df.index.name or ""] + list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [_fmt(v) for v in row]) + " |")
    return "\n".join(rows) + "\n"


def build_report(cfg, p) -> dict:
    primary = f"{cfg.rnaseq.contrasts[0][1]}_vs_{cfg.rnaseq.contrasts[0][2]}"
    rna_sum = _json(p.d("rnaseq") / "summary.json")
    prot = _json(p.prot_summary)
    integ = _json(p.integration_summary)
    clf = _json(p.clf_metrics)
    rna = pd.read_csv(p.rna_de(primary), sep="\t", index_col=0) if p.rna_de(primary).exists() else None
    da = pd.read_csv(p.prot_da, sep="\t", index_col=0) if p.prot_da.exists() else None
    conc = pd.read_csv(p.concordance, sep="\t", index_col=0) if p.concordance.exists() else None

    n_de = int(rna["significant"].sum()) if rna is not None else 0
    sig = integ.get("signature_transfer", {})
    fc = integ.get("fold_change", {})
    pw = integ.get("pathways", {})

    top_rna = None
    if rna is not None:
        s = rna[rna["significant"] & (rna["baseMean"] >= 50)]
        top_rna = pd.concat([s[s["log2FoldChange"] > 0].nlargest(8, "log2FoldChange"),
                             s[s["log2FoldChange"] < 0].nsmallest(8, "log2FoldChange")])
        top_rna = top_rna[["baseMean", "log2FoldChange", "padj"]]
    both = conc[conc["class"] == "concordant_both_sig"][["rna_log2fc", "prot_log2fc", "prot_padj"]] if conc is not None else None
    prot_top = da[["gene", "log2FoldChange", "padj"]] if da is not None else None

    lines = [
        f"# CardioOmics report: `{cfg.run_name}`",
        f"_generated {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Summary",
        f"- RNA-seq: {rna_sum.get('n_samples', '?')} LV samples, {rna_sum.get('n_genes', '?')} genes; "
        f"**{n_de:,} DE genes** ({primary.replace('_', ' ')}, FDR < {cfg.rnaseq.alpha}, |log2FC| >= {cfg.rnaseq.lfc_threshold})",
        f"- Proteomics: {prot.get('n_proteins_tested', '?')} proteins tested, "
        f"**{int(da['significant'].sum()) if da is not None else '?'} significant** "
        f"({cfg.proteomics.case} vs {cfg.proteomics.control}, Welch t-test + BH)",
        f"- RNA signature -> proteome: **AUROC {sig.get('auroc', float('nan')):.2f}**, "
        f"permutation p = {sig.get('perm_p', float('nan')):.3g}",
        f"- RNA vs protein fold change: rho = {fc.get('spearman_rho_all', float('nan')):.2f} "
        f"({fc.get('n_shared_genes', 0):,} genes); {fc.get('n_both_significant', 0)} genes significant in both",
        f"- Pathways (Reactome NES): rho = {pw.get('spearman_rho', float('nan')):.2f}; "
        f"{pw.get('n_both_fdr05_same_direction', 0)}/{pw.get('n_both_fdr05', 0)} shared significant pathways same direction",
        f"- Classifier ({cfg.classifier.positive} vs {cfg.classifier.negative}, nested CV): AUROC {clf.get('auroc', float('nan')):.3f} "
        f"vs {clf.get('baseline_auroc', float('nan')):.3f} for {'+'.join(clf.get('baseline_genes', []))}",
        "",
        "## Figures",
    ]
    for f in sorted(p.d("figures").glob("*.png")):
        lines.append(f"![{f.stem}](../figures/{f.name})")
    lines += [
        "",
        "## Top RNA genes",
        _md_table(top_rna, 16),
        "## Top proteins",
        _md_table(prot_top, 12),
        "## Significant in both layers",
        _md_table(both, 30),
    ]
    p.report_md.parent.mkdir(parents=True, exist_ok=True)
    p.report_md.write_text("\n".join(lines))
    summary = {
        "run_name": cfg.run_name,
        "rnaseq": rna_sum,
        "n_de_genes_primary": n_de,
        "proteomics": prot,
        "integration": integ,
        "classifier": clf,
    }
    p.run_summary.write_text(json.dumps(summary, indent=2, default=float))
    return summary
