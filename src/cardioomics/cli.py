from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import typer

from . import __version__
from .config import load_config
from .logutil import get_logger, log_kv
from .paths import paths_for

app = typer.Typer(add_completion=False, help="CardioOmics: heart-failure multi-omics pipeline.")
CONFIG = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, help="YAML config file")
THREADS = typer.Option(1, "--threads", "-t", min=1, help="CPU threads for this step")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gseapy")


def _setup(config: Path, step: str):
    cfg = load_config(config)
    p = paths_for(cfg)
    log = get_logger(step, p.log(step))
    log_kv(log, "start", config=str(config), run=cfg.run_name)
    return cfg, p, log


def _write_tsv(df: pd.DataFrame, path: Path, index: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)
    return path


def _write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path


def _primary(cfg) -> str:
    _, test, ref = cfg.rnaseq.contrasts[0]
    return f"{test}_vs_{ref}"


def _load_rna(p):
    counts = pd.read_csv(p.rna_counts, sep="\t", index_col=0).T
    meta = pd.read_csv(p.rna_meta, sep="\t", index_col=0)
    if counts.index.has_duplicates or set(counts.index) != set(meta.index):
        raise ValueError(f"sample mismatch between {p.rna_counts} and {p.rna_meta}")
    meta = meta.loc[counts.index]
    for col in ("etiology", "sex", "race"):
        meta[col] = meta[col].astype("category")
    return counts, meta


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def fetch(config: Path = CONFIG, force: bool = typer.Option(False, help="re-download even if present")) -> None:
    from .fetch import download

    cfg, p, log = _setup(config, "fetch")
    urls = {
        "rnaseq_counts": cfg.sources.rnaseq_counts_url,
        "series_matrix": cfg.sources.rnaseq_series_matrix_url,
        "gene_info": cfg.sources.gene_info_url,
        "proteomics_xlsx": cfg.sources.proteomics_url,
        "reactome_gmt": cfg.sources.reactome_gmt_url,
    }
    for key, dest in p.raw_inputs().items():
        if dest.exists() and not force:
            log_kv(log, "cached", file=str(dest))
            continue
        if cfg.offline:
            raise typer.BadParameter(f"offline mode and {dest} is missing")
        download(urls[key], dest)
        log_kv(log, "downloaded", file=str(dest), bytes=dest.stat().st_size)


@app.command("rnaseq-prep")
def rnaseq_prep(config: Path = CONFIG) -> None:
    from . import plots, rnaseq

    cfg, p, log = _setup(config, "rnaseq_prep")
    meta = rnaseq.parse_series_matrix(p.series_matrix)
    counts = rnaseq.load_counts(p.rnaseq_counts, rnaseq.load_gene_symbols(p.gene_info))
    no_counts = sorted(set(meta.index) - set(counts.index))
    if no_counts:
        log_kv(log, "samples without counts skipped", n=len(no_counts), samples=no_counts)
    prep = rnaseq.prepare(counts, meta, cfg.rnaseq)
    _write_tsv(prep.counts.T, p.rna_counts)
    _write_tsv(prep.meta, p.rna_meta)
    _write_tsv(prep.qc, p.rna_qc)
    _write_tsv(prep.pca, p.rna_pca)
    plots.pca(prep.pca, prep.explained_variance, p.figure("rna_pca"))
    plots.library_sizes(prep.qc, prep.meta, p.figure("rna_library_sizes"))
    summary = {
        "n_series_samples": int(len(meta)),
        "n_series_samples_without_counts": len(no_counts),
        "n_samples": int(prep.counts.shape[0]),
        "n_genes": int(prep.counts.shape[1]),
        "groups": prep.meta["etiology"].value_counts().to_dict(),
        "n_outliers_flagged": int(prep.qc["outlier"].sum()),
        "pca_explained_variance": prep.explained_variance,
    }
    _write_json(summary, p.d("rnaseq") / "summary.json")
    log_kv(log, "done", n_samples=summary["n_samples"], n_genes=summary["n_genes"])


@app.command("rnaseq-de")
def rnaseq_de(config: Path = CONFIG, threads: int = THREADS) -> None:
    from . import plots, rnaseq

    cfg, p, log = _setup(config, "rnaseq_de")
    counts, meta = _load_rna(p)
    for name, res in rnaseq.run_deseq2(counts, meta, cfg.rnaseq, n_cpus=threads).items():
        _write_tsv(res, p.rna_de(name))
        plots.volcano(res, f"{name.replace('_vs_', ' vs ')} (LV RNA-seq)", p.figure(f"volcano_rna_{name}"))
        log_kv(log, "contrast", contrast=name, n_significant=int(res["significant"].sum()))


@app.command()
def proteomics(config: Path = CONFIG) -> None:
    from . import plots
    from . import proteomics as prot

    cfg, p, log = _setup(config, "proteomics")
    data = prot.collapse_replicates(prot.load_pxd042155(p.proteomics_xlsx))
    _write_tsv(data.log2, p.prot_log2)
    _write_tsv(data.meta, p.prot_meta)
    res = prot.differential_abundance(data, cfg.proteomics)
    _write_tsv(res.table, p.prot_da)
    _write_tsv(res.missingness, p.prot_missing)
    summary = {**res.summary, "n_samples": res.n_samples, "n_significant": int(res.table["significant"].sum())}
    _write_json(summary, p.prot_summary)
    plots.missingness(res.missingness, res.summary["mnar_spearman_rho"], p.figure("proteomics_missingness"))
    plots.volcano(
        res.table,
        f"{cfg.proteomics.case} vs {cfg.proteomics.control} ({cfg.proteomics.chamber} proteome)",
        p.figure("volcano_protein"),
        lfc=0.0,
        label_col="gene",
    )
    log_kv(log, "done", n_tested=summary["n_proteins_tested"], n_significant=summary["n_significant"])


@app.command()
def enrich(config: Path = CONFIG, threads: int = THREADS) -> None:
    from . import enrichment, plots
    from .integrate import gene_level_protein

    cfg, p, log = _setup(config, "enrich")
    sets = enrichment.load_gmt(p.reactome_gmt)
    rna = pd.read_csv(p.rna_de(_primary(cfg)), sep="\t", index_col=0)
    prot_gene = gene_level_protein(pd.read_csv(p.prot_da, sep="\t", index_col=0))
    g_rna = enrichment.run_prerank(enrichment.ranking_from_table(rna, "stat"), sets, cfg.enrichment, cfg.seed, threads)
    g_prot = enrichment.run_prerank(enrichment.ranking_from_table(prot_gene, "t"), sets, cfg.enrichment, cfg.seed, threads)
    _write_tsv(g_rna, p.gsea_rna, index=False)
    _write_tsv(g_prot, p.gsea_prot, index=False)
    plots.gsea_bar(g_rna, "Reactome GSEA - transcriptome", p.figure("gsea_rna"))
    plots.gsea_bar(g_prot, "Reactome GSEA - proteome", p.figure("gsea_protein"))
    log_kv(log, "done", rna_sig=int(g_rna["significant"].sum()), prot_sig=int(g_prot["significant"].sum()))


@app.command()
def network(config: Path = CONFIG) -> None:
    import requests

    from . import network as net
    from . import plots

    cfg, p, log = _setup(config, "network")
    rna = pd.read_csv(p.rna_de(_primary(cfg)), sep="\t", index_col=0)
    top = rna[rna["significant"]].sort_values("padj").head(cfg.network.top_n)
    try:
        edges = net.fetch_string_network(
            top.index.tolist(),
            cfg.cachedir / "string",
            api=cfg.network.string_api,
            required_score=cfg.network.required_score,
            offline=cfg.offline,
        )
    except (FileNotFoundError, requests.RequestException, ValueError) as err:
        log.warning(f"STRING unavailable ({err}); continuing with an empty network")
        edges = pd.DataFrame(columns=["gene_a", "gene_b", "score"])
    G, hubs = net.hub_table(edges, top[["log2FoldChange", "padj"]])
    _write_tsv(edges, p.string_edges, index=False)
    _write_tsv(hubs, p.hubs, index=False)
    plots.network(G, top["log2FoldChange"].to_dict(), p.figure("string_network"))
    log_kv(log, "done", n_edges=len(edges), n_nodes=G.number_of_nodes())


@app.command()
def integrate(config: Path = CONFIG) -> None:
    from . import integrate as integ
    from . import plots
    from . import proteomics as prot

    cfg, p, log = _setup(config, "integrate")
    rna = pd.read_csv(p.rna_de(_primary(cfg)), sep="\t", index_col=0)
    prot_gene = integ.gene_level_protein(pd.read_csv(p.prot_da, sep="\t", index_col=0))
    conc, conc_sum = integ.concordance(rna, prot_gene)

    log2 = pd.read_csv(p.prot_log2, sep="\t", index_col=0)
    meta = pd.read_csv(p.prot_meta, sep="\t", index_col=0)
    sel = (meta["chamber"] == cfg.proteomics.chamber) & meta["condition"].isin([cfg.proteomics.case, cfg.proteomics.control])
    lv = log2.loc[sel]
    if cfg.proteomics.median_center:
        lv = prot.median_center(lv)
    scores, sig_sum = integ.signature_transfer(rna, lv, meta.loc[sel], cfg.proteomics.case, cfg.integration, cfg.seed)
    path_df, path_sum = integ.pathway_concordance(pd.read_csv(p.gsea_rna, sep="\t"), pd.read_csv(p.gsea_prot, sep="\t"))

    _write_tsv(conc, p.concordance)
    _write_tsv(scores, p.signature_scores)
    _write_tsv(path_df, p.pathway_concordance)
    _write_json({"fold_change": conc_sum, "signature_transfer": sig_sum, "pathways": path_sum}, p.integration_summary)
    plots.concordance(conc, conc_sum["spearman_rho_all"], p.figure("concordance"))
    plots.signature_scores(scores, sig_sum, p.figure("signature_transfer"))
    plots.nes_scatter(path_df, path_sum, p.figure("pathway_concordance"))
    log_kv(log, "done", sig_auroc=sig_sum["auroc"], sig_p=sig_sum["perm_p"])


@app.command()
def classify(config: Path = CONFIG, threads: int = THREADS) -> None:
    import joblib

    from . import classify as clf
    from . import plots

    cfg, p, log = _setup(config, "classify")
    counts, meta = _load_rna(p)
    res = clf.nested_cv(counts, meta, cfg.classifier, cfg.seed, n_jobs=threads)
    _write_json(res.metrics, p.clf_metrics)
    _write_tsv(res.oof, p.clf_table("oof_predictions"))
    _write_tsv(res.coefficients, p.clf_table("coefficients"), index=False)
    p.clf_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": res.final_model, "genes": counts.columns.tolist()}, p.clf_model)
    plots.roc(res.roc, res.metrics, p.figure("classifier_roc"))
    log_kv(log, "done", auroc=res.metrics["auroc"], baseline=res.metrics.get("baseline_auroc"))


@app.command()
def report(config: Path = CONFIG) -> None:
    from .report import build_report

    cfg, p, log = _setup(config, "report")
    build_report(cfg, p)
    log_kv(log, "done", report=str(p.report_md))


@app.command("make-fixtures")
def make_fixtures(outdir: Path = typer.Argument(..., help="where to write the synthetic fixture inputs")) -> None:
    from .fixtures import write_fixtures

    write_fixtures(outdir)
    typer.echo(f"fixtures written to {outdir}")


if __name__ == "__main__":
    app()
