from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ETIOLOGY_CODES = {
    "Non-Failing Donor": "NF",
    "Dilated cardiomyopathy (DCM)": "DCM",
    "Hypertrophic cardiomyopathy (HCM)": "HCM",
    "Peripartum cardiomyopathy (PPCM)": "PPCM",
}


def parse_series_matrix(path: str | Path) -> pd.DataFrame:
    fields: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, *vals = line.rstrip("\n").split("\t")
            vals = [v.strip().strip('"') for v in vals]
            if key == "!Sample_characteristics_ch1":
                characteristics.append(vals)
            elif key in ("!Sample_geo_accession", "!Sample_title"):
                fields[key] = vals
    if "!Sample_geo_accession" not in fields:
        raise ValueError(f"{path}: no !Sample_geo_accession line - not a series matrix?")
    meta = pd.DataFrame({"sample": fields["!Sample_geo_accession"], "title": fields.get("!Sample_title")})
    for row in characteristics:
        keys = {v.split(":", 1)[0].strip().lower() for v in row if ":" in v}
        if len(keys) != 1:
            raise ValueError(f"inconsistent characteristic keys in row: {sorted(keys)[:5]}")
        key = re.sub(r"\W+", "_", keys.pop())
        meta[key] = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in row]
    missing = {"etiology", "sex", "race", "age"} - set(meta.columns)
    if missing:
        raise ValueError(f"series matrix lacks characteristics: {sorted(missing)}")
    unknown = set(meta["etiology"]) - set(ETIOLOGY_CODES)
    if unknown:
        raise ValueError(f"unmapped etiology labels: {sorted(unknown)}")
    meta["etiology"] = meta["etiology"].map(ETIOLOGY_CODES)
    meta["age"] = pd.to_numeric(meta["age"], errors="raise")
    meta["sex"] = meta["sex"].str.strip().str.capitalize()
    meta["race"] = meta["race"].str.replace(r"\s+", "_", regex=True)
    return meta.set_index("sample")


def load_gene_symbols(gene_info_path: str | Path) -> pd.Series:
    gi = pd.read_csv(gene_info_path, sep="\t", usecols=["GeneID", "Symbol"], dtype={"GeneID": "int64"})
    return gi.drop_duplicates("GeneID").set_index("GeneID")["Symbol"]


def load_counts(counts_path: str | Path, symbols: pd.Series) -> pd.DataFrame:
    raw = pd.read_csv(counts_path, sep="\t", index_col=0)
    raw.index = raw.index.astype("int64")
    sym = symbols.reindex(raw.index)
    raw = raw.loc[sym.notna()]
    raw.index = sym.dropna().to_numpy()
    counts = raw.groupby(level=0).sum()
    if (counts.to_numpy() < 0).any():
        raise ValueError("negative counts found")
    return counts.T.astype("int64")


@dataclass
class PreparedRnaseq:
    counts: pd.DataFrame
    meta: pd.DataFrame
    qc: pd.DataFrame
    pca: pd.DataFrame
    explained_variance: list[float]


def log_cpm(counts: pd.DataFrame, prior: float = 1.0) -> pd.DataFrame:
    lib = counts.sum(axis=1)
    return np.log2((counts.add(prior)).div(lib + 2 * prior, axis=0) * 1e6)


def _mad_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / (mad if mad > 0 else 1.0)


def prepare(counts: pd.DataFrame, meta: pd.DataFrame, cfg) -> PreparedRnaseq:
    shared = counts.index.intersection(meta.index)
    if len(shared) == 0:
        raise ValueError("no overlap between count columns and series-matrix samples")
    meta = meta.loc[shared]
    meta = meta.loc[~meta["etiology"].isin(cfg.exclude_etiologies)].copy()
    counts = counts.loc[meta.index]

    min_samples = cfg.min_samples or int(meta["etiology"].value_counts().min())
    keep = (counts >= cfg.min_count).sum(axis=0) >= min_samples
    counts = counts.loc[:, keep]

    meta["age_z"] = (meta["age"] - meta["age"].mean()) / meta["age"].std(ddof=0)
    for col in ("etiology", "sex", "race"):
        meta[col] = meta[col].astype("category")

    lib = counts.sum(axis=1)
    lcpm = log_cpm(counts)
    top = lcpm.var(axis=0).nlargest(min(cfg.pca_top_genes, lcpm.shape[1])).index
    X = lcpm[top].to_numpy()
    X = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pca = pd.DataFrame(U[:, :5] * S[:5], index=counts.index, columns=[f"PC{i + 1}" for i in range(5)])
    pca = pca.join(meta[["etiology", "sex", "race"]])
    var_exp = (S**2 / (S**2).sum())[:5].tolist()

    dist = np.sqrt(pca["PC1"] ** 2 + pca["PC2"] ** 2)
    qc = pd.DataFrame(
        {
            "library_size": lib,
            "detected_genes": (counts > 0).sum(axis=1),
            "libsize_mad_z": _mad_z(np.log10(lib)),
            "pca_dist_mad_z": _mad_z(dist),
        }
    )
    qc["outlier"] = (qc["libsize_mad_z"].abs() > cfg.outlier_mad) | (qc["pca_dist_mad_z"] > cfg.outlier_mad)
    return PreparedRnaseq(counts, meta, qc, pca, var_exp)


def run_deseq2(counts: pd.DataFrame, meta: pd.DataFrame, cfg, n_cpus: int = 1) -> dict[str, pd.DataFrame]:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts, metadata=meta, design=cfg.design, refit_cooks=True, inference=inference, quiet=True
    )
    dds.deseq2()
    results: dict[str, pd.DataFrame] = {}
    for factor, test, ref in cfg.contrasts:
        stat = DeseqStats(dds, contrast=[factor, test, ref], alpha=cfg.alpha, inference=inference, quiet=True)
        stat.summary()
        res = stat.results_df.copy()
        res.index.name = "gene"
        res["significant"] = (res["padj"] < cfg.alpha) & (res["log2FoldChange"].abs() >= cfg.lfc_threshold)
        results[f"{test}_vs_{ref}"] = res.sort_values("pvalue")
    return results
