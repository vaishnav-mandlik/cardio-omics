from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.stats.multitest import multipletests

HEADER_ROWS = {
    "LV/ RV": "chamber",
    "Condition": "condition",
    "Diabetes": "diabetes",
    "Gender": "sex",
    "Age": "age",
    "Patient ID": "patient",
    "Sample ID": "sample_id",
    "Replicate": "replicate",
}


@dataclass
class ProteomicsData:
    log2: pd.DataFrame
    meta: pd.DataFrame


def load_pxd042155(path: str | Path) -> ProteomicsData:
    raw = pd.read_excel(path, header=None)
    labels = raw.iloc[:, 0].astype(str).str.strip()
    n_header = len(HEADER_ROWS)
    got = list(labels.iloc[:n_header])
    if got != list(HEADER_ROWS):
        raise ValueError(f"unexpected header rows {got}; expected {list(HEADER_ROWS)}")
    meta = raw.iloc[:n_header, 1:].T
    meta.columns = list(HEADER_ROWS.values())
    for col in meta.columns:
        meta[col] = meta[col].astype("string").str.strip()
    if meta[["chamber", "condition", "patient"]].isna().any(axis=1).any():
        raise ValueError("some sample columns lack chamber/condition/patient annotation")
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    meta["replicate"] = meta["replicate"].str.lower().eq("yes")
    meta.index = pd.Index([f"S{i:03d}" for i in range(len(meta))], name="sample")

    values = raw.iloc[n_header:, 1:].apply(pd.to_numeric, errors="coerce")
    values.index = raw.iloc[n_header:, 0].astype(str).str.strip().to_numpy()
    values.index.name = "protein_group"
    values.columns = meta.index
    values = values.where(values > 0)
    if values.index.duplicated().any():
        raise ValueError("duplicated protein-group identifiers")
    return ProteomicsData(np.log2(values.T), meta)


def collapse_replicates(data: ProteomicsData) -> ProteomicsData:
    keys = data.meta["patient"] + "|" + data.meta["chamber"]
    log2 = data.log2.groupby(keys.to_numpy()).mean()
    first = data.meta.assign(_key=keys.to_numpy()).drop_duplicates("_key").set_index("_key")
    meta = first.loc[log2.index].drop(columns=["sample_id", "replicate"])
    meta["n_runs"] = keys.value_counts().reindex(log2.index).to_numpy()
    log2.index.name = meta.index.name = "sample"
    return ProteomicsData(log2, meta)


def missingness_table(log2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"mean_log2": log2.mean(axis=0), "missing_frac": log2.isna().mean(axis=0)}
    ).dropna(subset=["mean_log2"])


def filter_valid(log2: pd.DataFrame, groups: pd.Series, min_frac: float) -> pd.DataFrame:
    valid = log2.notna().groupby(groups.to_numpy()).mean()
    return log2.loc[:, (valid >= min_frac).any(axis=0)]


def median_center(log2: pd.DataFrame) -> pd.DataFrame:
    med = log2.median(axis=1)
    return log2.sub(med, axis=0) + med.median()


@dataclass
class DAResult:
    table: pd.DataFrame
    summary: dict
    missingness: pd.DataFrame
    n_samples: dict


def differential_abundance(data: ProteomicsData, cfg) -> DAResult:
    meta = data.meta
    sel = (meta["chamber"] == cfg.chamber) & meta["condition"].isin([cfg.case, cfg.control])
    meta = meta.loc[sel]
    log2 = data.log2.loc[meta.index]
    miss = missingness_table(log2)
    groups = meta["condition"]

    log2 = filter_valid(log2, groups, cfg.min_valid_frac)
    if cfg.median_center:
        log2 = median_center(log2)
    is_case = (groups == cfg.case).to_numpy()
    case, ctrl = log2.loc[is_case], log2.loc[~is_case]

    t, p = sstats.ttest_ind(case, ctrl, axis=0, equal_var=False, nan_policy="omit")
    table = pd.DataFrame(
        {
            "gene": [g.split(";")[0] for g in log2.columns],
            "n_valid_case": case.notna().sum(axis=0).to_numpy(),
            "n_valid_control": ctrl.notna().sum(axis=0).to_numpy(),
            "mean_log2_case": case.mean(axis=0).to_numpy(),
            "mean_log2_control": ctrl.mean(axis=0).to_numpy(),
            "t": np.asarray(t, dtype=float),
            "pvalue": np.asarray(p, dtype=float),
        },
        index=log2.columns,
    )
    table["log2FoldChange"] = table["mean_log2_case"] - table["mean_log2_control"]
    ok = table["pvalue"].notna()
    table["padj"] = np.nan
    table.loc[ok, "padj"] = multipletests(table.loc[ok, "pvalue"], method="fdr_bh")[1]
    table["significant"] = table["padj"] < cfg.alpha
    table.index.name = "protein_group"
    rho = sstats.spearmanr(miss["mean_log2"], miss["missing_frac"]).statistic
    quint = miss.groupby(pd.qcut(miss["mean_log2"], 5, labels=False))["missing_frac"].mean()
    summary = {
        "n_proteins_input": int(miss.shape[0]),
        "n_proteins_tested": int(log2.shape[1]),
        "mnar_spearman_rho": float(rho),
        "missing_frac_by_abundance_quintile": [round(float(v), 4) for v in quint],
    }
    return DAResult(
        table.sort_values("pvalue"),
        summary,
        miss,
        {cfg.case: int(is_case.sum()), cfg.control: int((~is_case).sum())},
    )
