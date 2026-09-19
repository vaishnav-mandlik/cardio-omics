from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_gmt(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".gmt")]
            if len(members) != 1:
                raise ValueError(f"{path}: expected exactly one .gmt, found {members}")
            text = zf.read(members[0]).decode("utf-8")
    else:
        text = path.read_text()
    sets: dict[str, list[str]] = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        sets[f"{parts[0]} ({parts[1]})"] = sorted({g for g in parts[2:] if g})
    if not sets:
        raise ValueError(f"{path}: no gene sets parsed")
    return sets


def ranking_from_table(table: pd.DataFrame, stat_col: str, gene_col: str | None = None) -> pd.Series:
    genes = table.index.to_series() if gene_col is None else table[gene_col]
    df = pd.DataFrame({"gene": genes.to_numpy(), "stat": table[stat_col].to_numpy()}).dropna()
    df = df.assign(abs_stat=df["stat"].abs()).sort_values(["abs_stat", "gene"], ascending=[False, True])
    df = df.drop_duplicates("gene")
    return df.set_index("gene")["stat"].sort_values(ascending=False, kind="mergesort")


def run_prerank(ranking: pd.Series, gene_sets: dict[str, list[str]], cfg, seed: int, threads: int = 1) -> pd.DataFrame:
    import gseapy

    pre = gseapy.prerank(
        rnk=ranking,
        gene_sets=gene_sets,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        permutation_num=cfg.permutations,
        threads=threads,
        seed=seed,
        no_plot=True,
        outdir=None,
        verbose=False,
    )
    res = pre.res2d.rename(columns={"NOM p-val": "pvalue", "FDR q-val": "fdr", "Lead_genes": "lead_genes"})
    res = res[["Term", "ES", "NES", "pvalue", "fdr", "lead_genes"]].copy()
    for col in ("ES", "NES", "pvalue", "fdr"):
        res[col] = pd.to_numeric(res[col], errors="coerce")
    res["significant"] = res["fdr"] < cfg.fdr
    return res.sort_values(["fdr", "NES"], key=lambda s: s if s.name == "fdr" else -np.abs(s))
