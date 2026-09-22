from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

N_GENES = 400
PLANTED_UP = [f"UPG{i}" for i in range(1, 16)] + ["NPPA", "NPPB"]
PLANTED_DOWN = [f"DNG{i}" for i in range(1, 16)] + ["MYH6", "ATP2A2"]
GROUPS = {"Non-Failing Donor": 16, "Dilated cardiomyopathy (DCM)": 16, "Hypertrophic cardiomyopathy (HCM)": 8, "Peripartum cardiomyopathy (PPCM)": 3}


def _gene_names() -> list[str]:
    named = PLANTED_UP + PLANTED_DOWN
    return named + [f"BKG{i}" for i in range(1, N_GENES - len(named) + 1)]


def write_fixtures(raw_dir: str | Path, seed: int = 7) -> dict[str, Path]:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    genes = _gene_names()
    gene_ids = np.arange(100001, 100001 + len(genes))

    gi = pd.DataFrame({"#tax_id": 9606, "GeneID": gene_ids, "Symbol": genes, "LocusTag": "-", "Synonyms": "-"})
    with gzip.open(raw / "Homo_sapiens.gene_info.gz", "wt") as fh:
        gi.to_csv(fh, sep="\t", index=False)

    samples, etio = [], []
    for label, n in GROUPS.items():
        for _ in range(n):
            samples.append(f"GSM9{len(samples):05d}")
            etio.append(label)
    n = len(samples)
    sex = rng.choice(["Male", "Female"], n)
    race = rng.choice(["Caucasian", "African American"], n)
    age = rng.integers(18, 75, n)

    base = rng.lognormal(mean=5.5, sigma=1.2, size=len(genes))
    lfc = np.zeros((n, len(genes)))
    failing = np.isin(etio, ["Dilated cardiomyopathy (DCM)", "Hypertrophic cardiomyopathy (HCM)"])
    for g in PLANTED_UP:
        lfc[failing, genes.index(g)] = 1.5
    for g in PLANTED_DOWN:
        lfc[failing, genes.index(g)] = -1.5
    libsize = rng.uniform(0.7, 1.3, n)[:, None]
    mu = base[None, :] * (2.0**lfc) * libsize
    disp = 0.08
    counts = rng.negative_binomial(1 / disp, 1 / (1 + mu * disp))
    cm = pd.DataFrame(counts.T, index=gene_ids, columns=samples)
    cm.index.name = "GeneID"
    with gzip.open(raw / "GSE141910_raw_counts_GRCh38.p13_NCBI.tsv.gz", "wt") as fh:
        cm.to_csv(fh, sep="\t")

    def row(key, vals):
        return key + "\t" + "\t".join(f'"{v}"' for v in vals) + "\n"

    with gzip.open(raw / "GSE141910_series_matrix.txt.gz", "wt") as fh:
        fh.write('!Series_title\t"synthetic fixture"\n')
        fh.write(row("!Sample_title", [f"C{i:05d}" for i in range(n)]))
        fh.write(row("!Sample_geo_accession", samples))
        fh.write(row("!Sample_characteristics_ch1", [f"etiology: {e}" for e in etio]))
        fh.write(row("!Sample_characteristics_ch1", [f"Sex: {s}" for s in sex]))
        fh.write(row("!Sample_characteristics_ch1", [f"race: {r}" for r in race]))
        fh.write(row("!Sample_characteristics_ch1", [f"age: {a}" for a in age]))
        fh.write("!series_matrix_table_begin\n!series_matrix_table_end\n")

    prot_genes = genes[:200] + ["A2M;PZP"]
    cols = []
    for cond, n_pat in (("Donor", 7), ("ICM", 7)):
        for i in range(n_pat):
            for ch in ("LV", "RV"):
                cols.append((ch, cond, f"{cond}{i + 1}", "No"))
    cols.append(("LV", "Donor", "Donor1", "Yes"))
    header = {
        "LV/ RV": [c[0] for c in cols],
        "Condition": [c[1] for c in cols],
        "Diabetes": ["No"] * len(cols),
        "Gender": rng.choice(["M", "F"], len(cols)).tolist(),
        "Age": rng.integers(30, 70, len(cols)).tolist(),
        "Patient ID": [c[2] for c in cols],
        "Sample ID": [f"Raw_SmpID_{i}" for i in range(len(cols))],
        "Replicate": [c[3] for c in cols],
    }
    pbase = rng.normal(22, 2.0, len(prot_genes))
    vals = np.empty((len(prot_genes), len(cols)))
    for j, c in enumerate(cols):
        shift = np.zeros(len(prot_genes))
        if c[1] == "ICM":
            for g in PLANTED_UP:
                if g in prot_genes:
                    shift[prot_genes.index(g)] = 1.2
            for g in PLANTED_DOWN:
                if g in prot_genes:
                    shift[prot_genes.index(g)] = -1.2
        vals[:, j] = pbase + shift + rng.normal(0, 0.35, len(prot_genes))
    p_miss = 1 / (1 + np.exp((vals - 20.5) * 2.0))
    vals[rng.uniform(size=vals.shape) < p_miss] = np.nan
    intens = np.where(np.isnan(vals), np.nan, 2.0**vals)
    rows = [[k, *v] for k, v in header.items()] + [[g, *intens[i]] for i, g in enumerate(prot_genes)]
    pd.DataFrame(rows).to_excel(raw / "PXD042155_LVRV_normalised.xlsx", header=False, index=False, sheet_name="LVRV Normalised data")

    sets = {
        "Fixture up pathway (R-HSA-0000001)": PLANTED_UP,
        "Fixture down pathway (R-HSA-0000002)": PLANTED_DOWN,
    }
    bkg = [g for g in genes if g.startswith("BKG")]
    for i in range(6):
        sets[f"Fixture background {i} (R-HSA-00001{i:02d})"] = list(rng.choice(bkg, 20, replace=False))
    gmt = "".join(
        f"{name.rsplit(' (', 1)[0]}\t{name.rsplit('(', 1)[1].rstrip(')')}\t" + "\t".join(g) + "\n"
        for name, g in sets.items()
    )
    with zipfile.ZipFile(raw / "ReactomePathways.gmt.zip", "w") as zf:
        zf.writestr("ReactomePathways.gmt", gmt)

    return {
        "rnaseq_counts": raw / "GSE141910_raw_counts_GRCh38.p13_NCBI.tsv.gz",
        "series_matrix": raw / "GSE141910_series_matrix.txt.gz",
        "gene_info": raw / "Homo_sapiens.gene_info.gz",
        "proteomics_xlsx": raw / "PXD042155_LVRV_normalised.xlsx",
        "reactome_gmt": raw / "ReactomePathways.gmt.zip",
    }
