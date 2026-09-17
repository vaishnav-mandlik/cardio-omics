from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sources(_Strict):
    rnaseq_counts_url: str
    rnaseq_series_matrix_url: str
    gene_info_url: str
    proteomics_url: str
    reactome_gmt_url: str


class RnaseqCfg(_Strict):
    exclude_etiologies: list[str] = Field(default_factory=lambda: ["PPCM"])
    min_count: int = 10
    min_samples: int | None = None
    design: str = "~ sex + race + age_z + etiology"
    contrasts: list[tuple[str, str, str]] = Field(
        default_factory=lambda: [("etiology", "DCM", "NF"), ("etiology", "HCM", "NF")]
    )
    alpha: float = 0.05
    lfc_threshold: float = 0.585
    pca_top_genes: int = 500
    outlier_mad: float = 5.0


class ProteomicsCfg(_Strict):
    chamber: str = "LV"
    case: str = "ICM"
    control: str = "Donor"
    min_valid_frac: float = 0.7
    median_center: bool = True
    alpha: float = 0.05


class EnrichmentCfg(_Strict):
    min_size: int = 15
    max_size: int = 500
    permutations: int = 1000
    fdr: float = 0.05


class NetworkCfg(_Strict):
    top_n: int = 150
    required_score: int = 700
    string_api: str = "https://version-12-0.string-db.org/api"


class IntegrationCfg(_Strict):
    signature_padj: float = 0.01
    signature_lfc: float = 0.585
    signature_max_genes: int = 100
    n_permutations: int = 2000


class ClassifierCfg(_Strict):
    positive: str = "DCM"
    negative: str = "NF"
    outer_folds: int = 5
    inner_folds: int = 3
    n_top_var_genes: int = 2000
    C_grid: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0])
    l1_ratio_grid: list[float] = Field(default_factory=lambda: [0.2, 0.5, 0.8])
    baseline_genes: list[str] = Field(default_factory=lambda: ["NPPA", "NPPB"])


class Config(_Strict):
    run_name: str
    outdir: Path = Path("results")
    datadir: Path = Path("data")
    cachedir: Path = Path(".cache")
    seed: int = 20260923
    offline: bool = False
    sources: Sources
    rnaseq: RnaseqCfg = Field(default_factory=RnaseqCfg)
    proteomics: ProteomicsCfg = Field(default_factory=ProteomicsCfg)
    enrichment: EnrichmentCfg = Field(default_factory=EnrichmentCfg)
    network: NetworkCfg = Field(default_factory=NetworkCfg)
    integration: IntegrationCfg = Field(default_factory=IntegrationCfg)
    classifier: ClassifierCfg = Field(default_factory=ClassifierCfg)


def load_config(path: str | Path) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
