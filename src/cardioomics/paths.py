from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    datadir: Path
    outdir: Path

    @property
    def raw(self) -> Path:
        return self.datadir / "raw"

    @property
    def rnaseq_counts(self) -> Path:
        return self.raw / "GSE141910_raw_counts_GRCh38.p13_NCBI.tsv.gz"

    @property
    def series_matrix(self) -> Path:
        return self.raw / "GSE141910_series_matrix.txt.gz"

    @property
    def gene_info(self) -> Path:
        return self.raw / "Homo_sapiens.gene_info.gz"

    @property
    def proteomics_xlsx(self) -> Path:
        return self.raw / "PXD042155_LVRV_normalised.xlsx"

    @property
    def reactome_gmt(self) -> Path:
        return self.raw / "ReactomePathways.gmt.zip"

    def raw_inputs(self) -> dict[str, Path]:
        return {
            "rnaseq_counts": self.rnaseq_counts,
            "series_matrix": self.series_matrix,
            "gene_info": self.gene_info,
            "proteomics_xlsx": self.proteomics_xlsx,
            "reactome_gmt": self.reactome_gmt,
        }

    def d(self, name: str) -> Path:
        return self.outdir / name

    @property
    def rna_counts(self) -> Path:
        return self.d("rnaseq") / "counts_filtered.tsv.gz"

    @property
    def rna_meta(self) -> Path:
        return self.d("rnaseq") / "sample_meta.tsv"

    @property
    def rna_qc(self) -> Path:
        return self.d("rnaseq") / "qc.tsv"

    @property
    def rna_pca(self) -> Path:
        return self.d("rnaseq") / "pca.tsv"

    def rna_de(self, contrast: str) -> Path:
        return self.d("rnaseq") / f"de_{contrast}.tsv"

    @property
    def prot_log2(self) -> Path:
        return self.d("proteomics") / "log2_collapsed.tsv.gz"

    @property
    def prot_meta(self) -> Path:
        return self.d("proteomics") / "sample_meta.tsv"

    @property
    def prot_da(self) -> Path:
        return self.d("proteomics") / "differential_abundance.tsv"

    @property
    def prot_missing(self) -> Path:
        return self.d("proteomics") / "missingness.tsv"

    @property
    def prot_summary(self) -> Path:
        return self.d("proteomics") / "summary.json"

    @property
    def gsea_rna(self) -> Path:
        return self.d("enrichment") / "gsea_rna.tsv"

    @property
    def gsea_prot(self) -> Path:
        return self.d("enrichment") / "gsea_protein.tsv"

    @property
    def string_edges(self) -> Path:
        return self.d("network") / "string_edges.tsv"

    @property
    def hubs(self) -> Path:
        return self.d("network") / "hubs.tsv"

    @property
    def concordance(self) -> Path:
        return self.d("integration") / "fold_change_concordance.tsv"

    @property
    def signature_scores(self) -> Path:
        return self.d("integration") / "signature_scores.tsv"

    @property
    def pathway_concordance(self) -> Path:
        return self.d("integration") / "pathway_concordance.tsv"

    @property
    def integration_summary(self) -> Path:
        return self.d("integration") / "summary.json"

    @property
    def clf_metrics(self) -> Path:
        return self.d("classifier") / "metrics.json"

    @property
    def clf_model(self) -> Path:
        return self.d("classifier") / "model.joblib"

    def clf_table(self, name: str) -> Path:
        return self.d("classifier") / f"{name}.tsv"

    def figure(self, name: str) -> Path:
        return self.d("figures") / f"{name}.png"

    def log(self, step: str) -> Path:
        return self.d("logs") / f"{step}.jsonl"

    @property
    def report_md(self) -> Path:
        return self.d("report") / "report.md"

    @property
    def run_summary(self) -> Path:
        return self.d("report") / "summary.json"


def paths_for(cfg) -> Paths:
    return Paths(Path(cfg.datadir), Path(cfg.outdir))
