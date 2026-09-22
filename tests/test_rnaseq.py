import pandas as pd
import pytest

from cardioomics import rnaseq
from cardioomics.config import RnaseqCfg
from cardioomics.fixtures import PLANTED_DOWN, PLANTED_UP


@pytest.fixture(scope="module")
def prepared(fixture_files):
    meta = rnaseq.parse_series_matrix(fixture_files["series_matrix"])
    counts = rnaseq.load_counts(fixture_files["rnaseq_counts"], rnaseq.load_gene_symbols(fixture_files["gene_info"]))
    cfg = RnaseqCfg(min_count=5, design="~ sex + age_z + etiology", pca_top_genes=200)
    return rnaseq.prepare(counts, meta, cfg), cfg


def test_series_matrix_parsing(fixture_files):
    meta = rnaseq.parse_series_matrix(fixture_files["series_matrix"])
    assert set(meta["etiology"]) == {"NF", "DCM", "HCM", "PPCM"}
    assert meta["age"].dtype.kind in "if"
    assert meta.index.str.startswith("GSM").all()


def test_series_matrix_rejects_unknown_etiology(tmp_path):
    bad = tmp_path / "sm.txt"
    bad.write_text('!Sample_geo_accession\t"GSM1"\n!Sample_characteristics_ch1\t"etiology: Alien"\n'
                   '!Sample_characteristics_ch1\t"Sex: Male"\n!Sample_characteristics_ch1\t"race: X"\n'
                   '!Sample_characteristics_ch1\t"age: 3"\n')
    with pytest.raises(ValueError, match="unmapped etiology"):
        rnaseq.parse_series_matrix(bad)


def test_counts_are_symbol_keyed_integers(fixture_files):
    counts = rnaseq.load_counts(fixture_files["rnaseq_counts"], rnaseq.load_gene_symbols(fixture_files["gene_info"]))
    assert "NPPA" in counts.columns
    assert all(pd.api.types.is_integer_dtype(t) for t in counts.dtypes)


def test_prepare_excludes_ppcm_and_filters(prepared):
    prep, _ = prepared
    assert "PPCM" not in set(prep.meta["etiology"].astype(str))
    assert prep.counts.shape[0] == prep.meta.shape[0]
    assert {"library_size", "outlier"} <= set(prep.qc.columns)
    assert abs(prep.meta["age_z"].mean()) < 1e-9


def test_deseq2_recovers_planted_genes(prepared):
    prep, cfg = prepared
    res = rnaseq.run_deseq2(prep.counts, prep.meta, cfg, n_cpus=1)["DCM_vs_NF"]
    up = res.loc[[g for g in PLANTED_UP if g in res.index]]
    down = res.loc[[g for g in PLANTED_DOWN if g in res.index]]
    assert (up["log2FoldChange"] > 0.7).mean() > 0.8
    assert (down["log2FoldChange"] < -0.7).mean() > 0.8
    assert (up["padj"] < 0.05).mean() > 0.7
    bkg = res.loc[res.index.str.startswith("BKG")]
    assert (bkg["padj"] < 0.05).mean() < 0.1
