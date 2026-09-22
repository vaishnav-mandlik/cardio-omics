import pandas as pd
import pytest

from cardioomics import proteomics as prot
from cardioomics.config import ProteomicsCfg
from cardioomics.fixtures import PLANTED_UP


@pytest.fixture(scope="module")
def data(fixture_files):
    return prot.load_pxd042155(fixture_files["proteomics_xlsx"])


def test_parse_layout(data):
    assert data.log2.shape[0] == data.meta.shape[0] == 29
    assert {"chamber", "condition", "patient", "replicate"} <= set(data.meta.columns)
    assert data.log2.isna().any().any()
    assert "A2M;PZP" in data.log2.columns


def test_parse_rejects_wrong_sheet(tmp_path):
    bad = tmp_path / "bad.xlsx"
    pd.DataFrame([["Foo", 1], ["Bar", 2]]).to_excel(bad, header=False, index=False)
    with pytest.raises(ValueError, match="unexpected header rows"):
        prot.load_pxd042155(bad)


def test_collapse_replicates(data):
    c = prot.collapse_replicates(data)
    assert c.log2.shape[0] == 28
    assert c.meta.loc[c.meta["patient"] == "Donor1"].query("chamber == 'LV'")["n_runs"].iloc[0] == 2


def test_differential_abundance_finds_planted(data):
    c = prot.collapse_replicates(data)
    res = prot.differential_abundance(c, ProteomicsCfg(min_valid_frac=0.6))
    t = res.table.set_index("gene")
    planted = [g for g in PLANTED_UP if g in t.index]
    assert (t.loc[planted, "log2FoldChange"] > 0.5).mean() > 0.7
    assert res.summary["mnar_spearman_rho"] < -0.2
    assert res.n_samples == {"ICM": 7, "Donor": 7}
