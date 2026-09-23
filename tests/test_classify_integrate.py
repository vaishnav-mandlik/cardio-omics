import numpy as np
import pandas as pd
import pytest

from cardioomics import classify, integrate
from cardioomics.config import ClassifierCfg, IntegrationCfg


def _toy(n=60, g=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    mu = np.full((n, g), 200.0)
    mu[y == 1, :15] *= 3
    counts = pd.DataFrame(rng.poisson(mu), columns=[f"G{i}" for i in range(g)], index=[f"S{i}" for i in range(n)])
    counts = counts.rename(columns={"G0": "NPPA", "G1": "NPPB"})
    meta = pd.DataFrame(
        {
            "etiology": np.where(y == 1, "DCM", "NF"),
            "sex": rng.choice(["Male", "Female"], n),
            "race": rng.choice(["A", "B"], n),
            "age": rng.integers(20, 70, n),
        },
        index=counts.index,
    )
    return counts, meta


CFG = ClassifierCfg(outer_folds=3, inner_folds=3, n_top_var_genes=50, C_grid=[1.0], l1_ratio_grid=[0.5])


def test_topvariance_fits_on_train_only():
    X = np.array([[0, 1.0, 5], [0, 2.0, 5], [0, 3.0, 5]])
    assert classify.TopVariance(k=1).fit(X).support_.tolist() == [1]


def test_nested_cv_on_separable_data():
    counts, meta = _toy()
    res = classify.nested_cv(counts, meta, CFG, seed=0)
    assert res.metrics["auroc"] > 0.9
    assert res.oof["prob"].notna().all()
    assert "baseline_auroc" in res.metrics


def test_shuffled_metadata_rows_do_not_misalign_labels():
    counts, meta = _toy()
    assert classify.nested_cv(counts, meta.sample(frac=1, random_state=1), CFG, seed=0).metrics["auroc"] > 0.9


def test_mismatched_samples_raise():
    counts, meta = _toy()
    with pytest.raises(ValueError, match="differ"):
        classify.nested_cv(counts, meta.iloc[1:], CFG, seed=0)


def test_signature_transfer_beats_random():
    rng = np.random.default_rng(1)
    genes = [f"P{i}" for i in range(200)]
    rna = pd.DataFrame({"log2FoldChange": 0.0, "padj": 0.9}, index=genes)
    rna.loc[genes[:10], ["log2FoldChange", "padj"]] = [1.5, 1e-6]
    y = np.repeat(["Donor", "ICM"], 10)
    P = pd.DataFrame(rng.normal(20, 1, (20, 200)), columns=genes)
    P.loc[y == "ICM", genes[:10]] += 1.5
    meta = pd.DataFrame({"condition": y}, index=P.index)
    scores, s = integrate.signature_transfer(rna, P, meta, "ICM", IntegrationCfg(n_permutations=200), seed=0)
    assert s["auroc"] > 0.9 and s["perm_p"] < 0.05
    assert len(scores) == 20


def test_concordance_classes():
    rna = pd.DataFrame({"log2FoldChange": [1, -1, 1, 0.1], "padj": [0.001, 0.001, 0.001, 0.5]}, index=list("ABCD"))
    prot = pd.DataFrame({"log2FoldChange": [0.8, -0.9, -0.7, 0.0], "padj": [0.01, 0.01, 0.01, 0.9]}, index=list("ABCD"))
    df, s = integrate.concordance(rna, prot)
    assert df.loc["A", "class"] == "concordant_both_sig"
    assert df.loc["C", "class"] == "discordant_both_sig"
    assert s["direction_agreement_both_sig"] == pytest.approx(2 / 3)
