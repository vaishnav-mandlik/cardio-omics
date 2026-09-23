import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from cardioomics import enrichment, network
from cardioomics.config import load_config
from cardioomics.fetch import DownloadError, check_magic


def test_configs_validate():
    assert load_config("config/config.yaml").run_name
    assert load_config("config/test.yaml").offline is True


def test_unknown_config_key_is_rejected(tmp_path):
    with open("config/config.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["rnaseq"]["min_cout"] = 3
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_magic_check_rejects_html_as_gzip(tmp_path):
    tmp = tmp_path / "x.tmp"
    tmp.write_bytes(b"<!doctype html>")
    with pytest.raises(DownloadError):
        check_magic(tmp, tmp_path / "file.tsv.gz")


def test_gmt_loading(fixture_files):
    sets = enrichment.load_gmt(fixture_files["reactome_gmt"])
    assert any("R-HSA-0000001" in k for k in sets)


def test_ranking_keeps_one_value_per_gene():
    t = pd.DataFrame({"gene": ["A", "A", "B"], "t": [1.0, -3.0, 2.0]})
    assert enrichment.ranking_from_table(t, "t", "gene").to_dict() == {"B": 2.0, "A": -3.0}


def test_string_network_uses_cache_offline(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"preferredName_A": "A", "preferredName_B": "B", "score": 0.9}]

    monkeypatch.setattr(network.requests, "post", lambda *a, **k: calls.append(1) or Resp())
    e1 = network.fetch_string_network(["A", "B"], tmp_path)
    e2 = network.fetch_string_network(["B", "A"], tmp_path, offline=True)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(e1, e2)
    _, hubs = network.hub_table(e1)
    assert set(hubs["gene"]) == {"A", "B"}
    assert network.fetch_string_network([], tmp_path).empty
