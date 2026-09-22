from __future__ import annotations

import pytest

from cardioomics.fixtures import write_fixtures


@pytest.fixture(scope="session")
def fixture_files(tmp_path_factory):
    raw = tmp_path_factory.mktemp("data") / "raw"
    return write_fixtures(raw)
