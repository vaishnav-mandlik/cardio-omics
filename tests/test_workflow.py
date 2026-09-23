import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cardioomics.fixtures import write_fixtures

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_full_workflow_on_fixtures():
    work = ROOT / ".test_work"
    shutil.rmtree(work, ignore_errors=True)
    write_fixtures(work / "data" / "raw")
    cmd = [
        "snakemake", "-s", "workflow/Snakefile", "--configfile", "config/test.yaml",
        "--cores", "2", "--quiet", "rules",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    results = work / "results"
    assert "# CardioOmics report" in (results / "report" / "report.md").read_text()
    summary = json.loads((results / "report" / "summary.json").read_text())
    assert summary["n_de_genes_primary"] > 10
    assert summary["classifier"]["auroc"] > 0.8
