import json
import time

from typer.testing import CliRunner

from cli import app


def test_doctor_cli():
    runner = CliRunner()
    t0 = time.perf_counter()
    result = runner.invoke(app, ["doctor", "--json"])
    elapsed = time.perf_counter() - t0
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["roofline_ratio"] >= 1.0
    assert elapsed < 10


def test_hw_explain_set_json():
    runner = CliRunner()
    result = runner.invoke(app, ["hw", "explain", "grid8x8", "--set", "dram.channels=4", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["hw"]["dram"]["channels"] == 4
    assert payload["derived"]["dram_bytes_per_cycle"] == 48
