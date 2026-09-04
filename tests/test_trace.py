import json

from hw.schema import load_hw
from sim.engine import run_schedule
from tests.conftest import TINY_SCHEDULE


def test_perfetto_track_names_equal_layout_ids(tmp_path):
    from perfetto.trace_processor import TraceProcessor

    hw = load_hw("tiny")
    result = run_schedule(hw, TINY_SCHEDULE, run_dir=tmp_path / "run")
    trace_path = tmp_path / "run" / "trace.json"
    tp = TraceProcessor(trace=str(trace_path))
    try:
        names = set()
        for row in tp.query("SELECT name FROM thread WHERE name IS NOT NULL AND name != ''"):
            names.add(row.name)
        layout_ids = result.layout.ids()
        missing = layout_ids - names
        assert not missing, f"missing thread names: {sorted(missing)[:12]}"
    finally:
        tp.close()


def test_trace_has_compute_and_noc_slices(tmp_path):
    hw = load_hw("tiny")
    run_schedule(hw, TINY_SCHEDULE, run_dir=tmp_path / "run")
    events = json.loads((tmp_path / "run" / "trace.json").read_text())["traceEvents"]
    names = {e["name"] for e in events if e.get("ph") == "X"}
    assert "NOC_READ" in names
    assert "NOC_WRITE" in names
    assert "matmul" in names
