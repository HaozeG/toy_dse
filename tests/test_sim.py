import json

from hw.schema import load_hw
from sim.engine import run_schedule
from tests.conftest import TINY_SCHEDULE


def test_tiny_read_compute_write(tmp_path):
    hw = load_hw("tiny")
    result = run_schedule(hw, TINY_SCHEDULE, run_dir=tmp_path / "run")
    assert result.cycles > 0
    assert result.macs == 32768
    assert result.bytes_moved == 8192
    assert result.roofline_ratio >= 1.0
    assert (tmp_path / "run" / "trace.json").is_file()
    assert (tmp_path / "run" / "schedule.json").is_file()
    assert (tmp_path / "run" / "hw.json").is_file()
    assert (tmp_path / "run" / "layout.json").is_file()


def test_determinism(tmp_path):
    hw = load_hw("tiny")
    a = run_schedule(hw, TINY_SCHEDULE, run_dir=tmp_path / "a")
    b = run_schedule(hw, TINY_SCHEDULE, run_dir=tmp_path / "b")
    ta = (tmp_path / "a" / "trace.json").read_text()
    tb = (tmp_path / "b" / "trace.json").read_text()
    assert ta == tb
    assert a.cycles == b.cycles


def test_trace_track_ids_are_layout_ids(tmp_path):
    hw = load_hw("tiny")
    result = run_schedule(hw, TINY_SCHEDULE, run_dir=tmp_path / "run")
    trace = json.loads((tmp_path / "run" / "trace.json").read_text())
    layout_ids = result.layout.ids()
    thread_names = {
        ev["args"]["name"]
        for ev in trace["traceEvents"]
        if ev.get("ph") == "M" and ev.get("name") == "thread_name"
    }
    assert thread_names == layout_ids
    for ev in trace["traceEvents"]:
        if ev.get("ph") in {"X", "C"}:
            # pid/tid must belong to some layout track
            matches = [
                t
                for t in result.layout.tracks
                if t.pid == ev["pid"] and t.tid == ev["tid"]
            ]
            assert matches, ev
