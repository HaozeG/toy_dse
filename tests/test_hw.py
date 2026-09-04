import json

from hw.layout import build_layout
from hw.schema import HwConfig, dump_hw_schema, load_hw


def test_grid8x8_derived_peaks():
    hw = load_hw("grid8x8")
    d = hw.derived()
    assert d.n_cores == 64
    assert d.peak_macs_per_cycle == 64 * 1024
    assert abs(d.peak_int8_tops - 131.072) < 1e-6
    assert d.dram_bytes_per_cycle == 96
    assert abs(d.dram_gbps - 96.0) < 1e-9
    assert d.bisection_bytes_per_cycle == 8 * 2 * 32
    assert len(d.dram_positions) == 8


def test_tiny_preset():
    hw = load_hw("tiny")
    assert hw.grid.rows == 2 and hw.grid.cols == 2
    assert hw.dram.channels == 1
    assert hw.resolved_dram_positions() == [[1, 0]]


def test_set_overrides():
    hw = load_hw("grid8x8", ["dram.channels=4", "grid.rows=4"])
    assert hw.dram.channels == 4
    assert hw.grid.rows == 4
    assert len(hw.resolved_dram_positions()) == 4


def test_schema_roundtrip(tmp_path):
    hw = load_hw("tiny")
    data = json.loads(hw.model_dump_json())
    again = HwConfig.model_validate(data)
    assert again.derived() == hw.derived()


def test_layout_ids_unique():
    hw = load_hw("tiny")
    layout = build_layout(hw, {(0, 0): ["in0", "out0"]})
    ids = [t.id for t in layout.tracks]
    assert len(ids) == len(set(ids))
    assert "core(0,0).matrix" in layout.by_id
    assert "core(0,0).cb.in0" in layout.by_id
    assert "dram.ch0" in layout.by_id
    assert "noc0.link(0,0)->(0,1)" in layout.by_id
    assert layout.by_id["dram.ch0"].pid == 0
    assert layout.by_id["core(0,0).matrix"].pid == 1


def test_hw_schema_export(tmp_path):
    path = dump_hw_schema(tmp_path / "hw.schema.json")
    schema = json.loads(path.read_text())
    assert schema["title"] == "HwConfig"
    assert "properties" in schema
