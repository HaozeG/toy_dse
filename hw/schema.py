from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PRESETS_DIR = Path(__file__).parent / "presets"


class GridConfig(BaseModel):
    rows: int = Field(default=8, ge=1, description="Mesh rows.")
    cols: int = Field(default=8, ge=1, description="Mesh columns.")


class MatrixUnitConfig(BaseModel):
    macs_per_cycle: int = Field(default=1024, ge=1, description="MAC throughput of one matrix unit.")
    tile_shape: tuple[int, int, int] = Field(
        default=(32, 32, 32),
        description="Native [M, N, K] tile shape the matrix unit consumes in one burst.",
    )
    count: int = Field(default=1, ge=1, description="Matrix units per core.")


class VectorUnitConfig(BaseModel):
    lanes: int = Field(default=32, ge=1, description="Vector lanes.")
    elem_bytes: int = Field(default=4, ge=1, description="Element width in bytes (fp32 accumulate = 4).")
    ops_per_cycle: int = Field(default=1, ge=1, description="Ops per lane per cycle.")
    count: int = Field(default=1, ge=1, description="Vector units per core.")


class CoreConfig(BaseModel):
    sram_bytes: int = Field(default=1_572_864, ge=1, description="Local SRAM capacity per core in bytes.")
    matrix_unit: MatrixUnitConfig = Field(default_factory=MatrixUnitConfig, description="Systolic / matrix unit.")
    vector_unit: VectorUnitConfig = Field(default_factory=VectorUnitConfig, description="Vector ALU.")
    dm_engines: int = Field(default=2, ge=1, description="Concurrent outstanding NoC transactions per core.")
    noc_read_bytes_per_cycle: int = Field(
        default=32,
        ge=1,
        description="Per-core injection/ejection limit in bytes/cycle.",
    )


class NocConfig(BaseModel):
    count: int = Field(default=2, ge=1, description="Independent NoCs. Mapper assigns reads to noc0, writes to noc1.")
    link_bytes_per_cycle: int = Field(default=32, ge=1, description="Per-link payload bytes/cycle (each direction).")
    hop_latency_cycles: int = Field(default=2, ge=0, description="Fixed cycles added per hop.")
    multicast: bool = Field(default=True, description="Hardware multicast occupies the tree once.")


class DramConfig(BaseModel):
    channels: int = Field(default=8, ge=1, description="Number of DRAM channels.")
    bytes_per_cycle_per_channel: int = Field(default=12, ge=1, description="Peak payload bytes/cycle per channel.")
    latency_cycles: int = Field(default=300, ge=0, description="Uncontended DRAM access latency in cycles.")
    positions: str | list[list[int]] = Field(
        default="edge_left_right",
        description="edge_left_right or explicit [[row, col], ...] mesh-edge attachments.",
    )


class SyncConfig(BaseModel):
    semaphore_latency_cycles: int = Field(default=8, ge=0, description="Cycles for SEM_INC visibility.")


class DerivedPeaks(BaseModel):
    n_cores: int
    peak_macs_per_cycle: int
    peak_int8_tops: float
    dram_bytes_per_cycle: int
    dram_gbps: float
    bisection_bytes_per_cycle: int
    balance_bytes_per_mac: float
    clock_hz: float
    dram_positions: list[list[int]]


class HwConfig(BaseModel):
    name: str = Field(default="grid8x8", description="Preset or resolved configuration name.")
    clock_ghz: float = Field(default=1.0, gt=0, description="Clock in GHz. Simulator time unit is one cycle.")
    grid: GridConfig = Field(default_factory=GridConfig, description="Core mesh.")
    core: CoreConfig = Field(default_factory=CoreConfig, description="Per-core resources.")
    noc: NocConfig = Field(default_factory=NocConfig, description="Mesh NoC.")
    dram: DramConfig = Field(default_factory=DramConfig, description="DRAM channels at mesh edges.")
    sync: SyncConfig = Field(default_factory=SyncConfig, description="Synchronization.")

    @field_validator("clock_ghz")
    @classmethod
    def _clock_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("clock_ghz must be positive")
        return v

    @model_validator(mode="after")
    def _positions_match_channels(self) -> HwConfig:
        pos = self.resolved_dram_positions()
        if len(pos) != self.dram.channels:
            raise ValueError(
                f"DRAM positions ({len(pos)}) must equal dram.channels ({self.dram.channels})"
            )
        for r, c in pos:
            if not (0 <= r < self.grid.rows and 0 <= c < self.grid.cols):
                raise ValueError(f"DRAM position {(r, c)} is outside the {self.grid.rows}x{self.grid.cols} mesh")
        return self

    def resolved_dram_positions(self) -> list[list[int]]:
        spec = self.dram.positions
        if isinstance(spec, list):
            return [[int(r), int(c)] for r, c in spec]
        if spec == "edge_left_right":
            return expand_edge_left_right(self.grid.rows, self.grid.cols, self.dram.channels)
        raise ValueError(f"unknown dram.positions {spec!r}")

    def derived(self) -> DerivedPeaks:
        n_cores = self.grid.rows * self.grid.cols
        peak_macs_per_cycle = n_cores * self.core.matrix_unit.macs_per_cycle * self.core.matrix_unit.count
        peak_int8_tops = peak_macs_per_cycle * self.clock_ghz * 2.0 / 1000.0
        dram_bytes_per_cycle = self.dram.channels * self.dram.bytes_per_cycle_per_channel
        dram_gbps = dram_bytes_per_cycle * self.clock_ghz
        bisection_bytes_per_cycle = self.grid.rows * self.noc.count * self.noc.link_bytes_per_cycle
        balance = dram_bytes_per_cycle / peak_macs_per_cycle if peak_macs_per_cycle else 0.0
        return DerivedPeaks(
            n_cores=n_cores,
            peak_macs_per_cycle=peak_macs_per_cycle,
            peak_int8_tops=peak_int8_tops,
            dram_bytes_per_cycle=dram_bytes_per_cycle,
            dram_gbps=dram_gbps,
            bisection_bytes_per_cycle=bisection_bytes_per_cycle,
            balance_bytes_per_mac=balance,
            clock_hz=self.clock_ghz * 1e9,
            dram_positions=self.resolved_dram_positions(),
        )

    def export_json_schema(self) -> dict[str, Any]:
        return type(self).model_json_schema()


def expand_edge_left_right(rows: int, cols: int, channels: int) -> list[list[int]]:
    """Place channels on the left and right mesh edges, more on the left if odd."""
    left_n = (channels + 1) // 2
    right_n = channels // 2
    pos = [[r, 0] for r in _even_rows(left_n, rows)]
    pos += [[r, cols - 1] for r in _even_rows(right_n, rows)]
    return pos


def _even_rows(n: int, rows: int) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [rows // 2]
    return [int(i * rows / n + rows / n / 2) for i in range(n)]


def _coerce_override(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(data: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set expects path=value, got {item!r}")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        cursor: Any = data
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = _coerce_override(raw)
    return data


def load_hw(name_or_path: str, sets: list[str] | None = None) -> HwConfig:
    path = Path(name_or_path)
    if not path.suffix:
        path = PRESETS_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"hardware config not found: {name_or_path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if sets:
        data = apply_overrides(data, sets)
    return HwConfig.model_validate(data)


def dump_hw_schema(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(HwConfig.model_json_schema(), indent=2) + "\n")
    return dest
