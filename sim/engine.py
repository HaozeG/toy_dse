from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_trace = sys.modules.get("trace")
if _trace is not None and not hasattr(_trace, "__path__"):
    del sys.modules["trace"]

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import simpy

from hw.layout import Layout, build_layout
from hw.schema import HwConfig
from sim.core import Core, TraceSink
from sim.dram import DramSystem
from sim.noc import MeshNoc
from sim.resources import BandwidthResource, CircularBuffer, SemaphoreBank, SimError, Sram
from trace.writer import write_trace

CORE_CB_RE = re.compile(r"^core\((\d+),(\d+)\)\.(.+)$")
CORE_ID_RE = re.compile(r"^core\((\d+),(\d+)\)$")


class Collector(TraceSink):
    def __init__(self) -> None:
        self.slices: list[dict] = []
        self.counters: list[dict] = []

    def slice(self, track_id: str, name: str, t0: float, t1: float, cat: str, args: dict | None = None) -> None:
        if t1 - t0 <= 1e-15:
            return
        self.slices.append(
            {
                "track_id": track_id,
                "name": name,
                "t0": t0,
                "t1": t1,
                "cat": cat,
                "args": args or {},
            }
        )

    def counter(self, track_id: str, name: str, ts: float, value: float) -> None:
        self.counters.append({"track_id": track_id, "name": name, "ts": ts, "value": value})


@dataclass
class SimResult:
    run_dir: Path
    cycles: float
    layout: Layout
    roofline_ratio: float
    bytes_moved: int
    macs: int
    deadlock: bool = False
    deadlock_dump: dict | None = None


def load_schedule(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schedule_version") != "0.1":
        raise SimError(f"unsupported schedule_version {data.get('schedule_version')}")
    return data


def parse_cb_key(key: str) -> tuple[int, int, str]:
    m = CORE_CB_RE.match(key)
    if not m:
        raise SimError(f"CB key must look like core(r,c).name, got {key!r}")
    name = m.group(3)
    if name.startswith("cb."):
        name = name[3:]
    return int(m.group(1)), int(m.group(2)), name


def cb_names_by_core(schedule: dict) -> dict[tuple[int, int], list[str]]:
    out: dict[tuple[int, int], list[str]] = {}
    for key in schedule.get("cbs", {}):
        r, c, name = parse_cb_key(key)
        out.setdefault((r, c), []).append(name)
    for names in out.values():
        names.sort()
    return out


def _apply_sram_plan(hw: HwConfig, schedule: dict) -> dict[tuple[int, int], Sram]:
    srams: dict[tuple[int, int], Sram] = {}
    cbs = schedule.get("cbs", {})
    for core_id, mapping in (schedule.get("sram_plan") or {}).items():
        m = CORE_ID_RE.match(core_id)
        if not m:
            raise SimError(f"bad sram_plan core id {core_id}")
        r, c = int(m.group(1)), int(m.group(2))
        sram = Sram(hw.core.sram_bytes, core_id)
        for cb_name, addr in mapping.items():
            full = f"core({r},{c}).{cb_name}"
            spec = cbs.get(full) or cbs.get(f"core({r},{c}).cb.{cb_name}")
            if spec is None:
                raise SimError(f"sram_plan {core_id}.{cb_name} has no CB spec")
            size = int(spec["slots"]) * int(spec["slot_bytes"])
            sram.allocate(cb_name, int(addr), size)
        srams[(r, c)] = sram
    return srams


def _count_schedule_work(schedule: dict) -> tuple[int, int]:
    macs = 0
    bytes_moved = 0
    for core_sched in schedule.get("cores", {}).values():
        for stream in core_sched.values():
            for cmd in stream:
                op = cmd.get("op")
                if op == "COMPUTE":
                    macs += int((cmd.get("cost") or {}).get("macs", 0))
                if op in ("NOC_READ", "NOC_WRITE", "NOC_MULTICAST"):
                    bytes_moved += int(cmd.get("bytes", 0))
    return macs, bytes_moved


def deadlock_dump(cores: list[Core]) -> dict:
    dump: dict[str, Any] = {}
    for core in cores:
        entry = {}
        for name, state in core.states.items():
            entry[name] = {
                "pc": state.pc,
                "blocked_on": state.blocked_on,
                "done": state.done,
                "n_cmds": len(state.commands),
            }
        dump[core.prefix] = entry
    return dump


def run_schedule(
    hw: HwConfig,
    schedule: dict[str, Any] | Path,
    run_dir: Path | None = None,
    run_id: str | None = None,
) -> SimResult:
    if isinstance(schedule, Path):
        schedule_path = schedule
        schedule = load_schedule(schedule)
    else:
        schedule_path = None

    cb_map = cb_names_by_core(schedule)
    layout = build_layout(hw, cb_map)
    _apply_sram_plan(hw, schedule)

    env = simpy.Environment()
    tracer = Collector()
    noc = MeshNoc(env, hw)
    dram = DramSystem(env, hw)
    sems = SemaphoreBank(env, hw.sync.semaphore_latency_cycles)

    cores: list[Core] = []
    injections: dict[tuple[int, int], BandwidthResource] = {}
    cbs_by_core: dict[tuple[int, int], dict[str, CircularBuffer]] = {}

    for key, spec in schedule.get("cbs", {}).items():
        r, c, name = parse_cb_key(key)
        cb = CircularBuffer(env, int(spec["slots"]), int(spec["slot_bytes"]), name)
        cbs_by_core.setdefault((r, c), {})[name] = cb
        cbs_by_core[(r, c)][f"core({r},{c}).{name}"] = cb

    for r in range(hw.grid.rows):
        for c in range(hw.grid.cols):
            injections[(r, c)] = BandwidthResource(
                env, hw.core.noc_read_bytes_per_cycle, f"core({r},{c}).inject"
            )

    core_schedules = schedule.get("cores", {})
    for r in range(hw.grid.rows):
        for c in range(hw.grid.cols):
            key = f"{r},{c}"
            streams = core_schedules.get(key) or core_schedules.get(f"core({r},{c})") or {}
            core = Core(
                env,
                hw,
                layout,
                r,
                c,
                noc,
                dram,
                sems,
                tracer,
                cbs_by_core.get((r, c), {}),
                injections[(r, c)],
                streams if isinstance(streams, dict) else {},
            )
            core.start()
            cores.append(core)

    env.run()

    pending = [c for c in cores if not all(s.done for s in c.states.values())]
    dump = deadlock_dump(cores)
    if pending:
        raise SimError(f"deadlock: no further events with pending commands: {json.dumps(dump)}")

    cycles = float(env.now)
    macs, bytes_moved = _count_schedule_work(schedule)
    derived = hw.derived()
    t_compute = macs / derived.peak_macs_per_cycle if derived.peak_macs_per_cycle else 0.0
    t_memory = bytes_moved / derived.dram_bytes_per_cycle if derived.dram_bytes_per_cycle else 0.0
    analytical = max(t_compute, t_memory, 1e-12)
    ratio = cycles / analytical
    if ratio < 1.0 - 1e-9:
        raise SimError(f"roofline ratio {ratio} < 1 (cycles={cycles}, bound={analytical})")

    if run_dir is None:
        rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = Path("runs") / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    hw_json = json.loads(hw.model_dump_json())
    derived_json = json.loads(derived.model_dump_json())
    (run_dir / "hw.json").write_text(json.dumps({"hw": hw_json, "derived": derived_json}, indent=2, sort_keys=True) + "\n")
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    (run_dir / "layout.json").write_text(json.dumps(layout.to_json(), indent=2, sort_keys=True) + "\n")

    metadata = {
        "hw": hw_json,
        "derived": derived_json,
        "mapper": {},
        "workload": {},
        "cycles": cycles,
        "roofline_ratio": ratio,
        "bytes_moved": bytes_moved,
        "macs": macs,
    }
    write_trace(
        run_dir / "trace.json",
        layout,
        tracer.slices,
        tracer.counters,
        metadata,
        clock_ghz=hw.clock_ghz,
    )
    report = {
        "cycles": cycles,
        "roofline_ratio": ratio,
        "bytes_moved": bytes_moved,
        "macs": macs,
        "hw_name": hw.name,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(
        f"# run {run_dir.name}\n\n- cycles: {cycles}\n- roofline_ratio: {ratio:.4f}\n- bytes_moved: {bytes_moved}\n- macs: {macs}\n"
    )
    if schedule_path is not None:
        pass
    return SimResult(
        run_dir=run_dir,
        cycles=cycles,
        layout=layout,
        roofline_ratio=ratio,
        bytes_moved=bytes_moved,
        macs=macs,
    )
