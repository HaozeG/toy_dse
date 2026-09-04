from __future__ import annotations

import sys
from pathlib import Path

# Console scripts put venv/bin first on sys.path; the stdlib `trace` module
# would otherwise shadow this repo's trace/ package.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
_trace = sys.modules.get("trace")
if _trace is not None and not hasattr(_trace, "__path__"):
    del sys.modules["trace"]

import json
import time
from typing import Any, Optional

import typer

from hw.schema import dump_hw_schema, load_hw
from sim.engine import run_schedule
from sim.resources import SimError
from sw.roofline import RooflineRow, roofline_table

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, add_completion=False)
hw_app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, add_completion=False)
app.add_typer(hw_app, name="hw")

ROOT = _ROOT
TINY_SCHEDULE = ROOT / "tests" / "fixtures" / "tiny_read_compute_write.json"
HW_SCHEMA_PATH = ROOT / "docs" / "hw.schema.json"


def _die(msg: str, code: int = 1) -> None:
    typer.echo(msg, err=True)
    raise typer.Exit(code)


def _emit(json_out: bool, payload: Any, text: str) -> None:
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(text)


def _explain_text(hw, derived) -> str:
    lines = [
        f"name: {hw.name}",
        f"grid: {hw.grid.rows}x{hw.grid.cols} ({derived.n_cores} cores) @ {hw.clock_ghz} GHz",
        f"peak_macs_per_cycle: {derived.peak_macs_per_cycle}",
        f"peak_int8_tops: {derived.peak_int8_tops:.4f}  (2 ops/MAC)",
        f"dram_bytes_per_cycle: {derived.dram_bytes_per_cycle}  ({derived.dram_gbps:.3f} GB/s)",
        f"bisection_bytes_per_cycle: {derived.bisection_bytes_per_cycle}  (vertical cut, all NoCs)",
        f"balance_bytes_per_mac: {derived.balance_bytes_per_mac:.6g}",
        f"dram_positions: {derived.dram_positions}",
    ]
    return "\n".join(lines)


def _roofline_text(rows: list[RooflineRow]) -> str:
    header = (
        f"{'K':>6} {'B':>4} {'MACs':>14} {'weight_B':>12} {'kv_B':>12} {'act_B':>12} "
        f"{'t_cmp_us':>12} {'t_mem_us':>12} {'bound':>8} {'mem/cmp':>10}"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.kv_len:6d} {r.batch:4d} {r.macs:14d} {r.weight_bytes:12d} {r.kv_bytes:12d} "
            f"{r.activation_bytes:12d} {r.t_compute_s * 1e6:12.3f} {r.t_memory_s * 1e6:12.3f} "
            f"{r.bound:>8} {r.mem_over_compute:10.1f}"
        )
    return "\n".join(lines)


@hw_app.command("explain")
def hw_explain(
    preset: str = typer.Argument("grid8x8"),
    set_: list[str] = typer.Option([], "--set"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    try:
        hw = load_hw(preset, set_ or None)
        derived = hw.derived()
    except Exception as exc:
        _die(str(exc))
    payload = {"hw": json.loads(hw.model_dump_json()), "derived": json.loads(derived.model_dump_json())}
    _emit(json_out, payload, _explain_text(hw, derived))


@hw_app.command("schema")
def hw_schema(
    out: Path = typer.Option(HW_SCHEMA_PATH, "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    path = dump_hw_schema(out)
    payload = {"path": str(path)}
    _emit(json_out, payload, f"wrote {path}")


@app.command("roofline")
def roofline_cmd(
    hw: str = typer.Option("grid8x8", "--hw"),
    set_: list[str] = typer.Option([], "--set"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    try:
        cfg = load_hw(hw, set_ or None)
        rows = roofline_table(cfg)
    except Exception as exc:
        _die(str(exc))
    payload = {
        "hw": cfg.name,
        "derived": json.loads(cfg.derived().model_dump_json()),
        "rows": [r.as_dict() for r in rows],
    }
    _emit(json_out, payload, _roofline_text(rows))


@app.command("sim")
def sim_cmd(
    hw: str = typer.Option("tiny", "--hw"),
    schedule: Path = typer.Option(TINY_SCHEDULE, "--schedule"),
    set_: list[str] = typer.Option([], "--set"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    out: Optional[Path] = typer.Option(None, "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    try:
        cfg = load_hw(hw, set_ or None)
        result = run_schedule(cfg, schedule, run_dir=out, run_id=run_id)
    except SimError as exc:
        _die(f"simulation error: {exc}")
    except Exception as exc:
        _die(str(exc))
    payload = {
        "run_dir": str(result.run_dir),
        "cycles": result.cycles,
        "roofline_ratio": result.roofline_ratio,
        "bytes_moved": result.bytes_moved,
        "macs": result.macs,
    }
    text = (
        f"run_dir: {result.run_dir}\n"
        f"cycles: {result.cycles}\n"
        f"roofline_ratio: {result.roofline_ratio:.4f}\n"
        f"bytes_moved: {result.bytes_moved}\n"
        f"macs: {result.macs}"
    )
    _emit(json_out, payload, text)


def _thread_names(trace_path: Path) -> list[str]:
    from perfetto.trace_processor import TraceProcessor

    tp = TraceProcessor(trace=str(trace_path))
    try:
        names: set[str] = set()
        for query in (
            "SELECT name FROM thread WHERE name IS NOT NULL AND name != ''",
            "SELECT name FROM track WHERE name IS NOT NULL AND name != ''",
        ):
            try:
                for row in tp.query(query):
                    names.add(row.name)
            except Exception:
                continue
        return sorted(names)
    finally:
        tp.close()


@app.command("doctor")
def doctor_cmd(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    t0 = time.perf_counter()
    checks: list[dict] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "ok": True, "detail": detail})
        except Exception as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})
            raise

    try:
        def deps():
            import numpy  # noqa: F401
            import pydantic  # noqa: F401
            import simpy  # noqa: F401
            import transformers  # noqa: F401
            import yaml  # noqa: F401
            from perfetto.trace_processor import TraceProcessor  # noqa: F401

            return "simpy pydantic numpy perfetto transformers typer pyyaml"

        check("deps", deps)

        hw = load_hw("tiny")
        derived = hw.derived()
        check("tiny_preset", lambda: f"{hw.grid.rows}x{hw.grid.cols} dram={hw.dram.channels}")

        dump_hw_schema(HW_SCHEMA_PATH)
        check("hw_schema", lambda: str(HW_SCHEMA_PATH))

        result = run_schedule(hw, TINY_SCHEDULE, run_id="doctor")
        check("tiny_sim", lambda: f"cycles={result.cycles} ratio={result.roofline_ratio:.4f}")

        names = _thread_names(result.run_dir / "trace.json")
        layout_ids = result.layout.ids()
        missing = sorted(layout_ids - set(names))
        extra = sorted(
            n
            for n in names
            if n in layout_ids or n.startswith("core(") or n.startswith("noc") or n.startswith("dram.")
        )
        extra_bad = sorted(set(extra) - layout_ids)
        if missing:
            raise AssertionError(f"layout IDs missing from trace tracks: {missing[:8]}")
        if extra_bad:
            raise AssertionError(f"trace tracks not in layout: {extra_bad[:8]}")
        check("track_names", lambda: f"{len(layout_ids)} ids matched")

        elapsed = time.perf_counter() - t0
        payload = {
            "ok": True,
            "elapsed_s": elapsed,
            "run_dir": str(result.run_dir),
            "cycles": result.cycles,
            "roofline_ratio": result.roofline_ratio,
            "derived": json.loads(derived.model_dump_json()),
            "checks": checks,
        }
        text = (
            f"doctor ok in {elapsed:.2f}s\n"
            f"tiny cycles={result.cycles} roofline_ratio={result.roofline_ratio:.4f}\n"
            f"peak_int8_tops={derived.peak_int8_tops:.4f} dram_gbps={derived.dram_gbps:.3f}"
        )
        _emit(json_out, payload, text)
        if elapsed >= 10:
            _die(f"doctor took {elapsed:.2f}s (>= 10s)", 1)
    except typer.Exit:
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        if json_out:
            typer.echo(json.dumps({"ok": False, "elapsed_s": elapsed, "error": str(exc), "checks": checks}, indent=2))
        else:
            typer.echo(f"doctor failed: {exc}", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
