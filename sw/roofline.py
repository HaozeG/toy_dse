from __future__ import annotations

from dataclasses import dataclass

from hw.schema import HwConfig
from sw.workload.qwen3 import build_decode_ops, load_qwen3_0_6b, summarize_ops

DEFAULT_K = (128, 256, 1024)
DEFAULT_B = (1, 4, 16)


@dataclass(frozen=True)
class RooflineRow:
    kv_len: int
    batch: int
    macs: int
    weight_bytes: int
    kv_bytes: int
    activation_bytes: int
    total_bytes: int
    t_compute_s: float
    t_memory_s: float
    bound: str
    mem_over_compute: float

    def as_dict(self) -> dict:
        return {
            "K": self.kv_len,
            "B": self.batch,
            "macs": self.macs,
            "weight_bytes": self.weight_bytes,
            "kv_bytes": self.kv_bytes,
            "activation_bytes": self.activation_bytes,
            "total_bytes": self.total_bytes,
            "t_compute_s": self.t_compute_s,
            "t_memory_s": self.t_memory_s,
            "bound": self.bound,
            "mem_over_compute": self.mem_over_compute,
        }


def roofline_row(hw: HwConfig, kv_len: int, batch: int) -> RooflineRow:
    cfg = load_qwen3_0_6b()
    ops = build_decode_ops(cfg, kv_len=kv_len, batch=batch)
    totals = summarize_ops(ops)
    derived = hw.derived()
    clock_hz = derived.clock_hz
    peak_macs_s = derived.peak_macs_per_cycle * clock_hz
    peak_bytes_s = derived.dram_bytes_per_cycle * clock_hz
    t_compute = totals["macs"] / peak_macs_s if peak_macs_s else float("inf")
    total_bytes = totals["weight_bytes"] + totals["kv_bytes"] + totals["activation_bytes"]
    t_memory = total_bytes / peak_bytes_s if peak_bytes_s else float("inf")
    bound = "memory" if t_memory >= t_compute else "compute"
    ratio = t_memory / t_compute if t_compute > 0 else float("inf")
    return RooflineRow(
        kv_len=kv_len,
        batch=batch,
        macs=totals["macs"],
        weight_bytes=totals["weight_bytes"],
        kv_bytes=totals["kv_bytes"],
        activation_bytes=totals["activation_bytes"],
        total_bytes=total_bytes,
        t_compute_s=t_compute,
        t_memory_s=t_memory,
        bound=bound,
        mem_over_compute=ratio,
    )


def roofline_table(
    hw: HwConfig,
    kv_lens: tuple[int, ...] = DEFAULT_K,
    batches: tuple[int, ...] = DEFAULT_B,
) -> list[RooflineRow]:
    return [roofline_row(hw, k, b) for k in kv_lens for b in batches]
