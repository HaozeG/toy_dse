from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hw.layout import Layout


def cycles_to_us(cycles: float, clock_ghz: float) -> float:
    return round(cycles / (clock_ghz * 1000.0), 9)


def write_trace(
    path: Path,
    layout: Layout,
    slices: list[dict],
    counters: list[dict],
    metadata: dict[str, Any],
    clock_ghz: float,
) -> None:
    events: list[dict] = []
    seen_pid: set[int] = set()
    for track in layout.tracks:
        if track.pid not in seen_pid:
            proc_name = "system" if track.pid == 0 else next(
                (c["id"] for c in layout.cores if 1 + c["row"] * layout.cols + c["col"] == track.pid),
                f"pid{track.pid}",
            )
            events.append(
                {
                    "args": {"name": proc_name},
                    "cat": "__metadata",
                    "name": "process_name",
                    "ph": "M",
                    "pid": track.pid,
                    "tid": 0,
                    "ts": 0,
                }
            )
            seen_pid.add(track.pid)
        events.append(
            {
                "args": {"name": track.id},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": track.pid,
                "tid": track.tid,
                "ts": 0,
            }
        )

    for sl in slices:
        track = layout.by_id.get(sl["track_id"])
        if track is None:
            raise KeyError(f"slice track {sl['track_id']} is not a layout ID")
        events.append(
            {
                "args": sl.get("args") or {},
                "cat": sl.get("cat", "sim"),
                "dur": cycles_to_us(sl["t1"] - sl["t0"], clock_ghz),
                "name": sl["name"],
                "ph": "X",
                "pid": track.pid,
                "tid": track.tid,
                "ts": cycles_to_us(sl["t0"], clock_ghz),
            }
        )

    for ctr in counters:
        track = layout.by_id.get(ctr["track_id"])
        if track is None:
            raise KeyError(f"counter track {ctr['track_id']} is not a layout ID")
        events.append(
            {
                "args": {ctr["name"]: ctr["value"]},
                "cat": "counter",
                "name": ctr["name"],
                "ph": "C",
                "pid": track.pid,
                "tid": track.tid,
                "ts": cycles_to_us(ctr["ts"], clock_ghz),
            }
        )

    events.sort(key=lambda e: (e["ts"], e["pid"], e["tid"], e["name"], e["ph"]))
    payload = {
        "displayTimeUnit": "ns",
        "otherData": metadata,
        "traceEvents": events,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
