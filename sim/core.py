from __future__ import annotations

import math
from typing import Any, Callable

import simpy

from hw.layout import Layout
from hw.schema import HwConfig
from sim.dram import DramSystem
from sim.noc import MeshNoc, xy_hops
from sim.resources import (
    BandwidthResource,
    CircularBuffer,
    SemaphoreBank,
    SimError,
    StreamState,
    transfer_on,
)


class TraceSink:
    def slice(self, track_id: str, name: str, t0: float, t1: float, cat: str, args: dict | None = None) -> None:
        raise NotImplementedError

    def counter(self, track_id: str, name: str, ts: float, value: float) -> None:
        raise NotImplementedError


def parse_core_key(key: str) -> tuple[int, int]:
    parts = key.replace(" ", "").split(",")
    if len(parts) != 2:
        raise SimError(f"bad core key {key!r}")
    return int(parts[0]), int(parts[1])


class Core:
    def __init__(
        self,
        env: simpy.Environment,
        hw: HwConfig,
        layout: Layout,
        row: int,
        col: int,
        noc: MeshNoc,
        dram: DramSystem,
        sems: SemaphoreBank,
        tracer: TraceSink,
        cbs: dict[str, CircularBuffer],
        injection: BandwidthResource,
        streams: dict[str, list[dict[str, Any]]],
    ):
        self.env = env
        self.hw = hw
        self.layout = layout
        self.row = row
        self.col = col
        self.pos = (row, col)
        self.noc = noc
        self.dram = dram
        self.sems = sems
        self.tracer = tracer
        self.cbs = cbs
        self.injection = injection
        self.prefix = f"core({row},{col})"
        self.matrix = simpy.Resource(env, capacity=1)
        self.vector = simpy.Resource(env, capacity=1)
        self.dm = simpy.Resource(env, capacity=hw.core.dm_engines)
        self.states = {
            "reader": StreamState("reader", commands=list(streams.get("reader") or [])),
            "compute": StreamState("compute", commands=list(streams.get("compute") or [])),
            "writer": StreamState("writer", commands=list(streams.get("writer") or [])),
        }
        self.dm0 = f"{self.prefix}.dm0"
        self.dm1 = f"{self.prefix}.dm1" if hw.core.dm_engines > 1 else self.dm0
        self.matrix_id = f"{self.prefix}.matrix"
        self.vector_id = f"{self.prefix}.vector"

    def start(self) -> None:
        self.env.process(self._run_stream("reader", self.dm0))
        self.env.process(self._run_stream("compute", self.matrix_id))
        self.env.process(self._run_stream("writer", self.dm1))

    def _cb(self, name: str) -> CircularBuffer:
        if name in self.cbs:
            return self.cbs[name]
        key = f"{self.prefix}.{name}"
        if key in self.cbs:
            return self.cbs[key]
        raise SimError(f"{self.prefix}: unknown CB {name}")

    def _wait_slice(self, track: str, reason: str, t0: float, args: dict | None = None) -> None:
        if self.env.now > t0:
            self.tracer.slice(track, reason, t0, self.env.now, "stall", args)

    def _run_stream(self, stream_name: str, default_track: str):
        state = self.states[stream_name]
        cmds = state.commands
        try:
            for i, cmd in enumerate(cmds):
                state.pc = i
                op = cmd["op"]
                handler = self._handlers().get(op)
                if handler is None:
                    raise SimError(f"{self.prefix}.{stream_name}: unknown op {op}")
                result = handler(state, cmd, default_track)
                if result is not None:
                    yield from result
            state.done = True
            state.blocked_on = None
            state.pc = len(cmds)
        except SimError:
            state.blocked_on = state.blocked_on or "error"
            raise

    def _handlers(self) -> dict[str, Callable]:
        return {
            "CB_RESERVE": self._cb_reserve,
            "CB_PUSH": self._cb_push,
            "CB_WAIT": self._cb_wait,
            "CB_POP": self._cb_pop,
            "NOC_READ": self._noc_read,
            "NOC_WRITE": self._noc_write,
            "NOC_MULTICAST": self._noc_multicast,
            "COMPUTE": self._compute,
            "SEM_INC": self._sem_inc,
            "SEM_WAIT": self._sem_wait,
        }

    def _cb_reserve(self, state: StreamState, cmd: dict, track: str):
        cb = self._cb(cmd["cb"])
        n = int(cmd.get("count", 1))
        state.blocked_on = "wait_cb_full"
        t0 = self.env.now
        yield from cb.reserve(n)
        self._wait_slice(track, "wait_cb_full", t0, {"cb": cb.name})
        state.blocked_on = None
        self._cb_counter(cb)

    def _cb_push(self, state: StreamState, cmd: dict, track: str):
        cb = self._cb(cmd["cb"])
        cb.push(int(cmd.get("count", 1)))
        self._cb_counter(cb)

    def _cb_wait(self, state: StreamState, cmd: dict, track: str):
        cb = self._cb(cmd["cb"])
        n = int(cmd.get("count", 1))
        state.blocked_on = "wait_cb_empty"
        t0 = self.env.now
        yield from cb.wait(n)
        self._wait_slice(track, "wait_cb_empty", t0, {"cb": cb.name})
        state.blocked_on = None
        self._cb_counter(cb)

    def _cb_pop(self, state: StreamState, cmd: dict, track: str):
        cb = self._cb(cmd["cb"])
        cb.pop(int(cmd.get("count", 1)))
        self._cb_counter(cb)

    def _cb_counter(self, cb: CircularBuffer) -> None:
        track = f"{self.prefix}.cb.{cb.name.split('.')[-1]}"
        if track in self.layout.by_id:
            self.tracer.counter(track, "occupancy", self.env.now, cb.filled)
        sram = f"{self.prefix}.sram"
        if sram in self.layout.by_id:
            used = sum(c.filled * c.slot_bytes for c in self.cbs.values())
            self.tracer.counter(sram, "bytes_in_use", self.env.now, used)

    def _acquire_dm(self, state: StreamState, track: str):
        state.blocked_on = "wait_noc"
        t0 = self.env.now
        req = self.dm.request()
        yield req
        self._wait_slice(track, "wait_noc", t0, {"reason": "dm_engine"})
        state.blocked_on = None
        return req

    def _endpoint(self, token: str) -> tuple[str, tuple[int, int], BandwidthResource | None]:
        if token.startswith("dram."):
            ch = self.dram.lookup(token)
            return "dram", ch.position, ch.bw
        if token.startswith("core("):
            inner = token[len("core(") :].split(")", 1)[0]
            r, c = parse_core_key(inner)
            return "core", (r, c), None
        raise SimError(f"bad endpoint {token}")

    def _noc_move(
        self,
        state: StreamState,
        track: str,
        src_token: str,
        dst_token: str,
        nbytes: int,
        noc_id: int,
        op_name: str,
        args: dict,
    ):
        kind_s, pos_s, bw_s = self._endpoint(src_token)
        kind_d, pos_d, bw_d = self._endpoint(dst_token)
        dm_req = yield from self._acquire_dm(state, track)
        try:
            hops = xy_hops(pos_s, pos_d)
            resources: list[BandwidthResource] = [self.injection]
            if bw_s is not None:
                resources.append(bw_s)
            if bw_d is not None and bw_d is not bw_s:
                resources.append(bw_d)
            resources.extend(self.noc.path_resources(noc_id, pos_s, pos_d))
            t_lat0 = self.env.now
            extra_lat = 0
            if kind_s == "dram":
                extra_lat += self.dram.lookup(src_token).latency_cycles
            if kind_d == "dram":
                extra_lat += self.dram.lookup(dst_token).latency_cycles
            extra_lat += self.noc.hop_latency * len(hops)
            if extra_lat:
                state.blocked_on = "wait_dram" if "dram" in (kind_s, kind_d) else "wait_noc"
                yield self.env.timeout(extra_lat)
                reason = "wait_dram" if "dram" in (kind_s, kind_d) else "wait_noc"
                self._wait_slice(track, reason, t_lat0, {"hops": len(hops)})
                state.blocked_on = None
            t_pay0 = self.env.now
            state.blocked_on = "wait_noc"
            yield from transfer_on(self.env, resources, nbytes)
            t_pay1 = self.env.now
            state.blocked_on = None
            payload_cycles = max(t_pay1 - t_pay0, 1e-12)
            achieved = nbytes / payload_cycles
            path_ids = self.noc.path_ids(noc_id, pos_s, pos_d)
            slice_args = {
                **args,
                "bytes": nbytes,
                "achieved_bytes_per_cycle": achieved,
                "noc_path": path_ids,
            }
            self.tracer.slice(track, op_name, t_pay0, t_pay1, "noc", slice_args)
            for lid in path_ids:
                self.tracer.slice(lid, op_name, t_pay0, t_pay1, "noc", slice_args)
            for token, bw in ((src_token, bw_s), (dst_token, bw_d)):
                if bw is not None:
                    self.tracer.slice(token, op_name, t_pay0, t_pay1, "dram", slice_args)
        finally:
            self.dm.release(dm_req)

    def _noc_read(self, state: StreamState, cmd: dict, track: str):
        dst_core = self.prefix
        yield from self._noc_move(
            state,
            track,
            cmd["src"],
            dst_core,
            int(cmd["bytes"]),
            int(cmd.get("noc", 0)),
            "NOC_READ",
            {"task_id": cmd["id"], "dst_cb": cmd.get("dst_cb"), "offset": cmd.get("offset", 0)},
        )

    def _noc_write(self, state: StreamState, cmd: dict, track: str):
        yield from self._noc_move(
            state,
            track,
            self.prefix,
            cmd["dst"],
            int(cmd["bytes"]),
            int(cmd.get("noc", 1)),
            "NOC_WRITE",
            {"task_id": cmd["id"], "src_cb": cmd.get("src_cb"), "offset": cmd.get("offset", 0)},
        )

    def _noc_multicast(self, state: StreamState, cmd: dict, track: str):
        # Phase 1: one destination; tree occupancy is Phase 3.
        dsts = cmd.get("dsts") or [cmd["dst"]]
        if len(dsts) != 1:
            raise SimError("NOC_MULTICAST tree occupancy is Phase 3; one destination required")
        cmd = {**cmd, "dst": dsts[0]}
        yield from self._noc_write(state, cmd, track)

    def _compute(self, state: StreamState, cmd: dict, track: str):
        cost = cmd.get("cost") or {}
        macs = int(cost.get("macs", 0))
        vector_ops = int(cost.get("vector_ops", 0))
        macs_per = self.hw.core.matrix_unit.macs_per_cycle * self.hw.core.matrix_unit.count
        vec_per = (
            self.hw.core.vector_unit.lanes
            * self.hw.core.vector_unit.ops_per_cycle
            * self.hw.core.vector_unit.count
        )
        mat_cycles = math.ceil(macs / macs_per) if macs else 0
        vec_cycles = math.ceil(vector_ops / vec_per) if vector_ops else 0
        args = {
            "task_id": cmd["id"],
            "op": cmd.get("kernel", "compute"),
            "macs": macs,
            "vector_ops": vector_ops,
        }
        if mat_cycles:
            state.blocked_on = "wait_unit"
            t0 = self.env.now
            req = self.matrix.request()
            yield req
            self._wait_slice(self.matrix_id, "wait_unit", t0, {"unit": "matrix"})
            t1 = self.env.now
            yield self.env.timeout(mat_cycles)
            t2 = self.env.now
            self.matrix.release(req)
            state.blocked_on = None
            self.tracer.slice(self.matrix_id, cmd.get("kernel", "COMPUTE"), t1, t2, "compute", args)
        if vec_cycles:
            state.blocked_on = "wait_unit"
            t0 = self.env.now
            req = self.vector.request()
            yield req
            self._wait_slice(self.vector_id, "wait_unit", t0, {"unit": "vector"})
            t1 = self.env.now
            yield self.env.timeout(vec_cycles)
            t2 = self.env.now
            self.vector.release(req)
            state.blocked_on = None
            self.tracer.slice(self.vector_id, cmd.get("kernel", "COMPUTE"), t1, t2, "compute", args)

    def _sem_inc(self, state: StreamState, cmd: dict, track: str):
        state.blocked_on = "wait_sem"
        yield from self.sems.inc(int(cmd["sem"]), int(cmd.get("count", 1)))
        state.blocked_on = None

    def _sem_wait(self, state: StreamState, cmd: dict, track: str):
        state.blocked_on = "wait_sem"
        t0 = self.env.now
        yield from self.sems.wait(int(cmd["sem"]), int(cmd.get("count", 1)))
        self._wait_slice(track, "wait_sem", t0, {"sem": cmd["sem"]})
        state.blocked_on = None
