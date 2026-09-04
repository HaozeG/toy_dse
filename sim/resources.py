from __future__ import annotations

from dataclasses import dataclass, field

import simpy


class SimError(Exception):
    """Structural assertion failure: CB over/underflow, SRAM overflow, deadlock."""


@dataclass
class Occupancy:
    remaining: float


class BandwidthResource:
    """Fair share among active transfers; recomputed when the active set changes."""

    def __init__(self, env: simpy.Environment, bytes_per_cycle: float, name: str):
        self.env = env
        self.bytes_per_cycle = float(bytes_per_cycle)
        self.name = name
        self._active: list[Occupancy] = []
        self._epoch = 0.0
        self._change: simpy.Event | None = None

    def change_event(self) -> simpy.Event:
        if self._change is None or self._change.triggered:
            self._change = self.env.event()
        return self._change

    def _notify(self) -> None:
        old = self._change
        self._change = self.env.event()
        if old is not None and not old.triggered:
            old.succeed()

    def advance(self) -> None:
        dt = self.env.now - self._epoch
        if dt > 0 and self._active:
            share = self.bytes_per_cycle / len(self._active)
            for occ in self._active:
                occ.remaining = max(0.0, occ.remaining - share * dt)
        self._epoch = self.env.now

    def join(self, occ: Occupancy) -> None:
        self.advance()
        self._active.append(occ)
        self._notify()

    def leave(self, occ: Occupancy) -> None:
        self.advance()
        self._active.remove(occ)
        self._notify()

    def share(self) -> float:
        n = len(self._active)
        if n == 0:
            return 0.0
        return self.bytes_per_cycle / n

    def n_active(self) -> int:
        return len(self._active)


def transfer_on(env: simpy.Environment, resources: list[BandwidthResource], nbytes: float):
    """Occupy every resource for the payload duration; progress at min fair-share."""
    if nbytes <= 0:
        return
        yield  # make this a generator even if unused
    if not resources:
        return
        yield
    remaining = float(nbytes)
    tokens = [Occupancy(float(nbytes)) for _ in resources]
    for res, tok in zip(resources, tokens):
        res.join(tok)
    try:
        while remaining > 1e-9:
            share = min(res.share() for res in resources)
            if share <= 0:
                yield env.timeout(1.0)
                continue
            dt = remaining / share
            t0 = env.now
            change = simpy.AnyOf(env, [res.change_event() for res in resources])
            yield env.timeout(dt) | change
            remaining = max(0.0, remaining - share * (env.now - t0))
            for res in resources:
                res.advance()
    finally:
        for res, tok in zip(resources, tokens):
            res.leave(tok)


class CircularBuffer:
    def __init__(self, env: simpy.Environment, slots: int, slot_bytes: int, name: str):
        self.env = env
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.name = name
        self.filled = 0
        self.reserved = 0
        self._space = env.event()
        self._data = env.event()

    def free(self) -> int:
        return self.slots - self.filled - self.reserved

    def _wake(self, which: str) -> None:
        ev = self._space if which == "space" else self._data
        if not ev.triggered:
            ev.succeed()
        if which == "space":
            self._space = self.env.event()
        else:
            self._data = self.env.event()

    def reserve(self, n: int = 1):
        if n < 0:
            raise SimError(f"{self.name}: CB_RESERVE negative")
        while self.free() < n:
            yield self._space
        self.reserved += n
        if self.filled + self.reserved > self.slots:
            raise SimError(f"{self.name}: CB overflow")

    def push(self, n: int = 1) -> None:
        if n > self.reserved:
            raise SimError(f"{self.name}: CB_PUSH without reserve")
        self.reserved -= n
        self.filled += n
        self._wake("data")

    def wait(self, n: int = 1):
        if n < 0:
            raise SimError(f"{self.name}: CB_WAIT negative")
        while self.filled < n:
            yield self._data

    def pop(self, n: int = 1) -> None:
        if n > self.filled:
            raise SimError(f"{self.name}: CB underflow")
        self.filled -= n
        self._wake("space")


class Sram:
    def __init__(self, capacity: int, name: str):
        self.capacity = capacity
        self.name = name
        self.regions: list[tuple[int, int, str]] = []

    def allocate(self, label: str, addr: int, size: int) -> None:
        end = addr + size
        if addr < 0 or end > self.capacity:
            raise SimError(f"{self.name}: SRAM allocation {label} [{addr},{end}) exceeds {self.capacity}")
        for s, e, other in self.regions:
            if not (end <= s or addr >= e):
                raise SimError(f"{self.name}: SRAM overlap {label} [{addr},{end}) vs {other} [{s},{e})")
        self.regions.append((addr, end, label))

    def used(self) -> int:
        return sum(e - s for s, e, _ in self.regions)


class SemaphoreBank:
    def __init__(self, env: simpy.Environment, latency_cycles: int):
        self.env = env
        self.latency_cycles = latency_cycles
        self.values: dict[int, int] = {}
        self._events: dict[int, simpy.Event] = {}

    def _event(self, sid: int) -> simpy.Event:
        ev = self._events.get(sid)
        if ev is None or ev.triggered:
            ev = self.env.event()
            self._events[sid] = ev
        return ev

    def inc(self, sid: int, n: int = 1):
        yield self.env.timeout(self.latency_cycles)
        self.values[sid] = self.values.get(sid, 0) + n
        ev = self._events.get(sid)
        if ev is not None and not ev.triggered:
            ev.succeed()
            self._events[sid] = self.env.event()

    def wait(self, sid: int, n: int = 1):
        while self.values.get(sid, 0) < n:
            yield self._event(sid)


@dataclass
class StreamState:
    name: str
    pc: int = 0
    blocked_on: str | None = None
    done: bool = False
    commands: list = field(default_factory=list)
