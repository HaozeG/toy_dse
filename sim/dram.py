from __future__ import annotations

import simpy

from hw.schema import HwConfig
from sim.resources import BandwidthResource, SimError


class DramChannel:
    def __init__(
        self,
        env: simpy.Environment,
        index: int,
        position: tuple[int, int],
        bytes_per_cycle: int,
        latency_cycles: int,
    ):
        self.index = index
        self.position = position
        self.latency_cycles = latency_cycles
        self.id = f"dram.ch{index}"
        self.bw = BandwidthResource(env, bytes_per_cycle, self.id)


class DramSystem:
    def __init__(self, env: simpy.Environment, hw: HwConfig):
        self.env = env
        self.channels: list[DramChannel] = []
        for i, pos in enumerate(hw.resolved_dram_positions()):
            self.channels.append(
                DramChannel(
                    env,
                    i,
                    (pos[0], pos[1]),
                    hw.dram.bytes_per_cycle_per_channel,
                    hw.dram.latency_cycles,
                )
            )
        self._by_id = {ch.id: ch for ch in self.channels}

    def lookup(self, name: str) -> DramChannel:
        if name not in self._by_id:
            raise SimError(f"unknown DRAM endpoint {name}")
        return self._by_id[name]
