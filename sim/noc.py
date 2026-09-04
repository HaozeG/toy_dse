from __future__ import annotations

import simpy

from hw.layout import link_id
from hw.schema import HwConfig
from sim.resources import BandwidthResource


def xy_hops(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Dimension-order: column (X) first, then row (Y)."""
    r, c = src
    r2, c2 = dst
    hops: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while c != c2:
        nxt_c = c + (1 if c2 > c else -1)
        hops.append(((r, c), (r, nxt_c)))
        c = nxt_c
    while r != r2:
        nxt_r = r + (1 if r2 > r else -1)
        hops.append(((r, c), (nxt_r, c)))
        r = nxt_r
    return hops


class MeshNoc:
    def __init__(self, env: simpy.Environment, hw: HwConfig):
        self.env = env
        self.hw = hw
        self.hop_latency = hw.noc.hop_latency_cycles
        self.links: dict[tuple[int, tuple[int, int], tuple[int, int]], BandwidthResource] = {}
        rows, cols = hw.grid.rows, hw.grid.cols
        for n in range(hw.noc.count):
            for r in range(rows):
                for c in range(cols):
                    if c + 1 < cols:
                        self._add_link(n, (r, c), (r, c + 1))
                        self._add_link(n, (r, c + 1), (r, c))
                    if r + 1 < rows:
                        self._add_link(n, (r, c), (r + 1, c))
                        self._add_link(n, (r + 1, c), (r, c))

    def _add_link(self, noc: int, src: tuple[int, int], dst: tuple[int, int]) -> None:
        name = link_id(noc, src, dst)
        self.links[(noc, src, dst)] = BandwidthResource(self.env, self.hw.noc.link_bytes_per_cycle, name)

    def path_resources(self, noc: int, src: tuple[int, int], dst: tuple[int, int]) -> list[BandwidthResource]:
        hops = xy_hops(src, dst)
        missing = [(noc, a, b) for a, b in hops if (noc, a, b) not in self.links]
        if missing:
            raise KeyError(f"NoC path missing links {missing}")
        return [self.links[(noc, a, b)] for a, b in hops]

    def path_ids(self, noc: int, src: tuple[int, int], dst: tuple[int, int]) -> list[str]:
        return [link_id(noc, a, b) for a, b in xy_hops(src, dst)]
