from __future__ import annotations

from dataclasses import dataclass, field

from hw.schema import HwConfig


@dataclass(frozen=True)
class Track:
    id: str
    pid: int
    tid: int
    kind: str


@dataclass
class Layout:
    hw_name: str
    rows: int
    cols: int
    tracks: list[Track]
    by_id: dict[str, Track] = field(default_factory=dict)
    cores: list[dict] = field(default_factory=list)
    noc_links: list[dict] = field(default_factory=list)
    dram: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "hw_name": self.hw_name,
            "grid": {"rows": self.rows, "cols": self.cols},
            "cores": self.cores,
            "noc_links": self.noc_links,
            "dram": self.dram,
            "tracks": [
                {"id": t.id, "pid": t.pid, "tid": t.tid, "kind": t.kind} for t in self.tracks
            ],
        }

    def ids(self) -> set[str]:
        return {t.id for t in self.tracks}

    def core_pid(self, row: int, col: int) -> int:
        return 1 + row * self.cols + col


def core_pid(row: int, col: int, cols: int) -> int:
    return 1 + row * cols + col


def link_id(noc: int, src: tuple[int, int], dst: tuple[int, int]) -> str:
    r, c = src
    r2, c2 = dst
    return f"noc{noc}.link({r},{c})->({r2},{c2})"


def build_layout(hw: HwConfig, cb_names: dict[tuple[int, int], list[str]] | None = None) -> Layout:
    """Build layout.json. Track IDs are the Perfetto thread names.

    pid 0 = DRAM + NoC. Cores use pid = 1 + row*cols + col.
    """
    rows, cols = hw.grid.rows, hw.grid.cols
    tracks: list[Track] = []
    cores_json: list[dict] = []
    links_json: list[dict] = []
    dram_json: list[dict] = []

    system_tid = 1

    def add(track_id: str, pid: int, tid: int, kind: str) -> Track:
        t = Track(id=track_id, pid=pid, tid=tid, kind=kind)
        tracks.append(t)
        return t

    positions = hw.resolved_dram_positions()
    for ch, (r, c) in enumerate(positions):
        cid = f"dram.ch{ch}"
        add(cid, 0, system_tid, "dram")
        dram_json.append({"id": cid, "channel": ch, "row": r, "col": c})
        system_tid += 1

    for n in range(hw.noc.count):
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    for src, dst in (((r, c), (r, c + 1)), ((r, c + 1), (r, c))):
                        lid = link_id(n, src, dst)
                        add(lid, 0, system_tid, "noc")
                        links_json.append({"id": lid, "noc": n, "src": list(src), "dst": list(dst)})
                        system_tid += 1
                if r + 1 < rows:
                    for src, dst in (((r, c), (r + 1, c)), ((r + 1, c), (r, c))):
                        lid = link_id(n, src, dst)
                        add(lid, 0, system_tid, "noc")
                        links_json.append({"id": lid, "noc": n, "src": list(src), "dst": list(dst)})
                        system_tid += 1

    cb_names = cb_names or {}
    for r in range(rows):
        for c in range(cols):
            pid = core_pid(r, c, cols)
            tid = 1
            boxes = []

            def box(suffix: str, kind: str) -> str:
                nonlocal tid
                cid = f"core({r},{c}).{suffix}"
                add(cid, pid, tid, kind)
                boxes.append({"id": cid, "kind": kind})
                tid += 1
                return cid

            core_id = f"core({r},{c})"
            add(core_id, pid, tid, "core")
            tid += 1
            box("matrix", "matrix")
            box("vector", "vector")
            for dm in range(hw.core.dm_engines):
                box(f"dm{dm}", "dm")
            box("sram", "sram")
            cbs = []
            for name in cb_names.get((r, c), []):
                cid = box(f"cb.{name}", "cb")
                cbs.append(cid)
            cores_json.append({"id": core_id, "row": r, "col": c, "boxes": boxes, "cbs": cbs})

    layout = Layout(
        hw_name=hw.name,
        rows=rows,
        cols=cols,
        tracks=tracks,
        cores=cores_json,
        noc_links=links_json,
        dram=dram_json,
    )
    layout.by_id = {t.id: t for t in tracks}
    if len(layout.by_id) != len(tracks):
        raise AssertionError("duplicate layout component IDs")
    return layout
