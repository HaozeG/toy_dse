# accel-dse invariants

Event-level DSE for a Tenstorrent-style mesh (local SRAM, matrix + vector units, 2D NoC, edge DRAM). Read `accel-dse-plan.md` for the full contract. Do not reopen §0 decisions without a stated reason.

## Contracts

- The simulator executes `schedule.json` only. It never reads the model, HF config, or op graph.
- Numerical correctness lives only in `numpy_exec` (Phase 3+). The simulator uses byte widths and MAC counts.
- One Perfetto Chrome-JSON trace per run is the source of truth. Report, spatial visualizer, and sweep CSV are derived from it.
- Track name = `layout.json` component ID. Chrome `thread_name` metadata `args.name` equals that ID.
- pid 0 is reserved for DRAM channels and NoC links. Cores use `pid = 1 + row * cols + col`.
- Every result is labeled with the resolved `HwConfig` (preset + `--set` overrides) and derived peaks.
- A schema change and the JSON-schema export (`docs/hw.schema.json`, `docs/schedule.schema.json`) land in the same commit.
- Same config + schedule → byte-identical `trace.json`.
- Every CLI command accepts `--json`. Any assertion failure exits non-zero.

## Layout IDs

`core(r,c)`, `core(r,c).matrix`, `core(r,c).vector`, `core(r,c).dmN`, `core(r,c).sram`, `core(r,c).cb.<name>`, `nocN.link(r,c)->(r',c')`, `dram.chN`.

## Phases

Phase 0: analytical roofline. Phase 1: skeleton + tiny handwritten schedule. Do not start the next phase until that phase's acceptance tests pass.
