# accel-dse: local environment for accelerator design space exploration (v2)

Plan for Claude Code. Read fully before starting Phase 0. Each phase ends with an acceptance test; do not start the next phase until it passes.

## 0. Decisions already made (do not re-open without a stated reason)

| Decision | Choice | Reason |
|---|---|---|
| Simulation level | Event-level (tile granularity), not RTL. Compute latency is deterministic from unit throughput; NoC links and DRAM channels are bandwidth-shared resources with contention. | The ONNXim modeling decision (Ham et al., IEEE CAL 2024): cores consume tiles from local SRAM with deterministic latency, so cycle-accurate compute changes cost, not rankings. The knobs we sweep are visible at this level. |
| Simulator core | Python, SimPy, owned by us. `hw/`, `sw/`, `sim/`, `trace/`, `viz/` are separate packages with JSON contracts between them. | Fast to modify by human or agent; decode of Qwen3-0.6B at tile granularity is 10^4–10^5 tasks per token, seconds in Python. Port hot paths to C++ only after a run exceeds 60 s. |
| Software stack exploration | The simulator executes `schedule.json`, a command-stream IR (§3.4). Runtime and compiler experiments are written as new mappers against that IR. Kernel/ISA-level software work is deferred to gvsoc (tier 2, §10). | Among surveyed simulators only gvsoc/gem5-class tools execute programs; the rest replay traces or graphs and cannot evaluate a new software stack. gvsoc: 79 commits / 5 authors in the last 12 months, Python-generated configs, Perfetto output. |
| pyCircuit / Agentic Circuit | Not a dependency. Watch list. | Active (629 commits since Feb 2026) but 7 months old, 63% single-author, unreleased contract 0.4, needs LLVM/MLIR 22, NPU example replays PTO-ISA traces with behavior in C++ providers. |
| ONNXim | Read the paper for the modeling decision; do not integrate. | 4 commits / 1 author in 12 months. |
| Hardware model | Tenstorrent-style grid (§2): cores with local SRAM, one matrix unit, one vector unit, data-movement engines; 2D mesh NoC; DRAM as N channels at mesh-edge positions reached over the NoC; core↔core only via NoC; semaphores for sync. | Requested: general design derived from Tenstorrent ideas, NoC-only transfers. |
| Software model | Per core, three command streams (reader, compute, writer) coupled by circular buffers (CBs). Commands: `NOC_READ`, `NOC_WRITE`, `NOC_MULTICAST`, `CB_RESERVE/CB_PUSH` (producer), `CB_WAIT/CB_POP` (consumer), `COMPUTE`, `SEM_INC`, `SEM_WAIT`. No runtime scheduler in the simulator. | The Tenstorrent kernel split; it is what "software-controlled scheduling and memory access" means concretely. |
| Workload | Qwen3-0.6B decode only: 1 token, KV length K ∈ {128, 256, 1024}, batch B ∈ {1, 4, 16} as a parameter (B>1 is where compute knobs become visible). Prefill deferred. | Requested. |
| Precision | int8 weights and activations (1 byte), fp32 accumulate. Simulator uses byte width only; switching to fp8 is a config change. Numerical check in fp32. | numpy has no fp8; int8 is the common edge-NPU choice; fp8 is the common datacenter-GPU choice. Either is one byte to the simulator. |
| KV cache | Lives in DRAM, read every token, sharded across channels like weights. Residency in SRAM is a later mapper option, not designed now. | Requested: optional, not determined. |
| Trace | One source of truth: Perfetto trace (Chrome JSON; protobuf later). Report, spatial visualizer, sweep CSV are derived from it. | Perfetto UI for timeline; `perfetto.TraceProcessor` SQL is the agent interface. |
| Spatial visualizer | Single HTML file (d3, no build) reading `layout.json` + `timeline.json`. Trace track names = layout component IDs. | Copies the tt-npe → ttnn-visualizer NPE data model (zones, transfers, congestion per timestep) without their TT-specific tooling. |
| Correctness | (a) structural assertions in the simulator: CB never over/underflows, SRAM allocation never exceeds capacity, every NoC read targets data that has been written, no deadlock; (b) `numpy_exec` runs the same `schedule.json` on a 2-layer random-weight Qwen3 and compares with HF forward in fp32. | (b) catches mapper bugs; it runs outside the simulator. |

## 1. Repository layout

```
accel-dse/
  CLAUDE.md
  pyproject.toml            # python >=3.11; simpy, pydantic, numpy, perfetto, transformers (config only), typer
  hw/
    schema.py               # HwConfig (pydantic) + JSON schema export
    presets/tiny.yaml       # 2x2 grid, 1 DRAM channel — CI target
    presets/grid8x8.yaml    # 8x8 grid, 8 DRAM channels on two edges
    layout.py               # HwConfig -> layout.json
  sw/
    workload/qwen3.py       # HF config -> decode op graph (shapes, dtypes, bytes, MACs)
    taskgraph.py            # ops -> tile tasks with explicit dependencies; K-split accumulate tasks explicit
    mapper/                 # tiling, core partition, DRAM sharding, CB depth -> schedule.json
    numpy_exec.py
  sim/
    resources.py            # BandwidthResource, Unit, Sram(allocator), CircularBuffer, Semaphore
    core.py                 # three command interpreters per core
    noc.py                  # mesh routing (dimension-order), per-link bandwidth, multicast
    dram.py                 # channels: position on mesh edge, bytes/cycle, latency
    engine.py
  trace/
    writer.py  analyze.py  to_timeline.py
  viz/index.html
  cli.py
  tests/  docs/
```

## 2. Hardware configuration interface

```yaml
name: grid8x8
clock_ghz: 1.0
grid: {rows: 8, cols: 8}
core:
  sram_bytes: 1572864
  matrix_unit: {macs_per_cycle: 1024, tile_shape: [32, 32, 32], count: 1}
  vector_unit: {lanes: 32, elem_bytes: 4, ops_per_cycle: 1, count: 1}
  dm_engines: 2                    # concurrent outstanding NoC transactions per core
  noc_read_bytes_per_cycle: 32     # per-core injection/ejection limit
noc:
  count: 2                         # independent NoCs; mapper assigns reads to noc0, writes to noc1 by default
  link_bytes_per_cycle: 32
  hop_latency_cycles: 2
  multicast: true
dram:
  channels: 8
  bytes_per_cycle_per_channel: 12  # 8 x 12 B/cycle = 96 GB/s at 1 GHz
  latency_cycles: 300
  positions: edge_left_right       # or explicit [[row, col], ...]
sync: {semaphore_latency_cycles: 8}
```

Rules:
- Every field has a default and a docstring; `dse hw explain <preset>` prints a table with derived peak numbers (aggregate DRAM B/cycle, bisection bandwidth, peak int8 TOPS, balance point in bytes/MAC).
- CLI overrides: `dse sim --hw grid8x8 --set dram.channels=4 --set grid.rows=4`.
- Resolved config and derived numbers go into trace metadata and the report.
- Sweeps are a YAML grid; `dse sweep grid.yaml` runs points in parallel and writes `results.csv` + one run directory per point.

## 3. Software side

### 3.1 Workload
`qwen3.py` builds the decode op graph from `Qwen3Config` (0.6B: 28 layers, hidden 1024, 16 Q / 8 KV heads, head_dim 128, intermediate 3072, vocab 151936; assert against the HF config at build time). Per token: embed lookup, 28 × {rmsnorm, q/k/v proj, rope, attention over K cached positions, o proj, rmsnorm, gate/up proj, silu·mul, down proj}, final norm, lm_head. Ops carry MACs, weight bytes, activation bytes, KV bytes.

### 3.2 Task graph
Tasks are `(op, tile_coords) -> inputs, outputs, unit, cost`. Matmul tiles align to `matrix_unit.tile_shape`; with B=1 the M dimension is padded to one tile row and the padding waste is reported as a number (`matrix_util_max`), not hidden. Reduction over K produces explicit partial-sum and accumulate tasks so core-to-core reductions are visible.

### 3.3 Mapper (must re-target per hardware config)
Inputs: task graph + HwConfig + mapper params. Output: `schedule.json`.
- Parameters: tile sizes per op class; core partition axis for each matmul (N across cores, K across cores with NoC reduction, heads across cores for attention); weight and KV sharding across DRAM channels (round-robin by tile, or by output column); CB depth per CB; NoC assignment (reads on noc0, writes on noc1); multicast for broadcast operands (activations shared by all cores).
- v1 heuristic: N-partition for projections and MLP, head-partition for attention, weights round-robin across channels so every channel is busy, CB depth 2, activations multicast from the producing core.
- v2: budgeted search over an aligned grid (≤64 points per hw config) with the simulator as cost function; report top-5 spread; if the top-N are within a few percent, say so and stop.
- The simulator only executes `schedule.json`; it never reads the model.

### 3.4 `schedule.json`
Per core, three arrays of commands with integer ids. Every `NOC_READ` names a source (`dram.chN + offset` or `core(r,c) + sram addr`), byte count, destination CB, and NoC id. Every `COMPUTE` names a kernel id, input CBs (with wait counts), output CBs, and cost `{macs, vector_ops}`. Semaphores are global integer ids. This file is the contract a future compiler targets; version it (`schedule_version`) and keep a JSON schema in `docs/`.

### 3.5 Correctness (`numpy_exec.py`)
Executes `schedule.json` on numpy in fp32 with a 2-layer random-weight Qwen3 and compares with `transformers` forward (atol 1e-4). Runs in CI on `tiny`.

## 4. Simulator

- `BandwidthResource`: fair share among active transfers; recomputed when the active set changes. Instances: each NoC link (per direction, per NoC), each DRAM channel, each core's injection port.
- NoC: dimension-order routing on the mesh; a transfer occupies every link on its path for its duration (coarse congestion model, same class as tt-npe). Multicast occupies the tree links once.
- `CircularBuffer`: fixed slot count; `CB_RESERVE` blocks when full, `CB_WAIT` blocks until the requested count is present. Over/underflow is a simulation error.
- `Sram`: static allocation from the mapper's slot plan; failure is an error, not a stall.
- Deadlock: no scheduled event with pending commands → dump per-core program counters and blocked-on reasons to the trace, exit non-zero.
- Stall reasons recorded as events: `wait_cb_full`, `wait_cb_empty`, `wait_noc`, `wait_dram`, `wait_sem`, `wait_unit`.

## 5. Trace contract

Chrome JSON; validated in tests by loading with `perfetto.TraceProcessor`.
- `pid` = core index (row*cols+col), pid 0 reserved for DRAM channels and NoC links. `tid` = component. Metadata names: `core(3,2).matrix`, `core(3,2).vector`, `core(3,2).dm0`, `core(3,2).sram`, `core(3,2).cb.<name>`, `noc0.link(3,2)->(3,3)`, `dram.ch5`. **Track name = `layout.json` component ID**; tested.
- Slices: compute tasks, NoC transfers, waits. `args`: task id, op, tile coords, bytes, achieved bytes/cycle, stall reason, NoC path.
- Counters: SRAM bytes in use per core, CB occupancy, per-link active transfers, per-channel bytes/cycle.
- `metadata`: resolved hw config, mapper params, derived peaks, workload params (K, B).
- Run directory: `runs/<id>/{trace.json, schedule.json, hw.json, report.md, report.json}`; immutable after creation.

## 6. Analysis and reports

SQL on TraceProcessor; the same queries serve `dse report` and ad-hoc agent questions.
- End-to-end cycles per token; per-component utilization (busy fraction) and occupation (achieved/peak while busy), as two columns.
- Critical path through the task graph using actual times.
- Stall attribution per core by reason.
- DRAM: per-channel bytes/cycle over time; `dram_saturation` = aggregate achieved / aggregate peak.
- NoC: per-link peak utilization; top-10 congested links.
- Roofline check: `max(MACs/peak_MACs, bytes/peak_DRAM)` vs simulated; ratio < 1 is a simulator bug and fails the run.
- For decode the headline number is **cores needed to saturate DRAM**: the smallest active-core count at which `dram_saturation` ≥ 0.9 for the given NoC/channel config. The sweep reports it directly.

## 7. Spatial + temporal visualizer

- `layout.json`: core grid by row/col; inside each core boxes for matrix, vector, dm engines, SRAM, CBs; NoC links between neighbors; DRAM channels at their edge positions.
- `timeline.json`: fixed-width time bins; per component: busy fraction, stall mix; per link: bytes and active count; per channel: bytes; per core: SRAM and CB occupancy. Includes bin → time mapping.
- Viewer: time scrubber, play, bin width, metric selector. Cores and units fill by metric; links widen by bytes; channels show saturation. Click a component to list its top slices in the bin and open a Perfetto deep link (`ui.perfetto.dev`, filtered to that track).
- `dse viz <run>` writes both JSONs and opens the HTML. The JSONs are also the agent's spatial view.

## 8. Agent ergonomics

- Every command has `--json`; non-zero exit on any assertion failure.
- Determinism: same config + schedule → byte-identical trace. Tests diff traces.
- `dse doctor`: checks deps, runs `tiny` end to end in <10 s, prints the roofline ratio.
- `CLAUDE.md` holds the invariants: track name = layout ID; simulator never reads the model; numerics only in `numpy_exec`; one trace per run is the source of truth; every result is labeled with its resolved config; schema change and JSON-schema export land in the same commit.
- `results.csv` columns: `cycles_per_token, roofline_ratio, dram_saturation, cores_to_saturate, matrix_util_mean, matrix_occ_mean, noc_peak_link_util, critical_path_len` plus every hw, mapper and workload parameter.

## 9. Phases and acceptance tests

**Phase 0 — Roofline (half a day).** From the HF config and an HwConfig print MACs, weight bytes, KV bytes, activation bytes and `max(compute, memory)` time per decode token for K ∈ {128, 256, 1024}, B ∈ {1, 4, 16}. Acceptance: table for `grid8x8` with the binding term named. Expected: B=1 memory-bound by ~2 orders of magnitude; B=16 moves toward balanced.

**Phase 1 — Skeleton (1 day).** Schema, presets, layout, `dse hw explain`, `dse doctor` with a hand-written schedule of one NoC read + one compute + one write on `tiny`. Trace loads in Perfetto UI and TraceProcessor. Acceptance: tests green; track names equal layout IDs.

**Phase 2 — Workload + task graph (1–2 days).** Acceptance: task MAC and byte totals equal Phase 0 within 0.1%; `matrix_util_max` reported for B=1.

**Phase 3 — Mapper v1 + simulator (4–5 days).** Three-stream cores, CBs, mesh NoC with multicast, DRAM channels, semaphores, stall reasons, deadlock detection. Acceptance: decode K=256, B=1 runs on `grid8x8`; roofline ratio ≥ 1 and ≤ 1.5; `numpy_exec` matches HF on the 2-layer model; a K-split reduction across ≥2 cores appears in the trace as core→core NoC traffic.

**Phase 4 — Report + sweep (1–2 days).** Grid: `grid ∈ {2x2, 4x4, 8x8}`, `dram.channels ∈ {1, 2, 4, 8}`, `noc.link_bytes_per_cycle ∈ {16, 32, 64}`, `B ∈ {1, 16}`. Acceptance: `results.csv`; for B=1 cycles are flat in grid size beyond `cores_to_saturate` and scale with `dram.channels` (state this expectation in the report before running); for B=16 matrix throughput appears.

**Phase 5 — Visualizer (2 days).** Acceptance: for the `grid8x8` run a person can point to the bin where a DRAM channel or edge link saturates and read the same bin from `timeline.json`.

**Phase 6 — Mapper v2 (2–3 days).** Search over sharding, partition axis, CB depth, tile size. Acceptance: report includes top-N spread per hw config and the decision (search or table).

**Phase 7 — Tier 2 (later).** For one chosen design point, lower `schedule.json` to gvsoc with a custom cluster model (its configs are Python-generated; matrix/vector units would be new C++ components) to run real kernels and a real runtime. Only when a question needs kernel-level fidelity.

## 10. Reuse and maintenance survey (measured 2026-09-04, shallow clones; stars not available)

| Source | Last commit | Commits / authors, 12 mo | Executes software? | Reuse |
|---|---|---|---|---|
| gvsoc (PULP, ETH) | 2026-08-28 | 79 / 5 | Yes (RISC-V binaries) | Tier-2 backend; Perfetto conventions |
| SST-core | 2026-08-11 | 180 / 13 | Framework | Alternative DES kernel if Python is too slow |
| tt-npe / ttnn-visualizer | 2026-08-31 | 34 / 8 | No (NoC workload JSON) | Congestion model class; spatial data model |
| PTO-ISA/pyCircuit AC | 2026-09-04 | 629 / 18 (63% one author) | No (PTO-ISA trace replay) | Watch; trace argument conventions |
| ASTRA-sim + Chakra | 2026-03-25 | 17 / 6 | No (execution traces) | Trace-schema ideas for multi-device later |
| ONNXim | 2026-01-08 | 4 / 1 | No | Modeling decision only |
| SCALE-Sim v3 | 2025-11-29 | 3 / 2 | No | — |
| Timeloop | 2025-06-09 | 0 / 0 | No (analytical) | Mapping-space vocabulary |
| Perfetto | active | — | — | Trace format, UI, SQL |
| HF transformers | active | — | — | `Qwen3Config`, reference forward |

## 11. Open questions before Phase 3

1. DRAM channel placement: fixed edge positions (Tenstorrent-like) or a free parameter? The default is edge_left_right; making it free adds one mapper axis.
2. Multicast: hardware feature (one NoC transaction, tree occupancy) or software loop of unicasts? Default: hardware, switchable.
3. Should the decode sweep include weight-resident-in-SRAM configurations for the small per-layer matrices (K/V proj are 1 MB each at int8), or leave all weights in DRAM for v1? Default: all in DRAM.
