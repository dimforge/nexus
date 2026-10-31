# Port scheduling audit — 2026-10-31

Changes 0134–0141 address the six open findings from the previous audit and
additional empty-color, refit and application-readback costs. Shader paths remain
mirrored to Rust. Machine-readable dispositions and reference pointers are in
`PERFORMANCE_AUDIT.json`; accepted receipts are in
`evidence/0142-metal-scheduling/`.

## Measured result

Same Apple M4 Max, debug testbed, 63,005-body Balls scene, 300,000 pair capacity,
600,000 contact capacity, four substeps, one PGS iteration and twist friction.
The initial color stride is 35 (33 usable colors). Rust comparison was disabled.
The 19 physical input buffers and simulation settings match exactly. These are
initial moving-scene frames, not a settled pile or a measurement against Rust.

| Measurement | Before (`a776b24d4`) | After (`7e2a03abb`) |
| --- | ---: | ---: |
| Actual testbed last step after six frames, including body publication | 27.911 ms | 5.792 ms |
| 30-frame compiled-stage replay, median after first two frames, first run order | 23.054 ms | 4.157 ms |
| Same replay, reversed run order | 23.039 ms | 4.054 ms |
| GPU time in those replays | 21.52–21.62 ms | 2.817–2.819 ms |

The testbed measurement improved about 4.8×; repeated execution-only medians
improved about 5.5–5.7×. The Swift replay uses the same compiled libraries,
capacity-gate submissions, tracked buffers, indirect grids and ownership swaps.
It excludes Rust bookkeeping and application readback. Diagnostic state hashing
is disabled for timing runs. Per-stage instrumented timings are not used as
whole-frame results. Preparation and rendering are excluded from step times;
final testbed preparation still took 78.85 seconds with warm caches.

These results do not establish that the user's reported 30× Rust gap is closed.
No current side-by-side Rust timing or settled dense-contact benchmark was run.

## Fixed in this pass

| Change | Cost removed | Reference policy / qualification |
| --- | --- | --- |
| 0134 | Capacity-wide independent collision, contact, adjacency and colored dispatches | Metal uses GPU-written live-count grids. Cooperative radix/scans retain static launch shapes. Empty key generation still publishes zero sort count. |
| 0135 | Six full-capacity cache copies at every frame | Commit swaps logical cache owners; checkpoint/growth/reset follow the owners. Contact state halves copy only live rows. Failed attempts never swap. |
| 0136 | Serial contact color feedback, unnecessary sort passes, unbounded adjacency group count | 64-lane feedback reduction; omit subshape for simple shapes and pack collider keys when the reserved sentinel permits; cap adjacency at 65,535 groups with grid-striding. |
| 0137 | Inactive GJK/EPA and LBVH search tails | Bounded data-dependent exits; Rust's own bounded split-search behavior is retained. Runtime packed-key shift is initialized from collider count. |
| 0138 | Encoding unused color stages, even when their indirect grid is zero | Reuse completed palette-capacity feedback to omit commit stages. No new readback; changing colors re-enable stages. |
| 0139 | Completed refit lanes waiting through all 64 levels | Metal workgroup-uniform completion vote, with shared barriers and existing device-memory fences. No divergent barrier exit. |
| 0140 | C scratch lowering and native color-condition metadata defects found during qualification | Serial C feedback uses local scratch; focused composition and selected eight-frame mesh/joint coverage. |
| 0141 | Per-record decoding of application body readback | Bulk copy aligned Pod records, preserving the unaligned fallback and checked body prefix. No readback-frequency or data-layout change. |

The final Balls specialization has 659 stages versus 706 initially, with 476
indirect descriptors. Of its 396 conditionally encoded stages, 384 are omitted
when the first six frames use only color 1. This is scene-dependent; stages become
executable again as the palette grows.

Changes 0131/0132 from the preceding pass already fixed contact key/permutation
under-parallelism, composite traversal/projection exits, reduction run detection,
warmstart search and coloring neighbor/conflict exits. Their evidence remains in
`evidence/0133-scheduling-audit/`; it is not relabeled as new evidence here.

## Portability and maintenance

Metal runtime specialization opts into native indirect grids and cache swaps.
CUDA/Ascend runtime scenes now use device-resident virtual grids and the same
swaps/live copies (0146), pending hardware qualification. C keeps its static
schedule. Native indirect launch is
not required for Ascend: the TileLang IR transformation follows khal's CUDA
virtual-workgroup loop, with a bounded physical grid, device-read virtual grid,
grid-striding and uniform workgroup barriers. It is implemented and tested on
Metal, but CUDA/Ascend synchronization remains unqualified. No CPU dispatch-size
readback was added. See `NATIVE_RUNTIME.md` for the frame schema and ownership contracts.

Stable feature IDs and separate change records classify scheduling changes,
qualification fixes and ABI extensions. Physics arithmetic, reference code,
comparison tolerances and solver settings were not changed to obtain the gains.

## Validation

- Final native runtime: 24 tests passed, including tracked/untracked indirect
  zero/changing grids, offsets, transactional swaps, growth, checkpoint/reset,
  color-stage conditions and aligned/unaligned readback. Rust fmt and runtime
  clippy passed; the feature-enabled testbed rebuilt successfully.
- Seven focused Metal schedule tests passed in 21.031 seconds, including fresh
  Rust feedback/sort/colored output, 65,537-body adjacency, exact live-row state
  copying and dispatch producers. The subsequent optional condition composition
  test passed separately. Two C feedback/sort tests passed in 21.046 seconds.
- Three focused bounded-exit tests passed in 61.751 seconds, including fresh Rust
  GJK/EPA and LBVH build/escape cases, geometric checks and termination assertions.
- Six fresh Rust chunk/frontier refit cases passed on final Metal sources in
  8.222 seconds, covering single leaves, duplicate keys and multiple batches.
- Selected eight-frame mesh/joint integration passed in 205.762 seconds: optimized
  and prior schedules have exact body poses, velocities, full trajectory,
  pair count, plan and cache counts. This is a scheduling comparison, not new
  full native-production scene qualification.
- Six Balls frames have exact poses, velocities, pair sets/counts, canonical
  links, live states, cache counts and color feedback. Solver tile comparison
  normalizes legal atomic slot allocation and excludes only unused twist arm
  fields beyond each live link's length. All active fields remain exact. Final
  readback optimization has identical shaders/manifests to this state replay.
- Final routine Metal smoke: **71 regressions and 11 representative physics
  cases passed in 238.766 seconds**, within the unchanged 290-second budget.
- Rust CPU fixtures use pinned 1.98.1 with mutable-reference noalias disabled.
  Full matrix and reconciliation were not run; the full baseline stays stale.

## Remaining limitations

- Small hub/round-completion kernels still use capacity grids; all-color fusion
  still uses specialization capacity rather than Rust's live pair heuristic.
  Existing capacity safety gates and application body publication still wait on
  the CPU. This audit is not a claim of performance equivalence.
- CUDA/Ascend execution, synchronization and the virtual-grid fallback remain
  unqualified. MPM, 2D and articulated multibodies remain deferred.
- The production-reference contact discrepancy recorded in change 0127 remains
  blocking. Exact old/new scheduling comparisons do not resolve that discrepancy.
- This is a targeted scheduling/host-cost audit, not an exhaustive arithmetic,
  concurrency, compiler or long-running stability audit.

Routine validation remains `python ports/tilelang/smoke.py --target metal` with
its unchanged 290-second budget. The focused eight-frame integration and geometry
regressions are explicit targeted tests, not additions to each routine smoke.

CUDA/Ascend bring-up and evidence limits are documented in [DEVICE_BRINGUP.md](DEVICE_BRINGUP.md).
