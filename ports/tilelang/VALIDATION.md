# Initial validation — 2026-10-29

This records the first pilot's local validation. Current source reconciliation and
backend evidence are determined by `parity.py report`, not by this historical note.
Exact source fingerprints and environments are in `evidence/c.json` and
`evidence/metal.json`.

- Reference: current Nexus Rust source based on `c94e14bd1a8a20408a1fb159364b57a5347c0198`,
  Rust 1.99.1, standalone checked-in Cargo lockfile.
- Compiler: TileLang 0.1.15, TVM FFI 0.1.12, Python 3.13, NumPy 2.2.6.
- Host: macOS ARM64; GPU: Apple M4 Max.
- CPU generated C++: 17/17 portable replay cases passed.
- Native Metal: 17/17 portable replay cases passed.
- Maximum absolute differences on both backends: `9.5367431640625e-7` in impulse
  outputs and `4.76837158203125e-7` in velocities. All untouched words compared
  exactly; normal/friction/finite-value invariants passed.
- Change-tracking regressions: 12/12 passed.
- Replay corruption/race regressions: 7/7 passed, using actual exported buffers.
- Rust exporter: `cargo fmt --check` and Clippy with `-D warnings` passed.
- Python: Ruff F checks passed. Ruff is only a development formatter/linter,
  not a runtime or parity dependency.

## Deliberate drift experiment

Temporarily changed the actual Rust normal-impulse update from `cfm_factor * ...`
to `0.5 * cfm_factor * ...` in the worktree's `solver_utils.rs`.

1. The parity check failed, identified the Rust source change, and marked both
   accepted CPU and Metal reports stale.
2. Rebuilding the live Rust oracle and replaying TileLang failed numerically on
   `n1-b1-f1-i1-s20261030`.
3. Restoring the exact original bytes restored source reconciliation and both
   accepted reports. The reference binary was rebuilt from the restored source.

No production Rust solver changes remain in this milestone.

## Explicit failures and unavailable validation

The retained `--boundary-stress` diagnostic failed on
`n127-b1-f1-i1-s20261156`: six impulse outputs differed beyond tolerance, with a
maximum difference of about `0.0157851`. This is the documented cancellation at
zero distance selecting a different CFM branch. The portable suite tests exact
zero without rotated-point cancellation. The original diagnostic remains
reproducible, tolerances were not widened, and `check --release` correctly fails.

Source-lowering attempts with this macOS wheel could not validate the other
backends:

- CUDA: missing `tilelang.cuda._ffi_api.AnnotateDeviceBoundTmaCopies`.
- Ascend: `Target ascend is not supported`.

Neither is an execution result. CUDA and Ascend remain unvalidated in the parity
report. The Linux CPU CI workflow has been added but has not been run remotely.
Production runtime integration, same-GPU Rust-kernel comparison, full-scene tests,
Ascend toolkit compilation/execution, and performance measurements remain future
milestones.

## Native Rust execution milestone

The default Metal replay now uses the native Rust/khal adapter. All 17 portable
cases pass through its existing-device/queue/buffer API. Strict native Clippy
checks and two binding/dispatch contract tests pass; the 19 Python regressions
also pass. Execution is synchronous and not yet wired into the simulation step.

`numerics.py` runs the original production Rust Metal solver on the same input
buffers and records all three pairwise comparisons. The retained adversarial
suite has 3 cases passing all comparisons and 12 failing at least one comparison.
Two additional cases cannot be expressed by the production kernel's three-mode
interface and are reported unavailable, not passed. TileLang/Rust CPU disagree in
3 cases; TileLang/Rust Metal and Rust Metal/Rust CPU each disagree in 12.
`evidence/numerics.json` preserves these diagnostic results and exact input hashes.
The numerical gap is unresolved; no tolerance was widened and no Rust physics
was changed. This diagnostic is not accepted portable or release evidence.

## Hardware-free Ascend source milestone

Built TileLang commit `8944978dd8761eb975a28f53d2a9be4a800be241` with
`USE_ASCEND=ON`, `USE_CUDA=OFF`, `USE_ROCM=OFF`, `USE_METAL=ON` and runtime
stubs on this macOS host, without CANN or an NPU. All 17 solver specializations
lower successfully to Ascend source. The report records native compiler-library
hashes and version `0.1.15+git8944978d`. C/Metal replays still use the original
0.1.15 wheel and both pass 17/17 cases. These are distinct compiler environments.

`compiler.lock.json` pins the source and principal submodules; `compiler.py`
provides a dedicated-environment build recipe. Its record-only path was tested
against the actual build. The complete clean helper/CI build is not yet tested.
No vendor device binary was compiled. Ascend runtime, numerical correctness and
performance remain unvalidated. Source lowering cannot satisfy `--require ascend`.

## Mirrored solver-operation milestone

The shader tree now mirrors the corresponding Rust modules, including the split
between body integration arithmetic and solver-side clamping. Four separately
contracted operations pass 32/32 live-reference replay cases on generated C++ and
native Rust Metal; all 32 specializations also lower to Ascend source in the
separate pinned source-build environment. The 26 parity and actual-buffer guard
regressions pass, as do Rust exporter fmt/strict Clippy and Python Ruff F checks.
No global tolerance changed. Full scene stepping, complete contact preparation,
and production coloring remain unimplemented; the solve numerical release block
and missing Ascend execution remain explicit.

## Inventory, handoff and native contract checks

All 108 Rust shader modules have explicit dispositions and mirrored target paths;
19 capability groups keep remaining scope visible. The native binding manifest
now records a schema and exact specialization byte sizes. Native launch and fixture
loading reject mismatched sizes. Integration fixtures preserve the full parameters
actually used by their Rust oracle. Ambiguous duplicate evidence overrides fail.

C and native Metal pass all 32 cases; the pinned Ascend compiler lowers all 32.
The 41 Python regressions and three native contract tests pass; both Rust tools
pass strict Clippy. An actual immutable 32-case handoff was built and verified
from commit a828f306b (before the native size-contract fix), covering 297 files.
Returned-output round trips, corruption/missing files, path escape, and unavailable
NPU probes are tested. They are offline validation tests, not Ascend execution.

## Irregular integer kernel and numerical isolation

Added `dynamics/canonical_order.py::gpu_stabilize_body_constraint_ids`, calling the
original Rust GPU entry function on the host for its oracle. All six cases match
bit-for-bit on C and native Metal and lower to Ascend source. They cover empty and
skewed rows, high unsigned IDs, duplicates, workgroup boundaries and a 513-entry
row. This is evidence for one real indexed-read/variable-loop/scatter operation;
atomics, collective scans and BVH traversal remain unvalidated.

The five implemented operations now pass 38/38 C and native Metal cases and all
38 Ascend source lowerings. All 45 Python regressions pass. Three native contract
tests and both Rust tools' strict Clippy checks pass.

The fresh three-way diagnostic retains 12 failing boundary-stress cases, with
three passing and two unavailable modes. The portable diagnostic has nine failing
cases, six passing and two unavailable modes. A separate zero-frame diagnostic
has 15 passing and two unavailable modes: removing frame translations isolates
the zero-distance cancellation sensitivity. Every original portable Rust input and
expected-output binary across all 38 operation cases is byte-for-byte unchanged.
No solver arithmetic or tolerance changed, and neither failing suite was replaced.
The numerical policy remains unresolved and release-blocking. CPU/Metal development
checks pass; both the release gate and required Ascend execution gate fail as expected.

## Body lifecycle and repeated no-contact steps

Five more Rust entry points are mirrored in `dynamics/solver.py`: cleanup, solver
pose initialization, gravity increment initialization/application and finalization.
A separate composition contract reuses these and the existing clamped integrator
for no-contact steps. All 80 cases across eleven contracts pass on generated C++
and native Rust/khal Metal; all 80 lower to Ascend source. The 42 new cases include
zero batches, empty launches, multiple batches, extra capacity, partially locked
mass axes, static bodies with nonzero COM, zero gravity, velocity caps and up to
32 frames with four substeps. The largest absolute body-field difference from the
live Rust CPU oracle is `1.3113021850585938e-6` on both tested backends. Tolerances
remain `atol=2e-6`, `rtol=2e-5`. Copied fields, cleanup counters, padding and inactive
storage are checked bitwise. All 238 recorded original input/oracle binary files
from the earlier five contracts remain unchanged.

All 60 Python regressions and three native contract tests pass, along with Rust
fmt, strict Clippy, and Python Ruff F checks. Body handoff round trips exercise
the operation-specific comparator and reject missing, corrupt or oversized output.
Review also corrected launch metadata for standalone stages: 63 active bodies
with 65 slots use one group, while cleanup and the composition use two. The
previous extra groups were guarded and did not affect output correctness.

The composition fuses independent per-body operations; it does not validate
cross-body synchronization, contact scenes, or production pipeline integration.
World inertia updates and external force/torque accumulation remain unported.
This milestone's references are Rust CPU entry points, not additional production
Rust GPU comparisons. The retained contact numerical diagnostics are historical
and remain release-blocking; no numerical policy was changed. The development
CPU/Metal parity gate passes, while release and required-Ascend-execution gates
still reject the port as intended. Ascend vendor compilation, execution and
performance are unvalidated.


## World mass properties and collider propagation

The complete 3D `mprops_update.rs` module is now mirrored, including world-space
inverse inertia/COM computation and indexed collider world-pose propagation.
The body helper follows the pinned glam quaternion and matrix expressions. Tests
cover anisotropic, singular and zero inertias, rotated inertia frames, multiple
colliders per body, multi-batch parent indexing, empty inputs and dispatch tails.
Computed inertia matrices are checked for symmetry and positive semidefiniteness;
copied masses, homogeneous matrix constants, padding and untouched buffers remain
bitwise checked. Invalid parent indices, mass properties, layouts and output sizes
are rejected. Body, integer and pose replays now share one execution/provenance
implementation while retaining separate operation validators.

All 94 cases across thirteen contracts pass on C++ CPU and native Metal and lower
to Ascend source. Maximum new-operation differences from the live Rust CPU oracle
are `9.5367431640625e-7` for world mass properties and `2.384185791015625e-7` for
collider poses on both tested backends. No existing tolerance changed. All 73
Python regressions and three native tests pass, as do Rust formatting/strict
Clippy and Ruff F checks. The user explicitly deferred articulated multibodies,
MPM and 2D; these remain visible gaps rather than being marked implemented.

This is another implementation batch within the ongoing full-port goal. Contact
construction/persistence, complete scheduling, collisions, joints, native scene
integration and scene stability validation remain unfinished. The contact
rounding/branch sensitivity is still under investigation; this batch does not
clear it or claim Ascend device qualification.

## Coulomb contact preparation

`tile_kernels.py::gpu_prepare_constraints` now mirrors the complete 3D Coulomb
preparation entry point: current headers and mass scales, fresh manifold points,
nearest old-point impulse transfer, recycled anchors with refreshed restitution
and effective masses, and cached velocity deltas tagged with body IDs. Fresh
point restitution comes from the manifold and recycled restitution from the link,
matching the reference. Warmstart arithmetic is shared with the existing direct
warmstart operation and was revalidated for both uses.

The 21 added cases cover reordered points, nearest matches and strict ties,
rotated friction bases, no/empty old manifolds, threshold-adjacent anchors,
recycled point-count changes, both signs and zero in tangent construction, shared
readonly body inputs, split velocity slots, zero masses, inactive gaps, empty
launches and tile/workgroup boundaries. Copied impulses/distances/geometry and
integer tags are exact; only computed floats use the unchanged tolerances.
The empty launch specializes to a no-op before TileLang index analysis, which
otherwise fails on unreachable nested gather/matching loops.

All 115 cases across fourteen operations pass on generated C++ and native Rust
Metal; all 115 lower to Ascend source. The largest new-operation absolute error
is `5.960464477539063e-8` on both executed backends. All 88 Python regressions,
three native tests, Rust formatting/strict Clippy and Ruff F checks pass. The
handoff comparator exercises the new typed contract and rejects altered inputs,
copied fields, tags, padding, trailing storage and nonfinite outputs.

This implements preparation from existing links/manifolds, not collision
creation, recycle eligibility, warmstart gather or the complete scene schedule.
The full-port goal and the existing numerical-policy investigation continue.
Ascend vendor compilation, device execution and performance remain unvalidated.

## Contact-link persistence and color seeding

All 3D functions in `contact_links.rs` now have mirrored counterparts: link
initialization, identity matching in the shorter nonempty body adjacency list,
translation/rotation reuse decisions, state copying/refresh and previous-color
seeding. The Rust oracle calls all three entry points separately and in order;
a composed per-contact TileLang test uses immutable old topology and disjoint
old/new state halves. Production adjacency generation and changing-scene frame
stepping remain separate unfinished work.

The 36 new cases exercise 0/1/31/32/33/63/64/65/129 contacts, reverse adjacency
ordering and duplicate identities, missing matches, static/empty lists, stale
inactive IDs, high-bit multibody flags, zero/invalid previous colors, reversed
state-half ordering and exact state/padding copies. Recycling checks include
inclusive translation thresholds, excessive relative rotation, rigid motion that
preserves relative pose but exceeds the individual rotation limit, disabled
reuse and quaternion sign equivalence. All output words agree exactly with Rust.

Compiler-sensitive integer expressions are covered by the same cases. Unsigned
flag values are reduced to boolean zero tests before combination. Sentinel
checks precede address casts, avoiding a compiler-eliminated signed comparison.
The C backend's invalid constant bitcast is avoided by storing the exact f32-one
bits. These adaptations preserve native u32 data and do not alter Rust behavior.

All 151 cases across eighteen operation contracts pass on C++ CPU and native
Rust Metal; all 151 lower to Ascend source. All 103 Python regressions and three
native adapter tests pass, together with Rust formatting/strict Clippy and Ruff
F checks. Additional standalone generated-C++ checks at 65 and 129 contacts for
each of the four new operations passed with `-fsanitize=undefined` and
`-fno-sanitize-recover=all` (eight runs, outputs also checked exactly).
AddressSanitizer could not run on this host: a sampled process spun recursively
inside its shadow-memory/malloc initialization before `main` and was stopped.
That attempt is unavailable evidence, not a passed memory-safety check.

The full-port goal remains active. Warmstart gathering/scaling, production
scheduling, collision detection, joints, native scene integration and scene
stability tests remain unfinished. The existing floating branch-sensitivity
policy remains unresolved; Ascend vendor compilation/execution is unvalidated.

## Cached and split-slot warmstart

The mirrored `tile_kernels.py` now ports three more Coulomb entry points:
`gpu_scale_constraint_impulses`, `gpu_gather_warmstart_velocities` and
`gpu_hub_warmstart_constraints`. Native batch uniforms, buckets, adjacency
lists, ID/index indirection and hub counters drive their active ranges.
Scaling refreshes the cache and preserves Rust's constraint rewrite side effects:
inactive coupling and two model-specific header words are reset. Ordinary bodies
gather cached deltas in adjacency order; hub slots recompute the constraint delta
and then add it to the existing velocity. These are separate dispatch operations.

The 27 cases cover empty input, tile/workgroup boundaries, shared static/dynamic
bodies, both contact sides, reordered IDs and adjacency, two batches, zero/one/
fractional coefficients, zero-point contacts, skipped original hubs, requested
hub slots exceeding capacity, cache tags and padding. Cached values deliberately
differ from recomputed deltas. A finite cancellation-sensitive cache detects
accumulation reordering; it is an arithmetic regression, not a physical scene.
The new operation outputs agree exactly with live Rust in the tested environments;
the existing float tolerance policy is retained rather than assuming universal
bitwise equivalence.

Two compiler workarounds are explicit in the source. A one-element local buffer
for the cumulative adjacency start avoids invalid Metal source when a mutable
scalar receives a guarded load. Storing the mapped hub constraint index once also
avoids expensive repeated CPU interval analysis. The compiler-heavy attempt was
sampled and stopped; the final form passes all cases on each available target.

The complete registered suite now has 178 cases across twenty-one contracts.
All pass on generated C++ CPU and native Rust/khal Metal, and all lower to Ascend
source only. The 119 Python regressions and three native adapter tests pass,
as do Rust formatting/strict Clippy and Ruff F checks. Six extra C++ UBSan runs
(65 and 129 constraints for each new operation) pass with recovery disabled;
their output bytes also match the regular C++ replay already compared to Rust.

Production graph/hub construction, color dispatch, solver variants, collisions,
joints, native scene integration and physical stability evidence remain
unfinished. The earlier zero-distance branch-sensitivity release block remains.
Ascend vendor compilation, execution and performance remain unvalidated.
